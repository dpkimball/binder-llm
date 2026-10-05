use std::time::Duration;

use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use reqwest::{Client, Method, RequestBuilder};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::config::{LlmConfig, RetryPolicy};
use crate::error::{sanitize_error_message, LlmError, Result};
use crate::model::Provider;

pub struct HttpTransport {
    client: Client,
    config: LlmConfig,
}

impl HttpTransport {
    pub fn new(config: LlmConfig) -> Result<Self> {
        let client = Client::builder()
            .user_agent(&config.user_agent)
            .connect_timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| LlmError::connection(e.to_string()))?;
        Ok(Self { client, config })
    }

    pub fn config(&self) -> &LlmConfig {
        &self.config
    }

    pub fn provider(&self) -> Provider {
        self.config.provider
    }

    pub fn url(&self, path: &str) -> String {
        join_url(&self.config.base_url, path)
    }

    fn base_headers(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        for (name, value) in &self.config.headers {
            let n = HeaderName::from_bytes(name.as_bytes())
                .map_err(|e| LlmError::invalid(format!("invalid header name: {e}")))?;
            let v = HeaderValue::from_str(value)
                .map_err(|e| LlmError::invalid(format!("invalid header value: {e}")))?;
            headers.insert(n, v);
        }
        Ok(headers)
    }

    pub fn request(&self, method: Method, url: &str) -> Result<RequestBuilder> {
        let mut builder = self.client.request(method, url);
        builder = builder.headers(self.base_headers()?);
        if let Some(key) = self.config.api_key() {
            match self.config.provider {
                Provider::Anthropic => {
                    builder = builder.header("x-api-key", key);
                    builder = builder.header("anthropic-version", "2023-06-01");
                }
                Provider::Gemini => {
                    builder = builder.header("x-goog-api-key", key);
                }
                Provider::Ollama => {
                    builder = builder.bearer_auth(key);
                }
                Provider::Openai | Provider::Xai => {
                    builder = builder.bearer_auth(key);
                }
            }
        } else if self.config.provider == Provider::Anthropic {
            builder = builder.header("anthropic-version", "2023-06-01");
        }
        if let Some(org) = &self.config.organization {
            if matches!(self.config.provider, Provider::Openai) {
                builder = builder.header("OpenAI-Organization", org);
            }
        }
        Ok(builder)
    }

    pub async fn post_json<B, T>(&self, path: &str, body: &B) -> Result<HttpJson<T>>
    where
        B: Serialize,
        T: DeserializeOwned,
    {
        let url = self.url(path);
        let bytes = serde_json::to_vec(body).map_err(|e| LlmError::serialization(e.to_string()))?;
        let (status, raw, retries) = self
            .send_bytes_retry(Method::POST, &url, Some(bytes), true)
            .await?;
        let parsed = parse_json_body::<T>(self.provider(), status, &raw)?;
        Ok(HttpJson {
            body: parsed,
            retries,
        })
    }

    pub async fn get_json<T>(&self, path: &str) -> Result<HttpJson<T>>
    where
        T: DeserializeOwned,
    {
        let url = self.url(path);
        let (status, raw, retries) = self.send_bytes_retry(Method::GET, &url, None, true).await?;
        let parsed = parse_json_body::<T>(self.provider(), status, &raw)?;
        Ok(HttpJson {
            body: parsed,
            retries,
        })
    }

    /// POST JSON and return the raw response for streaming. Retries only the
    /// initial handshake (429/5xx) before bytes are consumed.
    pub async fn post_stream(&self, path: &str, body: &impl Serialize) -> Result<StreamStart> {
        let url = self.url(path);
        self.post_stream_url(&url, body).await
    }

    pub async fn post_stream_url(&self, url: &str, body: &impl Serialize) -> Result<StreamStart> {
        let bytes = serde_json::to_vec(body).map_err(|e| LlmError::serialization(e.to_string()))?;
        let policy = self.config.retries;
        let mut retries = 0u32;
        loop {
            let mut builder = self.request(Method::POST, url)?;
            builder = builder
                .header(reqwest::header::CONTENT_TYPE, "application/json")
                .body(bytes.clone());
            match builder.send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        return Ok(StreamStart {
                            response: resp,
                            retries,
                        });
                    }
                    let code = status.as_u16();
                    let raw = resp.text().await.unwrap_or_default();
                    let err = error_from_body(self.provider(), code, &raw);
                    if retries < policy.max_retries && LlmError::is_retryable_status(code) {
                        retries += 1;
                        sleep_backoff(policy, retries, &err).await;
                        continue;
                    }
                    return Err(err);
                }
                Err(e) => return Err(map_reqwest(e)),
            }
        }
    }

    async fn send_bytes_retry(
        &self,
        method: Method,
        url: &str,
        body: Option<Vec<u8>>,
        apply_timeout: bool,
    ) -> Result<(u16, String, u32)> {
        let policy = self.config.retries;
        let mut retries = 0u32;
        loop {
            let mut builder = self.request(method.clone(), url)?;
            if apply_timeout {
                builder = builder.timeout(self.config.timeout);
            }
            if let Some(ref bytes) = body {
                builder = builder
                    .header(reqwest::header::CONTENT_TYPE, "application/json")
                    .body(bytes.clone());
            }
            match builder.send().await {
                Ok(resp) => {
                    let status = resp.status();
                    let raw = resp.text().await.unwrap_or_default();
                    let code = status.as_u16();
                    if status.is_success() {
                        return Ok((code, raw, retries));
                    }
                    let err = error_from_body(self.provider(), code, &raw);
                    if retries < policy.max_retries && LlmError::is_retryable_status(code) {
                        retries += 1;
                        sleep_backoff(policy, retries, &err).await;
                        continue;
                    }
                    return Err(err);
                }
                Err(e) => return Err(map_reqwest(e)),
            }
        }
    }
}

pub struct HttpJson<T> {
    pub body: T,
    pub retries: u32,
}

pub struct StreamStart {
    pub response: reqwest::Response,
    pub retries: u32,
}

pub fn join_url(base: &str, path: &str) -> String {
    let base = base.trim_end_matches('/');
    let path = path.trim_start_matches('/');
    if path.is_empty() {
        base.to_string()
    } else {
        format!("{base}/{path}")
    }
}

pub fn map_reqwest(err: reqwest::Error) -> LlmError {
    let err = err.without_url();
    if err.is_timeout() {
        LlmError::Timeout
    } else if err.is_body() || err.is_decode() {
        LlmError::serialization(sanitize_error_message(err.to_string()))
    } else {
        LlmError::connection(sanitize_error_message(err.to_string()))
    }
}

fn parse_json_body<T: DeserializeOwned>(provider: Provider, status: u16, raw: &str) -> Result<T> {
    serde_json::from_str(raw).map_err(|e| {
        if raw.trim().is_empty() {
            LlmError::provider(provider, "empty response body", Some(status))
        } else {
            LlmError::serialization(sanitize_error_message(format!(
                "failed to decode provider JSON: {e}"
            )))
        }
    })
}

pub fn error_from_body(provider: Provider, status: u16, raw: &str) -> LlmError {
    let message = extract_provider_error_message(raw)
        .unwrap_or_else(|| sanitize_error_message(raw.to_string()));
    let mut err = LlmError::from_status(provider, status, message);
    if let LlmError::RateLimited { retry_after } = &mut err {
        *retry_after = parse_retry_after_from_json(raw);
    }
    err
}

fn extract_provider_error_message(raw: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(raw).ok()?;
    if let Some(msg) = value.pointer("/error/message").and_then(|v| v.as_str()) {
        return Some(sanitize_error_message(msg.to_string()));
    }
    if let Some(msg) = value.get("error").and_then(|v| v.as_str()) {
        return Some(sanitize_error_message(msg.to_string()));
    }
    if let Some(msg) = value.get("message").and_then(|v| v.as_str()) {
        return Some(sanitize_error_message(msg.to_string()));
    }
    None
}

fn parse_retry_after_from_json(raw: &str) -> Option<Duration> {
    let value: serde_json::Value = serde_json::from_str(raw).ok()?;
    value
        .pointer("/error/retry_after")
        .and_then(|v| v.as_u64())
        .map(Duration::from_secs)
}

async fn sleep_backoff(policy: RetryPolicy, attempt: u32, err: &LlmError) {
    let mut delay = policy.backoff_for_attempt(attempt);
    if let LlmError::RateLimited {
        retry_after: Some(after),
    } = err
    {
        delay = (*after).min(policy.max_backoff);
    }
    tracing::info!(
        attempt,
        delay_ms = delay.as_millis() as u64,
        error_kind = err.kind(),
        "llm.retry"
    );
    tokio::time::sleep(delay).await;
}

/// Split an HTTP body into lines (NDJSON / SSE).
pub fn response_lines(
    response: reqwest::Response,
) -> impl futures::Stream<Item = Result<String>> + Send {
    let stream = response.bytes_stream();
    futures::stream::unfold(
        (Box::pin(stream), Vec::<u8>::new(), false),
        |(mut stream, mut buf, done)| async move {
            if done {
                return None;
            }
            loop {
                if let Some(pos) = buf.iter().position(|b| *b == b'\n') {
                    let mut line: Vec<u8> = buf.drain(..=pos).collect();
                    if line.last() == Some(&b'\n') {
                        line.pop();
                    }
                    if line.last() == Some(&b'\r') {
                        line.pop();
                    }
                    let s = String::from_utf8_lossy(&line).into_owned();
                    return Some((Ok(s), (stream, buf, false)));
                }
                match stream.next().await {
                    Some(Ok(chunk)) => buf.extend_from_slice(&chunk),
                    Some(Err(e)) => {
                        return Some((Err(map_reqwest(e)), (stream, buf, true)));
                    }
                    None => {
                        if buf.is_empty() {
                            return None;
                        }
                        let s = String::from_utf8_lossy(&buf).into_owned();
                        buf.clear();
                        return Some((Ok(s), (stream, buf, true)));
                    }
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_error_body_drops_credentials() {
        let err = error_from_body(
            Provider::Openai,
            401,
            r#"{"error":{"message":"invalid Bearer sk-liveSECRETvalue123"}}"#,
        );
        let text = err.to_string();
        assert!(!text.contains("sk-liveSECRETvalue123"));
        assert!(!text.contains("liveSECRET"));
        assert!(text.contains("[redacted]"));
    }

    #[test]
    fn join_url_strips_slashes() {
        assert_eq!(
            join_url("http://localhost:11434/", "/api/chat"),
            "http://localhost:11434/api/chat"
        );
    }
}
