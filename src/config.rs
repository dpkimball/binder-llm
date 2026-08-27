use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use crate::error::{LlmError, Result};
use crate::model::Provider;

/// Immutable client configuration. Construct via [`crate::LlmClient::builder`].
#[derive(Clone)]
pub struct LlmConfig {
    pub(crate) provider: Provider,
    pub(crate) api_key: Option<String>,
    pub(crate) base_url: String,
    pub(crate) model: String,
    pub(crate) organization: Option<String>,
    pub(crate) timeout: Duration,
    pub(crate) retries: RetryPolicy,
    pub(crate) headers: BTreeMap<String, String>,
    pub(crate) user_agent: String,
}

impl LlmConfig {
    pub fn provider(&self) -> Provider {
        self.provider
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn organization(&self) -> Option<&str> {
        self.organization.as_deref()
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    pub fn retries(&self) -> RetryPolicy {
        self.retries
    }

    pub fn headers(&self) -> &BTreeMap<String, String> {
        &self.headers
    }

    pub fn user_agent(&self) -> &str {
        &self.user_agent
    }

    pub(crate) fn api_key(&self) -> Option<&str> {
        self.api_key.as_deref().filter(|k| !k.is_empty())
    }
}

impl fmt::Debug for LlmConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmConfig")
            .field("provider", &self.provider)
            .field("api_key", &self.api_key.as_ref().map(|_| "[redacted]"))
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("organization", &self.organization)
            .field("timeout", &self.timeout)
            .field("retries", &self.retries)
            .field("headers", &redact_headers(&self.headers))
            .field("user_agent", &self.user_agent)
            .finish()
    }
}

fn redact_headers(headers: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    headers
        .iter()
        .map(|(k, v)| {
            let key_l = k.to_ascii_lowercase();
            let value = if key_l.contains("auth")
                || key_l.contains("key")
                || key_l.contains("token")
                || key_l.contains("secret")
            {
                "[redacted]".to_string()
            } else {
                v.clone()
            };
            (k.clone(), value)
        })
        .collect()
}

/// Exponential backoff for infrastructure retries only (no prompt replay logic).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    pub multiplier: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(200),
            max_backoff: Duration::from_secs(8),
            multiplier: 2.0,
        }
    }
}

impl RetryPolicy {
    pub fn backoff_for_attempt(self, attempt: u32) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }
        let factor = self.multiplier.powi(attempt.saturating_sub(1) as i32);
        let millis = (self.initial_backoff.as_secs_f64() * 1000.0 * factor) as u64;
        Duration::from_millis(millis.min(self.max_backoff.as_millis() as u64))
    }
}

/// Fluent builder for [`crate::LlmClient`].
#[derive(Debug, Clone, Default)]
pub struct LlmClientBuilder {
    provider: Option<Provider>,
    api_key: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    organization: Option<String>,
    timeout: Option<Duration>,
    retries: Option<RetryPolicy>,
    headers: BTreeMap<String, String>,
    user_agent: Option<String>,
}

impl LlmClientBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn provider(mut self, provider: Provider) -> Self {
        self.provider = Some(provider);
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn organization(mut self, organization: impl Into<String>) -> Self {
        self.organization = Some(organization.into());
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn retries(mut self, retries: RetryPolicy) -> Self {
        self.retries = Some(retries);
        self
    }

    pub fn header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(name.into(), value.into());
        self
    }

    pub fn user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = Some(user_agent.into());
        self
    }

    pub fn build_config(self) -> Result<LlmConfig> {
        let provider = self
            .provider
            .ok_or_else(|| LlmError::invalid("provider is required"))?;
        let model = self
            .model
            .filter(|m| !m.is_empty())
            .ok_or_else(|| LlmError::invalid("model is required"))?;
        let api_key = self.api_key.filter(|k| !k.is_empty());
        if provider.requires_api_key() && api_key.is_none() {
            return Err(LlmError::invalid(format!(
                "api_key is required for provider {}",
                provider.as_str()
            )));
        }
        let base_url = self
            .base_url
            .filter(|u| !u.is_empty())
            .unwrap_or_else(|| provider.default_base_url().to_string());
        Ok(LlmConfig {
            provider,
            api_key,
            base_url,
            model,
            organization: self.organization.filter(|s| !s.is_empty()),
            timeout: self.timeout.unwrap_or(Duration::from_secs(60)),
            retries: self.retries.unwrap_or_default(),
            headers: self.headers,
            user_agent: self
                .user_agent
                .unwrap_or_else(|| format!("binder-llm/{}", env!("CARGO_PKG_VERSION"))),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requires_provider_and_model() {
        let err = LlmClientBuilder::new().build_config().unwrap_err();
        assert!(matches!(err, LlmError::InvalidRequest { .. }));
    }

    #[test]
    fn openai_requires_key() {
        let err = LlmClientBuilder::new()
            .provider(Provider::Openai)
            .model("gpt-4.1")
            .build_config()
            .unwrap_err();
        assert!(matches!(err, LlmError::InvalidRequest { .. }));
    }

    #[test]
    fn debug_redacts_key() {
        let cfg = LlmClientBuilder::new()
            .provider(Provider::Ollama)
            .model("qwen3-coder:30b")
            .api_key("secret-value-xyz")
            .build_config()
            .unwrap();
        let rendered = format!("{cfg:?}");
        assert!(!rendered.contains("secret-value-xyz"));
        assert!(rendered.contains("[redacted]"));
    }

    #[test]
    fn backoff_grows() {
        let p = RetryPolicy::default();
        assert_eq!(p.backoff_for_attempt(0), Duration::ZERO);
        assert!(p.backoff_for_attempt(2) > p.backoff_for_attempt(1));
    }
}
