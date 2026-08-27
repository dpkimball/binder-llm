use async_trait::async_trait;
use futures::StreamExt;

use crate::config::LlmConfig;
use crate::error::{LlmError, Result};
use crate::http::{response_lines, HttpTransport};
use crate::model::{ModelInfo, Provider};
use crate::provider::LlmProvider;
use crate::request::{ChatRequest, EmbeddingRequest};
use crate::response::{ChatResponse, EmbeddingResponse, FinishReason};
use crate::stream::{ChatStream, StreamChunk};
use crate::usage::Usage;

use super::mapper::{
    embeddings_body, responses_body, EmbeddingsApi, ModelsApi, ResponsesApi, SseEventPayload,
};
use super::models::model_info;

pub struct OpenAiClient {
    http: HttpTransport,
}

impl OpenAiClient {
    pub fn new(config: LlmConfig) -> Result<Self> {
        Ok(Self {
            http: HttpTransport::new(config)?,
        })
    }

    fn default_model(&self) -> &str {
        self.http.config().model()
    }
}

#[async_trait]
impl LlmProvider for OpenAiClient {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let model = request.resolved_model(self.default_model()).to_string();
        let body = responses_body(&request, self.default_model())?;
        let json = self
            .http
            .post_json::<_, ResponsesApi>("responses", &body)
            .await?;
        json.body.into_chat_response(&model, json.retries)
    }

    async fn stream(&self, mut request: ChatRequest) -> Result<ChatStream> {
        request.stream = true;
        let model = request.resolved_model(self.default_model()).to_string();
        let body = responses_body(&request, self.default_model())?;
        let start = self.http.post_stream("responses", &body).await?;
        Ok(ChatStream::from_stream(sse_to_chunks(
            response_lines(start.response),
            model,
            Provider::Openai,
        )))
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let model = request.resolved_model(self.default_model()).to_string();
        let inputs = request.input.into_texts();
        if inputs.is_empty() {
            return Err(LlmError::invalid("embedding request has no input"));
        }
        let body = embeddings_body(&model, &inputs);
        let json = self
            .http
            .post_json::<_, EmbeddingsApi>("embeddings", &body)
            .await?;
        let embeddings = json.body.data.into_iter().map(|d| d.embedding).collect();
        let usage = json
            .body
            .usage
            .map(|u| Usage::new(u.input_tokens.or(u.prompt_tokens).unwrap_or(0), 0))
            .unwrap_or_default();
        Ok(EmbeddingResponse {
            embeddings,
            model: json.body.model.unwrap_or(model),
            provider: Provider::Openai,
            usage,
            latency: std::time::Duration::ZERO,
            retries: json.retries,
        })
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let json = self.http.get_json::<ModelsApi>("models").await?;
        Ok(json
            .body
            .data
            .into_iter()
            .map(|m| model_info(m.id))
            .collect())
    }
}

pub(crate) fn sse_to_chunks(
    lines: impl futures::Stream<Item = Result<String>> + Send + 'static,
    model: String,
    provider: Provider,
) -> impl futures::Stream<Item = Result<StreamChunk>> + Send {
    futures::stream::unfold(
        SseState {
            lines: Box::pin(lines),
            data: String::new(),
            event: String::new(),
            finished: false,
            model,
            provider,
        },
        |mut state| async move {
            if state.finished {
                return None;
            }
            loop {
                match state.lines.next().await {
                    Some(Ok(line)) => {
                        if line.is_empty() {
                            if let Some(chunk) = state.flush_event() {
                                return Some((chunk, state));
                            }
                            continue;
                        }
                        if let Some(rest) = line.strip_prefix("event:") {
                            state.event = rest.trim().to_string();
                        } else if let Some(rest) = line.strip_prefix("data:") {
                            if !state.data.is_empty() {
                                state.data.push('\n');
                            }
                            state.data.push_str(rest.trim_start());
                        }
                    }
                    Some(Err(e)) => {
                        state.finished = true;
                        return Some((Err(e), state));
                    }
                    None => {
                        if let Some(chunk) = state.flush_event() {
                            state.finished = true;
                            return Some((chunk, state));
                        }
                        return None;
                    }
                }
            }
        },
    )
}

struct SseState {
    lines: std::pin::Pin<Box<dyn futures::Stream<Item = Result<String>> + Send>>,
    data: String,
    event: String,
    finished: bool,
    model: String,
    provider: Provider,
}

impl SseState {
    fn flush_event(&mut self) -> Option<Result<StreamChunk>> {
        if self.data.is_empty() {
            self.event.clear();
            return None;
        }
        let data = std::mem::take(&mut self.data);
        let event = std::mem::take(&mut self.event);
        if data.trim() == "[DONE]" {
            self.finished = true;
            return Some(Ok(StreamChunk::Finished {
                reason: FinishReason::Stop,
                message: None,
            }));
        }
        let payload: SseEventPayload = match serde_json::from_str(&data) {
            Ok(v) => v,
            Err(e) => {
                return Some(Err(LlmError::serialization(format!(
                    "invalid openai sse JSON: {e}"
                ))))
            }
        };
        let kind = if event.is_empty() {
            payload.r#type.as_deref().unwrap_or("")
        } else {
            event.as_str()
        };
        match kind {
            "response.output_text.delta" | "response.text.delta" => {
                let text = match &payload.delta {
                    Some(Value::String(s)) => s.clone(),
                    Some(other) => other
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    None => String::new(),
                };
                if text.is_empty() {
                    None
                } else {
                    Some(Ok(StreamChunk::ContentDelta { text }))
                }
            }
            "response.function_call_arguments.delta" => {
                let delta = payload
                    .delta
                    .as_ref()
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                Some(Ok(StreamChunk::ToolCallDelta {
                    index: 0,
                    id: None,
                    name: None,
                    arguments_delta: Some(delta),
                }))
            }
            "response.completed" => {
                self.finished = true;
                if let Some(resp) = payload.response {
                    match resp.into_chat_response(&self.model, 0) {
                        Ok(chat) => Some(Ok(StreamChunk::Finished {
                            reason: chat.finish_reason,
                            message: Some(chat.message),
                        })),
                        Err(e) => Some(Err(e)),
                    }
                } else {
                    Some(Ok(StreamChunk::Finished {
                        reason: FinishReason::Stop,
                        message: None,
                    }))
                }
            }
            "response.failed" => {
                self.finished = true;
                Some(Err(LlmError::provider(
                    self.provider,
                    "openai stream failed",
                    None,
                )))
            }
            _ => None,
        }
    }
}

use serde_json::Value;
