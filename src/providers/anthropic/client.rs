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

use super::mapper::{messages_body, ModelsApi, StreamEvent};
use super::models::model_info;

pub struct AnthropicClient {
    http: HttpTransport,
}

impl AnthropicClient {
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
impl LlmProvider for AnthropicClient {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let model = request.resolved_model(self.default_model()).to_string();
        let body = messages_body(&request, self.default_model())?;
        let json = self
            .http
            .post_json::<_, super::mapper::MessagesApi>("v1/messages", &body)
            .await?;
        json.body.into_chat_response(&model, json.retries)
    }

    async fn stream(&self, mut request: ChatRequest) -> Result<ChatStream> {
        request.stream = true;
        let body = messages_body(&request, self.default_model())?;
        let start = self.http.post_stream("v1/messages", &body).await?;
        Ok(ChatStream::from_stream(anthropic_sse(response_lines(
            start.response,
        ))))
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        Err(LlmError::unsupported("anthropic embeddings"))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let json = self.http.get_json::<ModelsApi>("v1/models").await?;
        Ok(json
            .body
            .data
            .into_iter()
            .map(|m| {
                let mut info = model_info(&m.id);
                if let Some(name) = m.display_name {
                    info.display_name = name;
                }
                info
            })
            .collect())
    }
}

fn anthropic_sse(
    lines: impl futures::Stream<Item = Result<String>> + Send + 'static,
) -> impl futures::Stream<Item = Result<StreamChunk>> + Send {
    futures::stream::unfold(
        AnthSse {
            lines: Box::pin(lines),
            data: String::new(),
            finished: false,
        },
        |mut state| async move {
            if state.finished {
                return None;
            }
            loop {
                match state.lines.next().await {
                    Some(Ok(line)) => {
                        if line.is_empty() {
                            if let Some(chunk) = state.flush() {
                                return Some((chunk, state));
                            }
                            continue;
                        }
                        if let Some(rest) = line.strip_prefix("data:") {
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
                        if let Some(chunk) = state.flush() {
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

struct AnthSse {
    lines: std::pin::Pin<Box<dyn futures::Stream<Item = Result<String>> + Send>>,
    data: String,
    finished: bool,
}

impl AnthSse {
    fn flush(&mut self) -> Option<Result<StreamChunk>> {
        if self.data.is_empty() {
            return None;
        }
        let data = std::mem::take(&mut self.data);
        let event: StreamEvent = match serde_json::from_str(&data) {
            Ok(v) => v,
            Err(e) => {
                return Some(Err(LlmError::serialization(format!(
                    "invalid anthropic sse JSON: {e}"
                ))))
            }
        };
        match event.kind.as_deref() {
            Some("content_block_delta") => {
                if let Some(delta) = event.delta {
                    if let Some(text) = delta.text {
                        return Some(Ok(StreamChunk::ContentDelta { text }));
                    }
                    if let Some(partial) = delta.partial_json {
                        return Some(Ok(StreamChunk::ToolCallDelta {
                            index: event.index.unwrap_or(0),
                            id: None,
                            name: None,
                            arguments_delta: Some(partial),
                        }));
                    }
                }
                None
            }
            Some("content_block_start") => {
                if let Some(block) = event.content_block {
                    if block.kind.as_deref() == Some("tool_use") {
                        return Some(Ok(StreamChunk::ToolCallDelta {
                            index: event.index.unwrap_or(0),
                            id: block.id,
                            name: block.name,
                            arguments_delta: None,
                        }));
                    }
                }
                None
            }
            Some("message_delta") => {
                if let Some(usage) = event.usage {
                    return Some(Ok(StreamChunk::Usage(Usage::new(
                        usage.input_tokens.unwrap_or(0),
                        usage.output_tokens.unwrap_or(0),
                    ))));
                }
                if let Some(reason) = event.delta.and_then(|d| d.stop_reason) {
                    return Some(Ok(StreamChunk::Finished {
                        reason: FinishReason::from_provider_str(&reason),
                        message: None,
                    }));
                }
                None
            }
            Some("message_stop") => {
                self.finished = true;
                Some(Ok(StreamChunk::Finished {
                    reason: FinishReason::Stop,
                    message: None,
                }))
            }
            Some("error") => {
                self.finished = true;
                Some(Err(LlmError::provider(
                    Provider::Anthropic,
                    "anthropic stream error",
                    None,
                )))
            }
            _ => None,
        }
    }
}
