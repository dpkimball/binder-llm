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

use super::mapper::{chat_body, embeddings_body, ChatApi, EmbeddingsApi, ModelsApi};
use super::models::model_info;

pub struct XaiClient {
    http: HttpTransport,
}

impl XaiClient {
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
impl LlmProvider for XaiClient {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let model = request.resolved_model(self.default_model()).to_string();
        let body = chat_body(&request, self.default_model())?;
        let json = self
            .http
            .post_json::<_, ChatApi>("chat/completions", &body)
            .await?;
        json.body.into_chat_response(&model, json.retries)
    }

    async fn stream(&self, mut request: ChatRequest) -> Result<ChatStream> {
        request.stream = true;
        let body = chat_body(&request, self.default_model())?;
        let start = self.http.post_stream("chat/completions", &body).await?;
        Ok(ChatStream::from_stream(compat_sse(response_lines(
            start.response,
        ))))
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
        Ok(EmbeddingResponse {
            embeddings: json.body.data.into_iter().map(|d| d.embedding).collect(),
            model: json.body.model.unwrap_or(model),
            provider: Provider::Xai,
            usage: json
                .body
                .usage
                .map(|u| Usage::new(u.prompt_tokens.unwrap_or(0), 0))
                .unwrap_or_default(),
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

fn compat_sse(
    lines: impl futures::Stream<Item = Result<String>> + Send + 'static,
) -> impl futures::Stream<Item = Result<StreamChunk>> + Send {
    futures::stream::unfold(
        CompatSse {
            lines: Box::pin(lines),
            finished: false,
        },
        |mut state| async move {
            if state.finished {
                return None;
            }
            loop {
                match state.lines.next().await {
                    Some(Ok(line)) => {
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }
                        let data = line.strip_prefix("data:").unwrap_or(line).trim();
                        if data == "[DONE]" {
                            state.finished = true;
                            return Some((
                                Ok(StreamChunk::Finished {
                                    reason: FinishReason::Stop,
                                    message: None,
                                }),
                                state,
                            ));
                        }
                        let parsed: ChatApi = match serde_json::from_str(data) {
                            Ok(v) => v,
                            Err(e) => {
                                state.finished = true;
                                return Some((
                                    Err(LlmError::serialization(format!(
                                        "invalid xai stream JSON: {e}"
                                    ))),
                                    state,
                                ));
                            }
                        };
                        if let Some(choice) = parsed.choices.first() {
                            if let Some(delta) = &choice.delta {
                                if let Some(text) = &delta.content {
                                    if !text.is_empty() {
                                        return Some((
                                            Ok(StreamChunk::ContentDelta { text: text.clone() }),
                                            state,
                                        ));
                                    }
                                }
                                if let Some(calls) = &delta.tool_calls {
                                    if let Some(call) = calls.first() {
                                        return Some((
                                            Ok(StreamChunk::ToolCallDelta {
                                                index: call.index.unwrap_or(0),
                                                id: call.id.clone(),
                                                name: call
                                                    .function
                                                    .as_ref()
                                                    .and_then(|f| f.name.clone()),
                                                arguments_delta: call
                                                    .function
                                                    .as_ref()
                                                    .and_then(|f| f.arguments.clone()),
                                            }),
                                            state,
                                        ));
                                    }
                                }
                            }
                            if let Some(reason) = &choice.finish_reason {
                                state.finished = true;
                                return Some((
                                    Ok(StreamChunk::Finished {
                                        reason: FinishReason::from_provider_str(reason),
                                        message: None,
                                    }),
                                    state,
                                ));
                            }
                        }
                    }
                    Some(Err(e)) => {
                        state.finished = true;
                        return Some((Err(e), state));
                    }
                    None => {
                        return None;
                    }
                }
            }
        },
    )
}

struct CompatSse {
    lines: std::pin::Pin<Box<dyn futures::Stream<Item = Result<String>> + Send>>,
    finished: bool,
}
