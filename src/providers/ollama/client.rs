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
    chat_body, embed_body, generate_body, uses_generate, ChatApiResponse, EmbedApiResponse,
};
use super::models::TagsResponse;

pub struct OllamaClient {
    http: HttpTransport,
}

impl OllamaClient {
    pub fn new(config: LlmConfig) -> Result<Self> {
        Ok(Self {
            http: HttpTransport::new(config)?,
        })
    }

    fn default_model(&self) -> &str {
        self.http.config().model()
    }

    async fn chat_once(&self, request: ChatRequest) -> Result<ChatResponse> {
        let model = request.resolved_model(self.default_model()).to_string();
        if uses_generate(&request) {
            let body = generate_body(&request, self.default_model())?;
            let json = self
                .http
                .post_json::<_, ChatApiResponse>("api/generate", &body)
                .await?;
            json.body.into_chat_response(&model, json.retries)
        } else {
            let body = chat_body(&request, self.default_model())?;
            let json = self
                .http
                .post_json::<_, ChatApiResponse>("api/chat", &body)
                .await?;
            json.body.into_chat_response(&model, json.retries)
        }
    }

    async fn stream_once(&self, mut request: ChatRequest) -> Result<ChatStream> {
        request.stream = true;
        let model = request.resolved_model(self.default_model()).to_string();
        let (path, body) = if uses_generate(&request) {
            (
                "api/generate",
                generate_body(&request, self.default_model())?,
            )
        } else {
            ("api/chat", chat_body(&request, self.default_model())?)
        };
        let start = self.http.post_stream(path, &body).await?;
        let retries = start.retries;
        let lines = response_lines(start.response);
        let stream = async_stream(lines, model, retries);
        Ok(ChatStream::from_stream(stream))
    }
}

fn async_stream(
    lines: impl futures::Stream<Item = Result<String>> + Send + 'static,
    model: String,
    retries: u32,
) -> impl futures::Stream<Item = Result<StreamChunk>> + Send {
    async_gen_stream(lines, model, retries)
}

fn async_gen_stream(
    lines: impl futures::Stream<Item = Result<String>> + Send + 'static,
    model: String,
    retries: u32,
) -> impl futures::Stream<Item = Result<StreamChunk>> + Send {
    futures::stream::unfold(
        StreamState {
            lines: Box::pin(lines),
            model,
            retries,
            content: String::new(),
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
                        let parsed: ChatApiResponse = match serde_json::from_str(line) {
                            Ok(v) => v,
                            Err(e) => {
                                state.finished = true;
                                return Some((
                                    Err(LlmError::serialization(format!(
                                        "invalid ollama stream JSON: {e}"
                                    ))),
                                    state,
                                ));
                            }
                        };
                        let delta = parsed.content_delta();
                        if !delta.is_empty() {
                            state.content.push_str(&delta);
                            return Some((Ok(StreamChunk::ContentDelta { text: delta }), state));
                        }
                        if parsed.done {
                            let usage = Usage::new(
                                parsed.prompt_eval_count.unwrap_or(0),
                                parsed.eval_count.unwrap_or(0),
                            );
                            let reason = parsed
                                .done_reason
                                .as_deref()
                                .map(FinishReason::from_provider_str)
                                .unwrap_or(FinishReason::Stop);
                            let message =
                                match parsed.into_chat_response(&state.model, state.retries) {
                                    Ok(resp) => resp.message,
                                    Err(_) => crate::ChatMessage::assistant(state.content.clone()),
                                };
                            state.finished = true;
                            // Emit usage then finish via two-step: stash finish as content empty done
                            return Some((
                                Ok(StreamChunk::Finished {
                                    reason,
                                    message: Some({
                                        let _ = usage;
                                        message
                                    }),
                                }),
                                state,
                            ));
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

struct StreamState {
    lines: std::pin::Pin<Box<dyn futures::Stream<Item = Result<String>> + Send>>,
    model: String,
    retries: u32,
    content: String,
    finished: bool,
}

#[async_trait]
impl LlmProvider for OllamaClient {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        self.chat_once(request).await
    }

    async fn stream(&self, request: ChatRequest) -> Result<ChatStream> {
        self.stream_once(request).await
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let model = request.resolved_model(self.default_model()).to_string();
        let inputs: Vec<String> = request.input.into_texts();
        if inputs.is_empty() {
            return Err(LlmError::invalid("embedding request has no input"));
        }
        let body = embed_body(&model, &inputs);
        let json = self
            .http
            .post_json::<_, EmbedApiResponse>("api/embed", &body)
            .await?;
        let mut embeddings = json.body.embeddings;
        if embeddings.is_empty() {
            if let Some(single) = json.body.embedding {
                embeddings.push(single);
            }
        }
        Ok(EmbeddingResponse {
            embeddings,
            model: json.body.model.unwrap_or(model),
            provider: Provider::Ollama,
            usage: Usage::default(),
            latency: std::time::Duration::ZERO,
            retries: json.retries,
        })
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let json = self.http.get_json::<TagsResponse>("api/tags").await?;
        Ok(json
            .body
            .models
            .into_iter()
            .map(|m| m.into_info())
            .collect())
    }
}
