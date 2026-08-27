use async_trait::async_trait;
use futures::StreamExt;

use crate::config::LlmConfig;
use crate::error::{LlmError, Result};
use crate::http::{join_url, response_lines, HttpTransport};
use crate::model::{ModelInfo, Provider};
use crate::provider::LlmProvider;
use crate::request::{ChatRequest, EmbeddingRequest};
use crate::response::{ChatResponse, EmbeddingResponse};
use crate::stream::{ChatStream, StreamChunk};

use super::mapper::{embed_body, generate_body, EmbedApi, GenerateApi, ModelsApi};
use super::models::{model_info, strip_models_prefix};

pub struct GeminiClient {
    http: HttpTransport,
}

impl GeminiClient {
    pub fn new(config: LlmConfig) -> Result<Self> {
        Ok(Self {
            http: HttpTransport::new(config)?,
        })
    }

    fn default_model(&self) -> &str {
        self.http.config().model()
    }

    fn model_path(&self, model: &str, action: &str) -> String {
        let id = strip_models_prefix(model);
        format!("models/{id}:{action}")
    }
}

#[async_trait]
impl LlmProvider for GeminiClient {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let model = request.resolved_model(self.default_model()).to_string();
        let body = generate_body(&request, self.default_model())?;
        let path = self.model_path(&model, "generateContent");
        let json = self.http.post_json::<_, GenerateApi>(&path, &body).await?;
        json.body.into_chat_response(&model, json.retries)
    }

    async fn stream(&self, mut request: ChatRequest) -> Result<ChatStream> {
        request.stream = true;
        let model = request.resolved_model(self.default_model()).to_string();
        let body = generate_body(&request, self.default_model())?;
        let path = self.model_path(&model, "streamGenerateContent");
        let url = format!("{}?alt=sse", join_url(self.http.config().base_url(), &path));
        let start = self.http.post_stream_url(&url, &body).await?;
        let model_for_stream = model.clone();
        Ok(ChatStream::from_stream(gemini_sse(
            response_lines(start.response),
            model_for_stream,
        )))
    }

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let model = request.resolved_model(self.default_model()).to_string();
        let texts = request.input.into_texts();
        if texts.is_empty() {
            return Err(LlmError::invalid("embedding request has no input"));
        }
        let mut embeddings = Vec::new();
        let mut retries = 0;
        for text in texts {
            let body = embed_body(&text);
            let path = self.model_path(&model, "embedContent");
            let json = self.http.post_json::<_, EmbedApi>(&path, &body).await?;
            retries += json.retries;
            embeddings.push(json.body.embedding.map(|e| e.values).unwrap_or_default());
        }
        Ok(EmbeddingResponse {
            embeddings,
            model,
            provider: Provider::Gemini,
            usage: crate::Usage::default(),
            latency: std::time::Duration::ZERO,
            retries,
        })
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let json = self.http.get_json::<ModelsApi>("models").await?;
        Ok(json
            .body
            .models
            .into_iter()
            .map(|m| {
                let mut info = model_info(strip_models_prefix(&m.name));
                if let Some(name) = m.display_name {
                    info.display_name = name;
                }
                info
            })
            .collect())
    }
}

fn gemini_sse(
    lines: impl futures::Stream<Item = Result<String>> + Send + 'static,
    model: String,
) -> impl futures::Stream<Item = Result<StreamChunk>> + Send {
    futures::stream::unfold(
        GemSse {
            lines: Box::pin(lines),
            data: String::new(),
            finished: false,
            model,
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

struct GemSse {
    lines: std::pin::Pin<Box<dyn futures::Stream<Item = Result<String>> + Send>>,
    data: String,
    finished: bool,
    model: String,
}

impl GemSse {
    fn flush(&mut self) -> Option<Result<StreamChunk>> {
        if self.data.is_empty() {
            return None;
        }
        let data = std::mem::take(&mut self.data);
        let parsed: GenerateApi = match serde_json::from_str(&data) {
            Ok(v) => v,
            Err(e) => {
                return Some(Err(LlmError::serialization(format!(
                    "invalid gemini sse JSON: {e}"
                ))))
            }
        };
        match parsed.into_chat_response(&self.model, 0) {
            Ok(resp) => {
                let text = resp.message.content().to_string();
                if text.is_empty() {
                    Some(Ok(StreamChunk::Finished {
                        reason: resp.finish_reason,
                        message: Some(resp.message),
                    }))
                } else {
                    Some(Ok(StreamChunk::ContentDelta { text }))
                }
            }
            Err(e) => Some(Err(e)),
        }
    }
}
