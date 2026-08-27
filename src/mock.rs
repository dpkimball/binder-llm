use std::sync::Mutex;

use async_trait::async_trait;
use futures::stream;

use crate::error::{LlmError, Result};
use crate::model::ModelInfo;
use crate::provider::LlmProvider;
use crate::request::{ChatRequest, EmbeddingRequest};
use crate::response::{ChatResponse, EmbeddingResponse};
use crate::stream::{ChatStream, StreamChunk};

/// In-memory provider for tests. No HTTP, no prompts stored beyond the request
/// struct the consumer already holds.
#[derive(Debug, Default)]
pub struct MockProvider {
    inner: Mutex<MockInner>,
}

#[derive(Debug, Default)]
struct MockInner {
    chat: Option<ChatResponse>,
    stream: Vec<StreamChunk>,
    embeddings: Option<EmbeddingResponse>,
    models: Vec<ModelInfo>,
    chat_error: Option<LlmError>,
    stream_error: Option<LlmError>,
    embed_error: Option<LlmError>,
    chat_calls: u32,
}

impl MockProvider {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_chat(self, response: ChatResponse) -> Self {
        self.inner.lock().expect("mock mutex").chat = Some(response);
        self
    }

    pub fn with_stream(self, chunks: Vec<StreamChunk>) -> Self {
        self.inner.lock().expect("mock mutex").stream = chunks;
        self
    }

    pub fn with_embeddings(self, response: EmbeddingResponse) -> Self {
        self.inner.lock().expect("mock mutex").embeddings = Some(response);
        self
    }

    pub fn with_models(self, models: Vec<ModelInfo>) -> Self {
        self.inner.lock().expect("mock mutex").models = models;
        self
    }

    pub fn with_chat_error(self, error: LlmError) -> Self {
        self.inner.lock().expect("mock mutex").chat_error = Some(error);
        self
    }

    pub fn chat_calls(&self) -> u32 {
        self.inner.lock().expect("mock mutex").chat_calls
    }
}

#[async_trait]
impl LlmProvider for MockProvider {
    async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse> {
        let mut inner = self.inner.lock().expect("mock mutex");
        inner.chat_calls += 1;
        if let Some(err) = inner.chat_error.clone() {
            return Err(err);
        }
        inner
            .chat
            .clone()
            .ok_or_else(|| LlmError::invalid("MockProvider has no chat response configured"))
    }

    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream> {
        let inner = self.inner.lock().expect("mock mutex");
        if let Some(err) = inner.stream_error.clone() {
            return Err(err);
        }
        let chunks = inner.stream.clone();
        Ok(ChatStream::from_stream(stream::iter(
            chunks.into_iter().map(Ok),
        )))
    }

    async fn embeddings(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let inner = self.inner.lock().expect("mock mutex");
        if let Some(err) = inner.embed_error.clone() {
            return Err(err);
        }
        inner
            .embeddings
            .clone()
            .ok_or_else(|| LlmError::invalid("MockProvider has no embeddings configured"))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(self.inner.lock().expect("mock mutex").models.clone())
    }
}
