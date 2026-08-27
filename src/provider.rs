use async_trait::async_trait;

use crate::error::Result;
use crate::model::ModelInfo;
use crate::request::{ChatRequest, EmbeddingRequest};
use crate::response::{ChatResponse, EmbeddingResponse};
use crate::stream::ChatStream;

/// Provider-neutral LLM transport.
///
/// Implementations live under `providers/` and must not leak provider JSON.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse>;

    async fn stream(&self, request: ChatRequest) -> Result<ChatStream>;

    async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse>;

    async fn list_models(&self) -> Result<Vec<ModelInfo>>;
}
