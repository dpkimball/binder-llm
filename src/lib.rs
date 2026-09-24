//! Canonical LLM abstraction for Binder.
//!
//! Applications, crates, plugins, and services must talk to models through
//! [`LlmClient`]. Provider JSON, SDKs, and prompts do not belong here.

mod client;
mod config;
mod error;
mod http;
mod message;
mod metrics;
mod mock;
mod model;
mod provider;
mod request;
mod response;
mod stream;
mod tool;
mod usage;

pub mod providers;

pub use client::LlmClient;
pub use config::{LlmClientBuilder, LlmConfig, RetryPolicy};
pub use error::{LlmError, Result};
pub use message::{ChatImage, ChatMessage, Metadata, Role};
pub use metrics::LlmCallMetrics;
pub use mock::MockProvider;
pub use model::{ModelInfo, Provider, ProviderModel};
pub use provider::LlmProvider;
pub use request::{ChatRequest, EmbeddingInput, EmbeddingRequest};
pub use response::{ChatResponse, EmbeddingResponse, FinishReason};
pub use stream::{ChatStream, StreamChunk};
pub use tool::{ToolCall, ToolChoice, ToolDefinition, ToolResult};
pub use usage::Usage;
