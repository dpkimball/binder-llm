use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::message::ChatMessage;
use crate::model::Provider;
use crate::usage::Usage;

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Error,
}

impl fmt::Display for FinishReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
            Self::ContentFilter => "content_filter",
            Self::Error => "error",
        })
    }
}

impl FinishReason {
    pub fn from_provider_str(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "stop" | "end_turn" | "stop_sequence" | "completed" | "finish" => Self::Stop,
            "length" | "max_tokens" | "max_output_tokens" => Self::Length,
            "tool_calls" | "tool_use" | "function_call" | "tool" => Self::ToolCalls,
            "content_filter" | "content_filtered" | "safety" => Self::ContentFilter,
            _ => Self::Stop,
        }
    }
}

/// Provider-neutral chat response.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatResponse {
    pub message: ChatMessage,
    pub usage: Usage,
    pub finish_reason: FinishReason,
    pub provider: Provider,
    pub model: String,
    pub latency: Duration,
    /// Infrastructure retries performed before success.
    pub retries: u32,
}

impl ChatResponse {
    pub fn new(provider: Provider, model: impl Into<String>, message: ChatMessage) -> Self {
        Self {
            message,
            usage: Usage::default(),
            finish_reason: FinishReason::Stop,
            provider,
            model: model.into(),
            latency: Duration::ZERO,
            retries: 0,
        }
    }
}

/// Provider-neutral embedding response. No vector-store metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub model: String,
    pub provider: Provider,
    pub usage: Usage,
    pub latency: Duration,
    pub retries: u32,
}
