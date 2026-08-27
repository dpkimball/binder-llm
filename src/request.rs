use serde::{Deserialize, Serialize};

use crate::message::{ChatMessage, Metadata};
use crate::tool::{ToolChoice, ToolDefinition};

/// Provider-neutral chat request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(default)]
    pub tool_choice: ToolChoice,
    #[serde(default)]
    pub json_mode: bool,
    /// Optional JSON Schema for structured outputs. Used with `json_mode`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub metadata: Metadata,
}

impl Default for ChatRequest {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            temperature: None,
            max_tokens: None,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            json_mode: false,
            json_schema: None,
            model: None,
            seed: None,
            stop: Vec::new(),
            stream: false,
            metadata: Metadata::new(),
        }
    }
}

impl ChatRequest {
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            ..Self::default()
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_json_mode(mut self, json_mode: bool) -> Self {
        self.json_mode = json_mode;
        self
    }

    pub fn resolved_model<'a>(&'a self, default_model: &'a str) -> &'a str {
        self.model
            .as_deref()
            .filter(|m| !m.is_empty())
            .unwrap_or(default_model)
    }
}

/// Embedding input: one string or a batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Text(String),
    Texts(Vec<String>),
}

impl EmbeddingInput {
    pub fn as_texts(&self) -> Vec<&str> {
        match self {
            Self::Text(s) => vec![s.as_str()],
            Self::Texts(v) => v.iter().map(String::as_str).collect(),
        }
    }

    pub fn into_texts(self) -> Vec<String> {
        match self {
            Self::Text(s) => vec![s],
            Self::Texts(v) => v,
        }
    }
}

impl From<String> for EmbeddingInput {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for EmbeddingInput {
    fn from(value: &str) -> Self {
        Self::Text(value.to_string())
    }
}

impl From<Vec<String>> for EmbeddingInput {
    fn from(value: Vec<String>) -> Self {
        Self::Texts(value)
    }
}

/// Provider-neutral embedding request. No vector-store fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub input: EmbeddingInput,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default)]
    pub metadata: Metadata,
}

impl EmbeddingRequest {
    pub fn new(input: impl Into<EmbeddingInput>) -> Self {
        Self {
            input: input.into(),
            model: None,
            metadata: Metadata::new(),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    pub fn resolved_model<'a>(&'a self, default_model: &'a str) -> &'a str {
        self.model
            .as_deref()
            .filter(|m| !m.is_empty())
            .unwrap_or(default_model)
    }
}
