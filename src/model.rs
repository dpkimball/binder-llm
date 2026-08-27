use serde::{Deserialize, Serialize};

/// LLM provider identifier.
///
/// Variants exist regardless of crate features so configuration can name a
/// provider that is not compiled in. Construction then returns
/// [`crate::LlmError::UnsupportedFeature`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    Ollama,
    Openai,
    Anthropic,
    Xai,
    Gemini,
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Provider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::Openai => "openai",
            Self::Anthropic => "anthropic",
            Self::Xai => "xai",
            Self::Gemini => "gemini",
        }
    }

    pub fn default_base_url(self) -> &'static str {
        match self {
            Self::Ollama => "http://localhost:11434",
            Self::Openai => "https://api.openai.com/v1",
            Self::Anthropic => "https://api.anthropic.com",
            Self::Xai => "https://api.x.ai/v1",
            Self::Gemini => "https://generativelanguage.googleapis.com/v1beta",
        }
    }

    pub fn requires_api_key(self) -> bool {
        !matches!(self, Self::Ollama)
    }

    pub fn is_local(self) -> bool {
        matches!(self, Self::Ollama)
    }
}

impl std::str::FromStr for Provider {
    type Err = crate::LlmError;

    fn from_str(s: &str) -> crate::Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ollama" => Ok(Self::Ollama),
            "openai" => Ok(Self::Openai),
            "anthropic" => Ok(Self::Anthropic),
            "xai" | "x-ai" | "grok" => Ok(Self::Xai),
            "gemini" | "google" => Ok(Self::Gemini),
            other => Err(crate::LlmError::invalid(format!(
                "unknown provider '{other}'"
            ))),
        }
    }
}

/// Strongly typed model descriptor returned by [`crate::LlmProvider::list_models`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub provider: Provider,
    pub id: String,
    pub display_name: String,
    pub context_window: Option<u32>,
    pub supports_tools: bool,
    pub supports_streaming: bool,
    pub supports_json: bool,
    pub supports_embeddings: bool,
    pub local: bool,
}

/// Alias matching the spec name [`ProviderModel`].
pub type ProviderModel = ModelInfo;

impl ModelInfo {
    pub fn new(provider: Provider, id: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            display_name: id.clone(),
            context_window: None,
            supports_tools: true,
            supports_streaming: true,
            supports_json: true,
            supports_embeddings: false,
            local: provider.is_local(),
            provider,
            id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_provider() {
        assert_eq!("openai".parse::<Provider>().unwrap(), Provider::Openai);
        assert_eq!("grok".parse::<Provider>().unwrap(), Provider::Xai);
        assert!("nope".parse::<Provider>().is_err());
    }
}
