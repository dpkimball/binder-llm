use crate::model::{ModelInfo, Provider};

pub fn model_info(id: impl Into<String>) -> ModelInfo {
    let id = id.into();
    let embeddings = id.contains("embed") || id.contains("nomic") || id.contains("minilm");
    ModelInfo {
        provider: Provider::Ollama,
        display_name: id.clone(),
        context_window: None,
        supports_tools: !embeddings,
        supports_streaming: !embeddings,
        supports_json: !embeddings,
        supports_embeddings: embeddings,
        local: true,
        id,
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct TagsResponse {
    #[serde(default)]
    pub models: Vec<TagModel>,
}

#[derive(Debug, serde::Deserialize)]
pub struct TagModel {
    pub name: String,
    #[serde(default)]
    pub details: Option<TagDetails>,
}

#[derive(Debug, serde::Deserialize)]
pub struct TagDetails {
    #[serde(default)]
    pub family: Option<String>,
    #[serde(default)]
    pub parameter_size: Option<String>,
}

impl TagModel {
    pub fn into_info(self) -> ModelInfo {
        model_info(self.name)
    }
}
