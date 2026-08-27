use crate::model::{ModelInfo, Provider};

pub fn model_info(id: impl Into<String>) -> ModelInfo {
    let id = id.into();
    let embeddings = id.contains("embedding");
    ModelInfo {
        provider: Provider::Xai,
        display_name: id.clone(),
        context_window: None,
        supports_tools: !embeddings,
        supports_streaming: !embeddings,
        supports_json: !embeddings,
        supports_embeddings: embeddings,
        local: false,
        id,
    }
}
