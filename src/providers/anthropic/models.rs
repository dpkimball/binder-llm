use crate::model::{ModelInfo, Provider};

pub fn model_info(id: impl Into<String>) -> ModelInfo {
    let id = id.into();
    ModelInfo {
        provider: Provider::Anthropic,
        display_name: id.clone(),
        context_window: None,
        supports_tools: true,
        supports_streaming: true,
        supports_json: true,
        supports_embeddings: false,
        local: false,
        id,
    }
}
