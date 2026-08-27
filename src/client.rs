use std::sync::Arc;
use std::time::Instant;

use serde::de::DeserializeOwned;

use crate::config::{LlmClientBuilder, LlmConfig};
use crate::error::{LlmError, Result};
use crate::model::ModelInfo;
use crate::provider::LlmProvider;
use crate::request::{ChatRequest, EmbeddingRequest};
use crate::response::{ChatResponse, EmbeddingResponse};
use crate::stream::ChatStream;

/// High-level façade. Applications depend on this type, not on providers.
#[derive(Clone)]
pub struct LlmClient {
    provider: Arc<dyn LlmProvider>,
    config: LlmConfig,
}

impl LlmClient {
    pub fn builder() -> LlmClientBuilder {
        LlmClientBuilder::new()
    }

    /// Wrap an existing provider (including [`crate::MockProvider`]).
    pub fn from_provider(provider: Arc<dyn LlmProvider>, config: LlmConfig) -> Self {
        Self { provider, config }
    }

    pub fn config(&self) -> &LlmConfig {
        &self.config
    }

    pub fn provider_kind(&self) -> crate::Provider {
        self.config.provider()
    }

    pub async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let request = self.prepare_chat(request);
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.config.model().to_string());
        tracing::info!(
            provider = %self.config.provider(),
            model = %model,
            message_count = request.messages.len(),
            has_tools = !request.tools.is_empty(),
            json_mode = request.json_mode,
            "llm.chat.start"
        );
        let started = Instant::now();
        let result = self.provider.chat(request).await;
        log_chat_result(&result, self.config.provider(), &model, started);
        result.map(|mut response| {
            if response.latency.is_zero() {
                response.latency = started.elapsed();
            }
            response
        })
    }

    pub async fn stream(&self, request: ChatRequest) -> Result<ChatStream> {
        let mut request = self.prepare_chat(request);
        request.stream = true;
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.config.model().to_string());
        tracing::info!(
            provider = %self.config.provider(),
            model = %model,
            message_count = request.messages.len(),
            "llm.stream.start"
        );
        self.provider.stream(request).await
    }

    pub async fn embeddings(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let request = self.prepare_embed(request);
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| self.config.model().to_string());
        tracing::info!(
            provider = %self.config.provider(),
            model = %model,
            input_count = request.input.as_texts().len(),
            "llm.embeddings.start"
        );
        let started = Instant::now();
        let result = self.provider.embeddings(request).await;
        match &result {
            Ok(resp) => tracing::info!(
                provider = %resp.provider,
                model = %resp.model,
                latency_ms = started.elapsed().as_millis() as u64,
                retries = resp.retries,
                vectors = resp.embeddings.len(),
                "llm.embeddings.ok"
            ),
            Err(err) => tracing::warn!(
                provider = %self.config.provider(),
                model = %model,
                error_kind = err.kind(),
                "llm.embeddings.err"
            ),
        }
        result
    }

    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        tracing::info!(provider = %self.config.provider(), "llm.models.start");
        self.provider.list_models().await
    }

    /// Chat and deserialize the assistant content as JSON.
    ///
    /// Sets `json_mode` on the request. Does not inject prompts.
    pub async fn chat_json<T: DeserializeOwned>(&self, mut request: ChatRequest) -> Result<T> {
        request.json_mode = true;
        request.stream = false;
        let response = self.chat(request).await?;
        let content = response.message.content().trim();
        serde_json::from_str(content).map_err(|e| {
            LlmError::serialization(format!("assistant content is not valid JSON: {e}"))
        })
    }

    fn prepare_chat(&self, mut request: ChatRequest) -> ChatRequest {
        if request.model.as_ref().map(|m| m.is_empty()).unwrap_or(true) {
            request.model = Some(self.config.model().to_string());
        }
        request
    }

    fn prepare_embed(&self, mut request: EmbeddingRequest) -> EmbeddingRequest {
        if request.model.as_ref().map(|m| m.is_empty()).unwrap_or(true) {
            request.model = Some(self.config.model().to_string());
        }
        request
    }
}

impl LlmClientBuilder {
    pub fn build(self) -> Result<LlmClient> {
        let config = self.build_config()?;
        let provider = build_provider(&config)?;
        Ok(LlmClient { provider, config })
    }
}

fn build_provider(config: &LlmConfig) -> Result<Arc<dyn LlmProvider>> {
    match config.provider() {
        crate::Provider::Ollama => construct_ollama(config),
        crate::Provider::Openai => construct_openai(config),
        crate::Provider::Anthropic => construct_anthropic(config),
        crate::Provider::Xai => construct_xai(config),
        crate::Provider::Gemini => construct_gemini(config),
    }
}

#[cfg(feature = "ollama")]
fn construct_ollama(config: &LlmConfig) -> Result<Arc<dyn LlmProvider>> {
    Ok(Arc::new(crate::providers::ollama::OllamaClient::new(
        config.clone(),
    )?))
}

#[cfg(not(feature = "ollama"))]
fn construct_ollama(_config: &LlmConfig) -> Result<Arc<dyn LlmProvider>> {
    Err(LlmError::unsupported(
        "ollama provider (enable feature `ollama`)",
    ))
}

#[cfg(feature = "openai")]
fn construct_openai(config: &LlmConfig) -> Result<Arc<dyn LlmProvider>> {
    Ok(Arc::new(crate::providers::openai::OpenAiClient::new(
        config.clone(),
    )?))
}

#[cfg(not(feature = "openai"))]
fn construct_openai(_config: &LlmConfig) -> Result<Arc<dyn LlmProvider>> {
    Err(LlmError::unsupported(
        "openai provider (enable feature `openai`)",
    ))
}

#[cfg(feature = "anthropic")]
fn construct_anthropic(config: &LlmConfig) -> Result<Arc<dyn LlmProvider>> {
    Ok(Arc::new(crate::providers::anthropic::AnthropicClient::new(
        config.clone(),
    )?))
}

#[cfg(not(feature = "anthropic"))]
fn construct_anthropic(_config: &LlmConfig) -> Result<Arc<dyn LlmProvider>> {
    Err(LlmError::unsupported(
        "anthropic provider (enable feature `anthropic`)",
    ))
}

#[cfg(feature = "xai")]
fn construct_xai(config: &LlmConfig) -> Result<Arc<dyn LlmProvider>> {
    Ok(Arc::new(crate::providers::xai::XaiClient::new(
        config.clone(),
    )?))
}

#[cfg(not(feature = "xai"))]
fn construct_xai(_config: &LlmConfig) -> Result<Arc<dyn LlmProvider>> {
    Err(LlmError::unsupported("xai provider (enable feature `xai`)"))
}

#[cfg(feature = "gemini")]
fn construct_gemini(config: &LlmConfig) -> Result<Arc<dyn LlmProvider>> {
    Ok(Arc::new(crate::providers::gemini::GeminiClient::new(
        config.clone(),
    )?))
}

#[cfg(not(feature = "gemini"))]
fn construct_gemini(_config: &LlmConfig) -> Result<Arc<dyn LlmProvider>> {
    Err(LlmError::unsupported(
        "gemini provider (enable feature `gemini`)",
    ))
}

fn log_chat_result(
    result: &Result<ChatResponse>,
    provider: crate::Provider,
    model: &str,
    started: Instant,
) {
    match result {
        Ok(resp) => tracing::info!(
            provider = %resp.provider,
            model = %resp.model,
            latency_ms = started.elapsed().as_millis() as u64,
            prompt_tokens = resp.usage.prompt_tokens,
            completion_tokens = resp.usage.completion_tokens,
            finish_reason = %resp.finish_reason,
            retries = resp.retries,
            "llm.chat.ok"
        ),
        Err(err) => tracing::warn!(
            provider = %provider,
            model = %model,
            latency_ms = started.elapsed().as_millis() as u64,
            error_kind = err.kind(),
            "llm.chat.err"
        ),
    }
}
