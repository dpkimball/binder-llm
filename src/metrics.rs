use std::time::Duration;

use crate::model::Provider;
use crate::usage::Usage;

/// Call-level metrics. No metrics backend is bundled; consumers record these.
#[derive(Debug, Clone, PartialEq)]
pub struct LlmCallMetrics {
    pub provider: Provider,
    pub model: String,
    pub latency: Duration,
    pub tokens: Usage,
    pub retries: u32,
    pub errors: u32,
    pub stream_duration: Option<Duration>,
}

impl LlmCallMetrics {
    pub fn new(provider: Provider, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            latency: Duration::ZERO,
            tokens: Usage::default(),
            retries: 0,
            errors: 0,
            stream_duration: None,
        }
    }
}
