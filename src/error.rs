use std::time::Duration;

use crate::model::Provider;

/// Unified error type for all LLM operations.
///
/// Provider-specific errors are mapped here. Raw provider payloads never
/// escape this crate.
#[derive(Debug, Clone, thiserror::Error)]
pub enum LlmError {
    #[error("authentication failed: {message}")]
    Authentication { message: String },

    #[error("connection failed: {message}")]
    Connection { message: String },

    #[error("request timed out")]
    Timeout,

    #[error("rate limited")]
    RateLimited { retry_after: Option<Duration> },

    #[error("invalid request: {message}")]
    InvalidRequest { message: String },

    #[error("provider error ({provider}): {message}")]
    ProviderError {
        provider: Provider,
        message: String,
        status: Option<u16>,
    },

    #[error("serialization error: {message}")]
    Serialization { message: String },

    #[error("unsupported feature: {feature}")]
    UnsupportedFeature { feature: String },

    #[error("request cancelled")]
    Cancelled,
}

/// Result alias for binder-llm operations.
pub type Result<T> = std::result::Result<T, LlmError>;

impl LlmError {
    /// Stable error kind for tracing metadata (never includes prompt text).
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Authentication { .. } => "authentication",
            Self::Connection { .. } => "connection",
            Self::Timeout => "timeout",
            Self::RateLimited { .. } => "rate_limited",
            Self::InvalidRequest { .. } => "invalid_request",
            Self::ProviderError { .. } => "provider_error",
            Self::Serialization { .. } => "serialization",
            Self::UnsupportedFeature { .. } => "unsupported_feature",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn unsupported(feature: impl Into<String>) -> Self {
        Self::UnsupportedFeature {
            feature: feature.into(),
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::InvalidRequest {
            message: message.into(),
        }
    }

    pub fn serialization(message: impl Into<String>) -> Self {
        Self::Serialization {
            message: message.into(),
        }
    }

    pub fn connection(message: impl Into<String>) -> Self {
        Self::Connection {
            message: message.into(),
        }
    }

    pub fn provider(provider: Provider, message: impl Into<String>, status: Option<u16>) -> Self {
        Self::ProviderError {
            provider,
            message: message.into(),
            status,
        }
    }

    /// HTTP statuses retried by infrastructure (429, 500, 502, 503, 504).
    pub fn is_retryable_status(status: u16) -> bool {
        matches!(status, 429 | 500 | 502 | 503 | 504)
    }

    pub fn from_status(provider: Provider, status: u16, message: impl Into<String>) -> Self {
        let message = sanitize_error_message(message.into());
        match status {
            401 | 403 => Self::Authentication { message },
            400 | 404 | 409 | 413 | 415 | 422 => Self::InvalidRequest { message },
            429 => Self::RateLimited { retry_after: None },
            _ if (400..500).contains(&status) => Self::InvalidRequest { message },
            _ => Self::provider(provider, message, Some(status)),
        }
    }
}

/// Drop anything that looks like a bearer token or overly long body echo.
pub(crate) fn sanitize_error_message(raw: String) -> String {
    let trimmed = raw.trim();
    let truncated = if trimmed.len() > 500 {
        format!("{}…", &trimmed[..500])
    } else {
        trimmed.to_string()
    };
    truncated
        .replace("Bearer ", "Bearer [redacted] ")
        .replace("sk-", "sk-[redacted]")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_mapping() {
        assert!(matches!(
            LlmError::from_status(Provider::Ollama, 401, "nope"),
            LlmError::Authentication { .. }
        ));
        assert!(matches!(
            LlmError::from_status(Provider::Ollama, 429, "slow"),
            LlmError::RateLimited { .. }
        ));
        assert!(LlmError::is_retryable_status(502));
        assert!(!LlmError::is_retryable_status(400));
    }

    #[test]
    fn sanitizes_keys() {
        let msg = sanitize_error_message("Bearer supersecret and sk-abc".into());
        assert!(!msg.contains("supersecret") || msg.contains("[redacted]"));
        assert!(msg.contains("[redacted]"));
    }
}
