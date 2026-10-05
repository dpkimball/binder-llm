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

const MAX_ERROR_CHARS: usize = 500;

/// Remove credential material, then truncate on a character boundary.
///
/// Redaction runs before truncation so a token is never sliced in half and
/// left in the returned text.
pub(crate) fn sanitize_error_message(raw: String) -> String {
    let redacted = redact_secrets(raw.trim());
    truncate_chars(&redacted, MAX_ERROR_CHARS)
}

fn truncate_chars(input: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for (i, ch) in input.chars().enumerate() {
        if i == max_chars {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

fn redact_secrets(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while !rest.is_empty() {
        if let Some(consumed) = consume_secret(rest) {
            out.push_str("[redacted]");
            rest = &rest[consumed..];
            continue;
        }
        let ch = rest.chars().next().expect("rest is non-empty");
        out.push(ch);
        rest = &rest[ch.len_utf8()..];
    }
    out
}

/// If `input` starts with a credential pattern, return how many bytes to drop.
fn consume_secret(input: &str) -> Option<usize> {
    consume_bearer(input)
        .or_else(|| consume_sk_key(input))
        .or_else(|| consume_assignment(input))
}

fn consume_bearer(input: &str) -> Option<usize> {
    let rest = strip_ascii_prefix(input, "bearer")?;
    let rest = strip_ascii_whitespace(rest)?;
    let token_len = leading_token_len(rest);
    if token_len == 0 {
        return None;
    }
    Some(input.len() - rest.len() + token_len)
}

fn consume_sk_key(input: &str) -> Option<usize> {
    let rest = strip_ascii_prefix(input, "sk-")?;
    let token_len = leading_key_len(rest);
    if token_len == 0 {
        return None;
    }
    Some(input.len() - rest.len() + token_len)
}

fn consume_assignment(input: &str) -> Option<usize> {
    let rest = [
        "api_key",
        "api-key",
        "apikey",
        "access_token",
        "access-token",
    ]
    .iter()
    .find_map(|name| strip_ascii_prefix(input, name))?;
    let rest = rest.trim_start_matches([' ', '\t']);
    let rest = rest.strip_prefix('=').or_else(|| rest.strip_prefix(':'))?;
    let rest = rest.trim_start_matches([' ', '\t']);
    let token_len = leading_token_len(rest);
    if token_len == 0 {
        return None;
    }
    Some(input.len() - rest.len() + token_len)
}

fn strip_ascii_prefix<'a>(input: &'a str, prefix: &str) -> Option<&'a str> {
    let mut input_chars = input.chars();
    for expected in prefix.chars() {
        let got = input_chars.next()?.to_ascii_lowercase();
        if got != expected {
            return None;
        }
    }
    Some(&input[prefix.len()..])
}

fn strip_ascii_whitespace(input: &str) -> Option<&str> {
    let rest = input.trim_start_matches([' ', '\t']);
    if rest.len() == input.len() {
        None
    } else {
        Some(rest)
    }
}

fn leading_token_len(input: &str) -> usize {
    input
        .chars()
        .take_while(|c| !c.is_whitespace())
        .map(char::len_utf8)
        .sum()
}

fn leading_key_len(input: &str) -> usize {
    input
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .map(char::len_utf8)
        .sum()
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
    fn sanitizes_bearer_token() {
        let msg =
            sanitize_error_message("authentication failed: Bearer supersecret-token-value".into());
        assert!(!msg.contains("supersecret"));
        assert!(msg.contains("[redacted]"));
    }

    #[test]
    fn sanitizes_api_key_shapes() {
        let msg = sanitize_error_message(
            "rejected sk-abcDEF123456 and api_key=anothersecretvalue".into(),
        );
        assert!(!msg.contains("abcDEF123456"));
        assert!(!msg.contains("anothersecretvalue"));
        assert!(!msg.contains("sk-"));
    }

    #[test]
    fn truncates_on_char_boundary() {
        // 499 ASCII bytes, then a 3-byte character, so a byte slice at 500
        // would land inside that character.
        let mut raw = "x".repeat(499);
        raw.push('你');
        raw.push_str(" Bearer supersecret-token-value");
        let msg = sanitize_error_message(raw);
        assert!(!msg.contains("supersecret"));
        assert!(msg.ends_with('…'));
        assert!(msg.is_char_boundary(msg.len()));
    }

    #[test]
    fn provider_status_drops_credential() {
        let err = LlmError::from_status(
            Provider::Openai,
            401,
            r#"{"error":{"message":"bad Bearer sk-liveSECRETvalue123"}}"#,
        );
        let text = err.to_string();
        assert!(!text.contains("sk-liveSECRETvalue123"));
        assert!(!text.contains("liveSECRET"));
    }
}
