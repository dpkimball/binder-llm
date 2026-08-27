//! Provider implementations. Each provider is isolated behind a feature flag.

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "xai")]
pub mod xai;

#[cfg(feature = "gemini")]
pub mod gemini;
