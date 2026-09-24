use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::tool::ToolCall;

/// Arbitrary provider-neutral metadata. Never used for secrets.
pub type Metadata = BTreeMap<String, serde_json::Value>;

/// Image bytes attached to a user message. Encoding for a provider stays in that provider.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatImage {
    pub media_type: String,
    pub bytes: Vec<u8>,
}

/// Chat turn role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Provider-neutral chat message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum ChatMessage {
    System {
        content: String,
        #[serde(default)]
        metadata: Metadata,
    },
    User {
        content: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        images: Vec<ChatImage>,
        #[serde(default)]
        metadata: Metadata,
    },
    Assistant {
        content: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCall>,
        #[serde(default)]
        metadata: Metadata,
    },
    Tool {
        tool_call_id: String,
        content: String,
        #[serde(default)]
        metadata: Metadata,
    },
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
            metadata: Metadata::new(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::user_with_images(content, Vec::new())
    }

    pub fn user_with_images(content: impl Into<String>, images: Vec<ChatImage>) -> Self {
        Self::User {
            content: content.into(),
            images,
            metadata: Metadata::new(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::Assistant {
            content: content.into(),
            tool_calls: Vec::new(),
            metadata: Metadata::new(),
        }
    }

    pub fn assistant_with_tools(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self::Assistant {
            content: content.into(),
            tool_calls,
            metadata: Metadata::new(),
        }
    }

    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::Tool {
            tool_call_id: tool_call_id.into(),
            content: content.into(),
            metadata: Metadata::new(),
        }
    }

    pub fn role(&self) -> Role {
        match self {
            Self::System { .. } => Role::System,
            Self::User { .. } => Role::User,
            Self::Assistant { .. } => Role::Assistant,
            Self::Tool { .. } => Role::Tool,
        }
    }

    pub fn reject_if_images(messages: &[ChatMessage]) -> crate::error::Result<()> {
        if messages.iter().any(|message| !message.images().is_empty()) {
            Err(crate::error::LlmError::unsupported(
                "this provider does not accept images",
            ))
        } else {
            Ok(())
        }
    }

    pub fn images(&self) -> &[ChatImage] {
        match self {
            Self::User { images, .. } => images,
            _ => &[],
        }
    }

    pub fn content(&self) -> &str {
        match self {
            Self::System { content, .. }
            | Self::User { content, .. }
            | Self::Assistant { content, .. }
            | Self::Tool { content, .. } => content,
        }
    }

    pub fn metadata(&self) -> &Metadata {
        match self {
            Self::System { metadata, .. }
            | Self::User { metadata, .. }
            | Self::Assistant { metadata, .. }
            | Self::Tool { metadata, .. } => metadata,
        }
    }

    pub fn tool_calls(&self) -> &[ToolCall] {
        match self {
            Self::Assistant { tool_calls, .. } => tool_calls,
            _ => &[],
        }
    }

    pub fn tool_call_id(&self) -> Option<&str> {
        match self {
            Self::Tool { tool_call_id, .. } => Some(tool_call_id),
            _ => None,
        }
    }

    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        match &mut self {
            Self::System { metadata: slot, .. }
            | Self::User { metadata: slot, .. }
            | Self::Assistant { metadata: slot, .. }
            | Self::Tool { metadata: slot, .. } => *slot = metadata,
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_roundtrip() {
        let msg = ChatMessage::user("hello");
        let json = serde_json::to_string(&msg).unwrap();
        let back: ChatMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(msg, back);
        assert!(json.contains("\"role\":\"user\""));
    }
}
