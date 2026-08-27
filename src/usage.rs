use serde::{Deserialize, Serialize};

/// Token accounting for a single LLM call.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl Usage {
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens.saturating_add(completion_tokens),
        }
    }

    pub fn is_empty(self) -> bool {
        self.total_tokens == 0 && self.prompt_tokens == 0 && self.completion_tokens == 0
    }
}
