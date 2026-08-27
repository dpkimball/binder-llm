use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::{LlmError, Result};
use crate::message::ChatMessage;
use crate::model::Provider;
use crate::request::ChatRequest;
use crate::response::{ChatResponse, FinishReason};
use crate::tool::{ToolCall, ToolChoice, ToolDefinition};
use crate::usage::Usage;

pub fn chat_body(request: &ChatRequest, default_model: &str) -> Result<Value> {
    if request.messages.is_empty() {
        return Err(LlmError::invalid("chat request has no messages"));
    }
    let mut body = json!({
        "model": request.resolved_model(default_model),
        "messages": request.messages.iter().map(map_message).collect::<Vec<_>>(),
        "stream": request.stream,
    });
    if let Some(t) = request.temperature {
        body["temperature"] = json!(t);
    }
    if let Some(max) = request.max_tokens {
        body["max_tokens"] = json!(max);
    }
    if let Some(seed) = request.seed {
        body["seed"] = json!(seed);
    }
    if !request.stop.is_empty() {
        body["stop"] = json!(request.stop);
    }
    if !request.tools.is_empty() {
        body["tools"] = Value::Array(request.tools.iter().map(map_tool).collect());
        body["tool_choice"] = map_tool_choice(&request.tool_choice);
    }
    if let Some(schema) = &request.json_schema {
        body["response_format"] = json!({
            "type": "json_schema",
            "json_schema": { "name": "binder_output", "schema": schema, "strict": true }
        });
    } else if request.json_mode {
        body["response_format"] = json!({ "type": "json_object" });
    }
    Ok(body)
}

fn map_message(message: &ChatMessage) -> Value {
    match message {
        ChatMessage::System { content, .. } => json!({"role": "system", "content": content}),
        ChatMessage::User { content, .. } => json!({"role": "user", "content": content}),
        ChatMessage::Assistant {
            content,
            tool_calls,
            ..
        } => {
            let mut value = json!({"role": "assistant", "content": content});
            if !tool_calls.is_empty() {
                value["tool_calls"] = Value::Array(
                    tool_calls
                        .iter()
                        .map(|c| {
                            json!({
                                "id": c.id,
                                "type": "function",
                                "function": { "name": c.name, "arguments": c.arguments }
                            })
                        })
                        .collect(),
                );
            }
            value
        }
        ChatMessage::Tool {
            tool_call_id,
            content,
            ..
        } => json!({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content
        }),
    }
}

fn map_tool(tool: &ToolDefinition) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters
        }
    })
}

fn map_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Named(name) => json!({"type": "function", "function": {"name": name}}),
    }
}

pub fn embeddings_body(model: &str, inputs: &[String]) -> Value {
    json!({ "model": model, "input": inputs })
}

#[derive(Debug, Deserialize)]
pub struct ChatApi {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub choices: Vec<Choice>,
    #[serde(default)]
    pub usage: Option<CompatUsage>,
}

#[derive(Debug, Deserialize)]
pub struct Choice {
    #[serde(default)]
    pub message: Option<ChoiceMessage>,
    #[serde(default)]
    pub delta: Option<ChoiceMessage>,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChoiceMessage {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<CompatToolCall>>,
}

#[derive(Debug, Deserialize)]
pub struct CompatToolCall {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub index: Option<usize>,
    #[serde(default)]
    pub function: Option<CompatFunction>,
}

#[derive(Debug, Deserialize)]
pub struct CompatFunction {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CompatUsage {
    #[serde(default)]
    pub prompt_tokens: Option<u32>,
    #[serde(default)]
    pub completion_tokens: Option<u32>,
    #[serde(default)]
    pub total_tokens: Option<u32>,
}

impl ChatApi {
    pub fn into_chat_response(self, default_model: &str, retries: u32) -> Result<ChatResponse> {
        let model = self.model.unwrap_or_else(|| default_model.to_string());
        let choice = self
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::provider(Provider::Xai, "empty choices", None))?;
        let msg = choice.message.unwrap_or(ChoiceMessage {
            content: None,
            tool_calls: None,
        });
        let content = msg.content.unwrap_or_default();
        let tool_calls = msg
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|c| {
                ToolCall::new(
                    c.id.unwrap_or_default(),
                    c.function
                        .as_ref()
                        .and_then(|f| f.name.clone())
                        .unwrap_or_default(),
                    c.function.and_then(|f| f.arguments).unwrap_or_default(),
                )
            })
            .collect::<Vec<_>>();
        let finish_reason = if !tool_calls.is_empty() {
            FinishReason::ToolCalls
        } else {
            choice
                .finish_reason
                .as_deref()
                .map(FinishReason::from_provider_str)
                .unwrap_or(FinishReason::Stop)
        };
        let message = if tool_calls.is_empty() {
            ChatMessage::assistant(content)
        } else {
            ChatMessage::assistant_with_tools(content, tool_calls)
        };
        let mut response = ChatResponse::new(Provider::Xai, model, message);
        response.finish_reason = finish_reason;
        if let Some(usage) = self.usage {
            response.usage = Usage {
                prompt_tokens: usage.prompt_tokens.unwrap_or(0),
                completion_tokens: usage.completion_tokens.unwrap_or(0),
                total_tokens: usage.total_tokens.unwrap_or(0),
            };
            if response.usage.total_tokens == 0 {
                response.usage.total_tokens = response
                    .usage
                    .prompt_tokens
                    .saturating_add(response.usage.completion_tokens);
            }
        }
        response.retries = retries;
        Ok(response)
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsApi {
    #[serde(default)]
    pub data: Vec<EmbeddingData>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub usage: Option<CompatUsage>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct ModelsApi {
    #[serde(default)]
    pub data: Vec<ModelRow>,
}

#[derive(Debug, Deserialize)]
pub struct ModelRow {
    pub id: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_mode() {
        let req = ChatRequest::new(vec![ChatMessage::user("hi")]).with_json_mode(true);
        let body = chat_body(&req, "grok-3").unwrap();
        assert_eq!(body["response_format"]["type"], "json_object");
    }
}
