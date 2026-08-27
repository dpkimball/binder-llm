use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::{LlmError, Result};
use crate::message::ChatMessage;
use crate::model::Provider;
use crate::request::ChatRequest;
use crate::response::{ChatResponse, FinishReason};
use crate::tool::{ToolCall, ToolChoice, ToolDefinition};
use crate::usage::Usage;

pub fn messages_body(request: &ChatRequest, default_model: &str) -> Result<Value> {
    if request.messages.is_empty() {
        return Err(LlmError::invalid("chat request has no messages"));
    }
    let mut system = Vec::new();
    let mut messages = Vec::new();
    for message in &request.messages {
        match message {
            ChatMessage::System { content, .. } => system.push(content.clone()),
            ChatMessage::User { content, .. } => {
                messages.push(json!({"role": "user", "content": content}));
            }
            ChatMessage::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut parts = Vec::new();
                if !content.is_empty() {
                    parts.push(json!({"type": "text", "text": content}));
                }
                for call in tool_calls {
                    let input = serde_json::from_str(&call.arguments).unwrap_or(json!({}));
                    parts.push(json!({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": input
                    }));
                }
                messages.push(json!({"role": "assistant", "content": parts}));
            }
            ChatMessage::Tool {
                tool_call_id,
                content,
                ..
            } => {
                messages.push(json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": content
                    }]
                }));
            }
        }
    }
    let mut body = json!({
        "model": request.resolved_model(default_model),
        "messages": messages,
        "max_tokens": request.max_tokens.unwrap_or(4096),
        "stream": request.stream,
    });
    if !system.is_empty() {
        body["system"] = Value::String(system.join("\n"));
    }
    if let Some(t) = request.temperature {
        body["temperature"] = json!(t);
    }
    if !request.stop.is_empty() {
        body["stop_sequences"] = json!(request.stop);
    }
    if !request.tools.is_empty() {
        body["tools"] = Value::Array(request.tools.iter().map(map_tool).collect());
        body["tool_choice"] = map_tool_choice(&request.tool_choice);
    }
    if let Some(schema) = &request.json_schema {
        body["output_format"] = json!({
            "type": "json_schema",
            "schema": schema
        });
    } else if request.json_mode {
        return Err(LlmError::unsupported(
            "anthropic json_mode requires json_schema on ChatRequest",
        ));
    }
    Ok(body)
}

fn map_tool(tool: &ToolDefinition) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters
    })
}

fn map_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!({"type": "auto"}),
        ToolChoice::None => json!({"type": "none"}),
        ToolChoice::Required => json!({"type": "any"}),
        ToolChoice::Named(name) => json!({"type": "tool", "name": name}),
    }
}

#[derive(Debug, Deserialize)]
pub struct MessagesApi {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub content: Vec<ContentBlock>,
    #[serde(default)]
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub usage: Option<AnthropicUsage>,
    #[serde(default)]
    pub error: Option<ApiError>,
}

#[derive(Debug, Deserialize)]
pub struct ApiError {
    #[serde(default)]
    pub message: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub input: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicUsage {
    #[serde(default)]
    pub input_tokens: Option<u32>,
    #[serde(default)]
    pub output_tokens: Option<u32>,
}

impl MessagesApi {
    pub fn into_chat_response(self, default_model: &str, retries: u32) -> Result<ChatResponse> {
        if let Some(err) = self.error.and_then(|e| e.message) {
            return Err(LlmError::provider(Provider::Anthropic, err, None));
        }
        let model = self.model.unwrap_or_else(|| default_model.to_string());
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        for block in self.content {
            match block.kind.as_deref() {
                Some("tool_use") => {
                    let args = block.input.unwrap_or(json!({}));
                    tool_calls.push(ToolCall::new(
                        block.id.unwrap_or_default(),
                        block.name.unwrap_or_default(),
                        args.to_string(),
                    ));
                }
                _ => {
                    if let Some(t) = block.text {
                        text.push_str(&t);
                    }
                }
            }
        }
        let finish_reason = if !tool_calls.is_empty() {
            FinishReason::ToolCalls
        } else {
            self.stop_reason
                .as_deref()
                .map(FinishReason::from_provider_str)
                .unwrap_or(FinishReason::Stop)
        };
        let message = if tool_calls.is_empty() {
            ChatMessage::assistant(text)
        } else {
            ChatMessage::assistant_with_tools(text, tool_calls)
        };
        let mut response = ChatResponse::new(Provider::Anthropic, model, message);
        response.finish_reason = finish_reason;
        if let Some(usage) = self.usage {
            response.usage = Usage::new(
                usage.input_tokens.unwrap_or(0),
                usage.output_tokens.unwrap_or(0),
            );
        }
        response.retries = retries;
        Ok(response)
    }
}

#[derive(Debug, Deserialize)]
pub struct ModelsApi {
    #[serde(default)]
    pub data: Vec<ModelRow>,
}

#[derive(Debug, Deserialize)]
pub struct ModelRow {
    pub id: String,
    #[serde(default)]
    pub display_name: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct StreamEvent {
    #[serde(rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub delta: Option<StreamDelta>,
    #[serde(default)]
    pub content_block: Option<ContentBlock>,
    #[serde(default)]
    pub message: Option<MessagesApi>,
    #[serde(default)]
    pub usage: Option<AnthropicUsage>,
    #[serde(default)]
    pub index: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct StreamDelta {
    #[serde(rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub partial_json: Option<String>,
    #[serde(default)]
    pub stop_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_is_top_level() {
        let req = ChatRequest::new(vec![ChatMessage::system("sys"), ChatMessage::user("hi")]);
        let body = messages_body(&req, "claude").unwrap();
        assert_eq!(body["system"], "sys");
        assert_eq!(body["messages"][0]["role"], "user");
    }

    #[test]
    fn json_mode_without_schema_is_unsupported() {
        let req = ChatRequest::new(vec![ChatMessage::user("hi")]).with_json_mode(true);
        let err = messages_body(&req, "claude").unwrap_err();
        assert!(matches!(err, LlmError::UnsupportedFeature { .. }));
    }
}
