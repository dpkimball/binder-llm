use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::{LlmError, Result};
use crate::message::ChatMessage;
use crate::model::Provider;
use crate::request::ChatRequest;
use crate::response::{ChatResponse, FinishReason};
use crate::tool::{ToolCall, ToolChoice, ToolDefinition};
use crate::usage::Usage;

pub fn responses_body(request: &ChatRequest, default_model: &str) -> Result<Value> {
    if request.messages.is_empty() {
        return Err(LlmError::invalid("chat request has no messages"));
    }
    let mut instructions = Vec::new();
    let mut input = Vec::new();
    for message in &request.messages {
        match message {
            ChatMessage::System { content, .. } => instructions.push(content.clone()),
            ChatMessage::User { content, .. } => {
                input.push(json!({"role": "user", "content": content}));
            }
            ChatMessage::Assistant {
                content,
                tool_calls,
                ..
            } => {
                if !content.is_empty() {
                    input.push(json!({"role": "assistant", "content": content}));
                }
                for call in tool_calls {
                    input.push(json!({
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": call.arguments
                    }));
                }
            }
            ChatMessage::Tool {
                tool_call_id,
                content,
                ..
            } => {
                input.push(json!({
                    "type": "function_call_output",
                    "call_id": tool_call_id,
                    "output": content
                }));
            }
        }
    }
    let mut body = json!({
        "model": request.resolved_model(default_model),
        "input": input,
        "stream": request.stream,
    });
    if !instructions.is_empty() {
        body["instructions"] = Value::String(instructions.join("\n"));
    }
    if let Some(t) = request.temperature {
        body["temperature"] = json!(t);
    }
    if let Some(max) = request.max_tokens {
        body["max_output_tokens"] = json!(max);
    }
    if !request.stop.is_empty() {
        body["stop"] = json!(request.stop);
    }
    if let Some(seed) = request.seed {
        body["seed"] = json!(seed);
    }
    if !request.tools.is_empty() {
        body["tools"] = Value::Array(request.tools.iter().map(map_tool).collect());
        body["tool_choice"] = map_tool_choice(&request.tool_choice);
    }
    if let Some(schema) = &request.json_schema {
        body["text"] = json!({
            "format": {
                "type": "json_schema",
                "name": "binder_output",
                "strict": true,
                "schema": schema
            }
        });
    } else if request.json_mode {
        body["text"] = json!({ "format": { "type": "json_object" } });
    }
    Ok(body)
}

fn map_tool(tool: &ToolDefinition) -> Value {
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters
    })
}

fn map_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::None => json!("none"),
        ToolChoice::Required => json!("required"),
        ToolChoice::Named(name) => json!({"type": "function", "name": name}),
    }
}

pub fn embeddings_body(model: &str, inputs: &[String]) -> Value {
    json!({ "model": model, "input": inputs })
}

#[derive(Debug, Deserialize)]
pub struct ResponsesApi {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub output: Vec<OutputItem>,
    #[serde(default)]
    pub usage: Option<ResponsesUsage>,
    #[serde(default)]
    pub error: Option<ApiError>,
}

#[derive(Debug, Deserialize)]
pub struct ApiError {
    #[serde(default)]
    pub message: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OutputItem {
    #[serde(default)]
    pub type_: Option<String>,
    #[serde(rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<Vec<ContentPart>>,
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
}

impl OutputItem {
    fn kind(&self) -> &str {
        self.kind.as_deref().or(self.type_.as_deref()).unwrap_or("")
    }
}

#[derive(Debug, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ResponsesUsage {
    #[serde(default)]
    pub input_tokens: Option<u32>,
    #[serde(default)]
    pub output_tokens: Option<u32>,
    #[serde(default)]
    pub total_tokens: Option<u32>,
    #[serde(default)]
    pub prompt_tokens: Option<u32>,
    #[serde(default)]
    pub completion_tokens: Option<u32>,
}

impl ResponsesApi {
    pub fn into_chat_response(self, default_model: &str, retries: u32) -> Result<ChatResponse> {
        if let Some(err) = self.error.and_then(|e| e.message) {
            return Err(LlmError::provider(Provider::Openai, err, None));
        }
        let model = self.model.unwrap_or_else(|| default_model.to_string());
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        for item in self.output {
            match item.kind() {
                "function_call" | "tool_call" => {
                    tool_calls.push(ToolCall::new(
                        item.call_id.unwrap_or_default(),
                        item.name.unwrap_or_default(),
                        item.arguments.unwrap_or_default(),
                    ));
                }
                _ => {
                    if let Some(t) = item.text {
                        text.push_str(&t);
                    }
                    if let Some(parts) = item.content {
                        for part in parts {
                            if let Some(t) = part.text {
                                text.push_str(&t);
                            }
                        }
                    }
                    let _ = item.role;
                }
            }
        }
        let finish_reason = if !tool_calls.is_empty() {
            FinishReason::ToolCalls
        } else if self.status.as_deref() == Some("incomplete") {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };
        let message = if tool_calls.is_empty() {
            ChatMessage::assistant(text)
        } else {
            ChatMessage::assistant_with_tools(text, tool_calls)
        };
        let mut response = ChatResponse::new(Provider::Openai, model, message);
        response.finish_reason = finish_reason;
        if let Some(usage) = self.usage {
            response.usage = Usage {
                prompt_tokens: usage.input_tokens.or(usage.prompt_tokens).unwrap_or(0),
                completion_tokens: usage.output_tokens.or(usage.completion_tokens).unwrap_or(0),
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
    pub usage: Option<ResponsesUsage>,
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

#[derive(Debug, Deserialize)]
pub struct SseEventPayload {
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub delta: Option<Value>,
    #[serde(default)]
    pub response: Option<ResponsesApi>,
    #[serde(default)]
    pub item: Option<OutputItem>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_system_to_instructions() {
        let req = ChatRequest::new(vec![
            ChatMessage::system("be brief"),
            ChatMessage::user("hi"),
        ]);
        let body = responses_body(&req, "gpt-4.1").unwrap();
        assert_eq!(body["instructions"], "be brief");
        assert_eq!(body["input"][0]["role"], "user");
    }

    #[test]
    fn json_mode_sets_format() {
        let req = ChatRequest::new(vec![ChatMessage::user("hi")]).with_json_mode(true);
        let body = responses_body(&req, "gpt-4.1").unwrap();
        assert_eq!(body["text"]["format"]["type"], "json_object");
    }
}
