use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::{LlmError, Result};
use crate::message::ChatMessage;
use crate::model::Provider;
use crate::request::ChatRequest;
use crate::response::{ChatResponse, FinishReason};
use crate::tool::{ToolCall, ToolChoice, ToolDefinition};
use crate::usage::Usage;

pub fn uses_generate(request: &ChatRequest) -> bool {
    if !request.tools.is_empty() {
        return false;
    }
    let mut systems = 0u32;
    let mut users = 0u32;
    for message in &request.messages {
        match message {
            ChatMessage::System { .. } => systems += 1,
            ChatMessage::User { .. } => users += 1,
            ChatMessage::Assistant { .. } | ChatMessage::Tool { .. } => return false,
        }
    }
    users == 1 && systems <= 1
}

pub fn chat_body(request: &ChatRequest, default_model: &str) -> Result<Value> {
    if request.messages.is_empty() {
        return Err(LlmError::invalid("chat request has no messages"));
    }
    let mut body = json!({
        "model": request.resolved_model(default_model),
        "messages": map_messages(&request.messages)?,
        "stream": request.stream,
    });
    if !request.tools.is_empty() {
        body["tools"] = Value::Array(request.tools.iter().map(map_tool).collect());
        if let Some(choice) = map_tool_choice(&request.tool_choice) {
            body["tool_choice"] = choice;
        }
    }
    apply_format(&mut body, request);
    apply_options(&mut body, request);
    Ok(body)
}

pub fn generate_body(request: &ChatRequest, default_model: &str) -> Result<Value> {
    if request.messages.is_empty() {
        return Err(LlmError::invalid("chat request has no messages"));
    }
    let mut system = None;
    let mut prompt = String::new();
    for message in &request.messages {
        match message {
            ChatMessage::System { content, .. } => system = Some(content.clone()),
            ChatMessage::User { content, .. } => prompt = content.clone(),
            _ => {
                return Err(LlmError::invalid(
                    "ollama generate requires system/user messages only",
                ))
            }
        }
    }
    let mut body = json!({
        "model": request.resolved_model(default_model),
        "prompt": prompt,
        "stream": request.stream,
    });
    let images: Vec<Value> = request
        .messages
        .iter()
        .flat_map(|message| message.images())
        .map(|image| Value::String(base64::engine::general_purpose::STANDARD.encode(&image.bytes)))
        .collect();
    if !images.is_empty() {
        body["images"] = Value::Array(images);
    }
    if let Some(system) = system {
        body["system"] = Value::String(system);
    }
    apply_format(&mut body, request);
    apply_options(&mut body, request);
    Ok(body)
}

pub fn embed_body(model: &str, inputs: &[String]) -> Value {
    if inputs.len() == 1 {
        json!({ "model": model, "input": inputs[0] })
    } else {
        json!({ "model": model, "input": inputs })
    }
}

fn apply_format(body: &mut Value, request: &ChatRequest) {
    if let Some(schema) = &request.json_schema {
        body["format"] = schema.clone();
    } else if request.json_mode {
        body["format"] = Value::String("json".into());
    }
}

fn apply_options(body: &mut Value, request: &ChatRequest) {
    let mut options = serde_json::Map::new();
    if let Some(t) = request.temperature {
        options.insert("temperature".into(), json!(t));
    }
    if let Some(max) = request.max_tokens {
        options.insert("num_predict".into(), json!(max));
    }
    if let Some(seed) = request.seed {
        options.insert("seed".into(), json!(seed));
    }
    if !request.stop.is_empty() {
        options.insert("stop".into(), json!(request.stop));
    }
    if !options.is_empty() {
        body["options"] = Value::Object(options);
    }
}

fn map_messages(messages: &[ChatMessage]) -> Result<Vec<Value>> {
    messages.iter().map(map_message).collect()
}

fn map_message(message: &ChatMessage) -> Result<Value> {
    Ok(match message {
        ChatMessage::System { content, .. } => json!({"role": "system", "content": content}),
        ChatMessage::User {
            content, images, ..
        } => {
            let mut value = json!({"role": "user", "content": content});
            if !images.is_empty() {
                value["images"] = Value::Array(
                    images
                        .iter()
                        .map(|image| {
                            Value::String(
                                base64::engine::general_purpose::STANDARD.encode(&image.bytes),
                            )
                        })
                        .collect(),
                );
            }
            value
        }
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
                        .map(|call| {
                            let args = parse_args(&call.arguments);
                            json!({
                                "id": call.id,
                                "function": {
                                    "name": call.name,
                                    "arguments": args
                                }
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
            "tool_name": tool_call_id,
            "content": content
        }),
    })
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

fn map_tool_choice(choice: &ToolChoice) -> Option<Value> {
    match choice {
        ToolChoice::Auto => None,
        ToolChoice::None => Some(json!("none")),
        ToolChoice::Required => Some(json!("required")),
        ToolChoice::Named(name) => Some(json!({"type": "function", "function": {"name": name}})),
    }
}

fn parse_args(arguments: &str) -> Value {
    serde_json::from_str(arguments).unwrap_or_else(|_| json!({}))
}

#[derive(Debug, Deserialize)]
pub struct ChatApiResponse {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub message: Option<ApiMessage>,
    #[serde(default)]
    pub response: Option<String>,
    #[serde(default)]
    pub done: bool,
    #[serde(default)]
    pub done_reason: Option<String>,
    #[serde(default)]
    pub prompt_eval_count: Option<u32>,
    #[serde(default)]
    pub eval_count: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ApiMessage {
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<ApiToolCall>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ApiToolCall {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<ApiFunction>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ApiFunction {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<Value>,
}

impl ChatApiResponse {
    pub fn into_chat_response(self, default_model: &str, retries: u32) -> Result<ChatResponse> {
        let model = self
            .model
            .clone()
            .unwrap_or_else(|| default_model.to_string());
        let tool_calls = self
            .message
            .as_ref()
            .and_then(|m| m.tool_calls.as_ref())
            .map(|calls| {
                calls
                    .iter()
                    .enumerate()
                    .map(|(i, call)| {
                        let name = call
                            .function
                            .as_ref()
                            .and_then(|f| f.name.clone())
                            .unwrap_or_default();
                        let args = call
                            .function
                            .as_ref()
                            .and_then(|f| f.arguments.clone())
                            .unwrap_or(json!({}));
                        let id = call
                            .id
                            .clone()
                            .unwrap_or_else(|| format!("ollama_tool_{i}"));
                        ToolCall::new(id, name, args.to_string())
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let content = self
            .message
            .as_ref()
            .and_then(|m| m.content.clone())
            .or(self.response.clone())
            .unwrap_or_default();
        let finish_reason = if !tool_calls.is_empty() {
            FinishReason::ToolCalls
        } else {
            self.done_reason
                .as_deref()
                .map(FinishReason::from_provider_str)
                .unwrap_or(FinishReason::Stop)
        };
        let message = if tool_calls.is_empty() {
            ChatMessage::assistant(content)
        } else {
            ChatMessage::assistant_with_tools(content, tool_calls)
        };
        let mut response = ChatResponse::new(Provider::Ollama, model, message);
        response.finish_reason = finish_reason;
        response.usage = Usage::new(
            self.prompt_eval_count.unwrap_or(0),
            self.eval_count.unwrap_or(0),
        );
        response.retries = retries;
        Ok(response)
    }

    pub fn content_delta(&self) -> String {
        self.message
            .as_ref()
            .and_then(|m| m.content.clone())
            .or(self.response.clone())
            .unwrap_or_default()
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbedApiResponse {
    #[serde(default)]
    pub embeddings: Vec<Vec<f32>>,
    #[serde(default)]
    pub embedding: Option<Vec<f32>>,
    #[serde(default)]
    pub model: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::ChatImage;
    use crate::tool::ToolDefinition;

    #[test]
    fn generate_for_single_user() {
        let req = ChatRequest::new(vec![ChatMessage::user("hi")]);
        assert!(uses_generate(&req));
        let body = generate_body(&req, "qwen").unwrap();
        assert_eq!(body["prompt"], "hi");
    }

    #[test]
    fn chat_when_tools_present() {
        let req =
            ChatRequest::new(vec![ChatMessage::user("hi")]).with_tools(vec![ToolDefinition::new(
                "t",
                "d",
                json!({"type": "object"}),
            )]);
        assert!(!uses_generate(&req));
        let body = chat_body(&req, "qwen").unwrap();
        assert!(body["tools"].is_array());
    }

    #[test]
    fn json_mode_sets_format() {
        let req = ChatRequest::new(vec![ChatMessage::user("hi")]).with_json_mode(true);
        let body = chat_body(&req, "qwen").unwrap();
        assert_eq!(body["format"], "json");
    }

    #[test]
    fn user_image_is_base64_on_the_generate_body() {
        let png = vec![0x89, 0x50, 0x4e, 0x47];
        let req = ChatRequest::new(vec![ChatMessage::user_with_images(
            "Is this a company logo?",
            vec![ChatImage {
                media_type: "image/png".into(),
                bytes: png.clone(),
            }],
        )]);
        assert!(uses_generate(&req));
        let body = generate_body(&req, "moondream").unwrap();
        let encoded = body["images"][0].as_str().unwrap();
        assert_eq!(
            encoded,
            base64::engine::general_purpose::STANDARD.encode(png)
        );
    }
}
