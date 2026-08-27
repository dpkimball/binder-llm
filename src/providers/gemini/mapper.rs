use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::error::{LlmError, Result};
use crate::message::ChatMessage;
use crate::model::Provider;
use crate::request::ChatRequest;
use crate::response::{ChatResponse, FinishReason};
use crate::tool::{ToolCall, ToolChoice, ToolDefinition};
use crate::usage::Usage;

use super::models::strip_models_prefix;

pub fn generate_body(request: &ChatRequest, default_model: &str) -> Result<Value> {
    let _ = default_model;
    if request.messages.is_empty() {
        return Err(LlmError::invalid("chat request has no messages"));
    }
    let mut system = Vec::new();
    let mut contents = Vec::new();
    for message in &request.messages {
        match message {
            ChatMessage::System { content, .. } => system.push(content.clone()),
            ChatMessage::User { content, .. } => {
                contents.push(json!({"role": "user", "parts": [{"text": content}]}));
            }
            ChatMessage::Assistant {
                content,
                tool_calls,
                ..
            } => {
                let mut parts = Vec::new();
                if !content.is_empty() {
                    parts.push(json!({"text": content}));
                }
                for call in tool_calls {
                    let args = serde_json::from_str(&call.arguments).unwrap_or(json!({}));
                    parts.push(json!({"functionCall": {"name": call.name, "args": args}}));
                }
                contents.push(json!({"role": "model", "parts": parts}));
            }
            ChatMessage::Tool {
                tool_call_id,
                content,
                ..
            } => {
                let parsed = serde_json::from_str::<Value>(content)
                    .unwrap_or_else(|_| json!({"result": content}));
                contents.push(json!({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": tool_call_id,
                            "response": parsed
                        }
                    }]
                }));
            }
        }
    }
    let mut body = json!({ "contents": contents });
    if !system.is_empty() {
        body["systemInstruction"] = json!({"parts": [{"text": system.join("\n")}]});
    }
    let mut gen = Map::new();
    if let Some(t) = request.temperature {
        gen.insert("temperature".into(), json!(t));
    }
    if let Some(max) = request.max_tokens {
        gen.insert("maxOutputTokens".into(), json!(max));
    }
    if let Some(seed) = request.seed {
        gen.insert("seed".into(), json!(seed));
    }
    if !request.stop.is_empty() {
        gen.insert("stopSequences".into(), json!(request.stop));
    }
    if request.json_mode {
        gen.insert("responseMimeType".into(), json!("application/json"));
        if let Some(schema) = &request.json_schema {
            gen.insert("responseSchema".into(), schema.clone());
        }
    }
    if !gen.is_empty() {
        body["generationConfig"] = Value::Object(gen);
    }
    if !request.tools.is_empty() {
        body["tools"] = json!([{
            "functionDeclarations": request.tools.iter().map(map_tool).collect::<Vec<_>>()
        }]);
        body["toolConfig"] = json!({
            "functionCallingConfig": map_tool_choice(&request.tool_choice)
        });
    }
    Ok(body)
}

fn map_tool(tool: &ToolDefinition) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters
    })
}

fn map_tool_choice(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!({"mode": "AUTO"}),
        ToolChoice::None => json!({"mode": "NONE"}),
        ToolChoice::Required => json!({"mode": "ANY"}),
        ToolChoice::Named(name) => json!({"mode": "ANY", "allowedFunctionNames": [name]}),
    }
}

pub fn embed_body(text: &str) -> Value {
    json!({"content": {"parts": [{"text": text}]}})
}

#[derive(Debug, Deserialize)]
pub struct GenerateApi {
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    #[serde(default)]
    pub usage_metadata: Option<GeminiUsage>,
    #[serde(default, rename = "usageMetadata")]
    pub usage_metadata_camel: Option<GeminiUsage>,
}

#[derive(Debug, Deserialize)]
pub struct Candidate {
    #[serde(default)]
    pub content: Option<GeminiContent>,
    #[serde(default, rename = "finishReason")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct GeminiContent {
    #[serde(default)]
    pub parts: Vec<GeminiPart>,
}

#[derive(Debug, Deserialize)]
pub struct GeminiPart {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default, rename = "functionCall")]
    pub function_call: Option<GeminiFunctionCall>,
}

#[derive(Debug, Deserialize)]
pub struct GeminiFunctionCall {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub args: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct GeminiUsage {
    #[serde(default, rename = "promptTokenCount")]
    pub prompt_token_count: Option<u32>,
    #[serde(default, rename = "candidatesTokenCount")]
    pub candidates_token_count: Option<u32>,
    #[serde(default, rename = "totalTokenCount")]
    pub total_token_count: Option<u32>,
}

impl GenerateApi {
    pub fn into_chat_response(self, model: &str, retries: u32) -> Result<ChatResponse> {
        let candidate = self
            .candidates
            .into_iter()
            .next()
            .ok_or_else(|| LlmError::provider(Provider::Gemini, "empty candidates", None))?;
        let mut text = String::new();
        let mut tool_calls = Vec::new();
        if let Some(content) = candidate.content {
            for (i, part) in content.parts.into_iter().enumerate() {
                if let Some(t) = part.text {
                    text.push_str(&t);
                }
                if let Some(call) = part.function_call {
                    let args = call.args.unwrap_or(json!({}));
                    tool_calls.push(ToolCall::new(
                        format!("gemini_tool_{i}"),
                        call.name.unwrap_or_default(),
                        args.to_string(),
                    ));
                }
            }
        }
        let finish_reason = if !tool_calls.is_empty() {
            FinishReason::ToolCalls
        } else {
            candidate
                .finish_reason
                .as_deref()
                .map(FinishReason::from_provider_str)
                .unwrap_or(FinishReason::Stop)
        };
        let message = if tool_calls.is_empty() {
            ChatMessage::assistant(text)
        } else {
            ChatMessage::assistant_with_tools(text, tool_calls)
        };
        let mut response = ChatResponse::new(Provider::Gemini, strip_models_prefix(model), message);
        response.finish_reason = finish_reason;
        if let Some(usage) = self.usage_metadata.or(self.usage_metadata_camel) {
            response.usage = Usage {
                prompt_tokens: usage.prompt_token_count.unwrap_or(0),
                completion_tokens: usage.candidates_token_count.unwrap_or(0),
                total_tokens: usage.total_token_count.unwrap_or(0),
            };
        }
        response.retries = retries;
        Ok(response)
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbedApi {
    #[serde(default)]
    pub embedding: Option<EmbedValues>,
}

#[derive(Debug, Deserialize)]
pub struct EmbedValues {
    #[serde(default)]
    pub values: Vec<f32>,
}

#[derive(Debug, Deserialize)]
pub struct ModelsApi {
    #[serde(default)]
    pub models: Vec<ModelRow>,
}

#[derive(Debug, Deserialize)]
pub struct ModelRow {
    pub name: String,
    #[serde(default, rename = "displayName")]
    pub display_name: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_system_instruction() {
        let req = ChatRequest::new(vec![ChatMessage::system("sys"), ChatMessage::user("hi")]);
        let body = generate_body(&req, "gemini-2.0-flash").unwrap();
        assert!(body["systemInstruction"].is_object());
        assert_eq!(body["contents"][0]["role"], "user");
    }
}
