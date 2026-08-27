use std::time::Duration;

use futures::StreamExt;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use binder_llm::{
    ChatMessage, ChatRequest, EmbeddingRequest, LlmClient, Provider, RetryPolicy, StreamChunk,
    ToolDefinition,
};

async fn client(server: &MockServer) -> LlmClient {
    LlmClient::builder()
        .provider(Provider::Ollama)
        .base_url(server.uri())
        .model("qwen3-coder:30b")
        .retries(RetryPolicy {
            max_retries: 2,
            initial_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(10),
            multiplier: 2.0,
        })
        .build()
        .unwrap()
}

#[tokio::test]
async fn generate_single_user() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "qwen3-coder:30b",
            "response": "hi there",
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 3,
            "eval_count": 2
        })))
        .mount(&server)
        .await;

    let client = client(&server).await;
    let resp = client
        .chat(ChatRequest::new(vec![ChatMessage::user("hello")]))
        .await
        .unwrap();
    assert_eq!(resp.message.content(), "hi there");
    assert_eq!(resp.usage.prompt_tokens, 3);
    assert_eq!(resp.provider, Provider::Ollama);
}

#[tokio::test]
async fn chat_with_tools() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "qwen3-coder:30b",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "lookup",
                        "arguments": {"q": "abc"}
                    }
                }]
            },
            "done": true
        })))
        .mount(&server)
        .await;

    let client = client(&server).await;
    let resp =
        client
            .chat(
                ChatRequest::new(vec![ChatMessage::user("go")]).with_tools(vec![
                    ToolDefinition::new("lookup", "look things up", json!({"type":"object"})),
                ]),
            )
            .await
            .unwrap();
    assert_eq!(resp.finish_reason.to_string(), "tool_calls");
    assert_eq!(resp.message.tool_calls()[0].name, "lookup");
}

#[tokio::test]
async fn stream_chat() {
    let server = MockServer::start().await;
    let body = [
        json!({"message":{"role":"assistant","content":"hel"},"done":false}).to_string(),
        json!({"message":{"role":"assistant","content":"lo"},"done":false}).to_string(),
        json!({"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"})
            .to_string(),
    ]
    .join("\n");
    Mock::given(method("POST"))
        .and(path("/api/chat"))
        .respond_with(ResponseTemplate::new(200).set_body_raw(body, "application/x-ndjson"))
        .mount(&server)
        .await;

    let client = client(&server).await;
    let mut stream = client
        .stream(ChatRequest::new(vec![
            ChatMessage::system("s"),
            ChatMessage::user("u"),
            ChatMessage::assistant("a"),
        ]))
        .await
        .unwrap();
    let mut text = String::new();
    while let Some(item) = stream.next().await {
        if let StreamChunk::ContentDelta { text: d } = item.unwrap() {
            text.push_str(&d);
        }
    }
    assert_eq!(text, "hello");
}

#[tokio::test]
async fn tags_and_embed() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/api/tags"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "models": [{"name": "qwen3-coder:30b"}, {"name": "nomic-embed-text"}]
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/embed"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "nomic-embed-text",
            "embeddings": [[0.1, 0.2, 0.3]]
        })))
        .mount(&server)
        .await;

    let client = client(&server).await;
    let models = client.list_models().await.unwrap();
    assert_eq!(models.len(), 2);
    assert!(models.iter().all(|m| m.local));
    let embed = client
        .embeddings(EmbeddingRequest::new("hello").with_model("nomic-embed-text"))
        .await
        .unwrap();
    assert_eq!(embed.embeddings[0].len(), 3);
}

#[tokio::test]
async fn retries_on_429() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({"error":"slow down"})))
        .up_to_n_times(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "response": "ok",
            "done": true
        })))
        .mount(&server)
        .await;

    let client = client(&server).await;
    let resp = client
        .chat(ChatRequest::new(vec![ChatMessage::user("hello")]))
        .await
        .unwrap();
    assert_eq!(resp.message.content(), "ok");
    assert!(resp.retries >= 1);
}

#[tokio::test]
async fn auth_error_is_unified() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({"error":"nope"})))
        .mount(&server)
        .await;
    let client = client(&server).await;
    let err = client
        .chat(ChatRequest::new(vec![ChatMessage::user("hello")]))
        .await
        .unwrap_err();
    assert_eq!(err.kind(), "authentication");
}
