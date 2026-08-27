use std::time::Duration;

use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use binder_llm::{ChatMessage, ChatRequest, EmbeddingRequest, LlmClient, Provider, RetryPolicy};

fn retries() -> RetryPolicy {
    RetryPolicy {
        max_retries: 0,
        initial_backoff: Duration::from_millis(1),
        max_backoff: Duration::from_millis(1),
        multiplier: 2.0,
    }
}

#[tokio::test]
async fn responses_api_chat_and_tools() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer test-key"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "gpt-4.1",
            "status": "completed",
            "output": [
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]},
                {"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{\"q\":\"x\"}"}
            ],
            "usage": {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10}
        })))
        .mount(&server)
        .await;

    let client = LlmClient::builder()
        .provider(Provider::Openai)
        .api_key("test-key")
        .base_url(server.uri())
        .model("gpt-4.1")
        .retries(retries())
        .build()
        .unwrap();
    let resp = client
        .chat(ChatRequest::new(vec![ChatMessage::user("hi")]))
        .await
        .unwrap();
    assert_eq!(resp.provider, Provider::Openai);
    assert_eq!(resp.message.content(), "hello");
    assert_eq!(resp.message.tool_calls()[0].name, "lookup");
    assert_eq!(resp.usage.total_tokens, 10);
}

#[tokio::test]
async fn embeddings() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "text-embedding-3-small",
            "data": [{"embedding": [0.1, 0.2]}],
            "usage": {"prompt_tokens": 2}
        })))
        .mount(&server)
        .await;
    let client = LlmClient::builder()
        .provider(Provider::Openai)
        .api_key("test-key")
        .base_url(server.uri())
        .model("text-embedding-3-small")
        .retries(retries())
        .build()
        .unwrap();
    let resp = client
        .embeddings(EmbeddingRequest::new("hello"))
        .await
        .unwrap();
    assert_eq!(resp.embeddings[0], vec![0.1, 0.2]);
}
