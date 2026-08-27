use std::time::Duration;

use serde_json::json;
use wiremock::matchers::{method, path};
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
async fn generate_content() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/gemini-2.0-flash:generateContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "candidates": [{
                "content": {"parts": [{"text": "hello gemini"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3, "totalTokenCount": 5}
        })))
        .mount(&server)
        .await;

    let client = LlmClient::builder()
        .provider(Provider::Gemini)
        .api_key("test-key")
        .base_url(server.uri())
        .model("gemini-2.0-flash")
        .retries(retries())
        .build()
        .unwrap();
    let resp = client
        .chat(ChatRequest::new(vec![ChatMessage::user("hi")]))
        .await
        .unwrap();
    assert_eq!(resp.provider, Provider::Gemini);
    assert_eq!(resp.message.content(), "hello gemini");
    assert_eq!(resp.usage.total_tokens, 5);
}

#[tokio::test]
async fn embed_content() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/models/text-embedding-004:embedContent"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embedding": {"values": [0.4, 0.5]}
        })))
        .mount(&server)
        .await;
    let client = LlmClient::builder()
        .provider(Provider::Gemini)
        .api_key("test-key")
        .base_url(server.uri())
        .model("text-embedding-004")
        .retries(retries())
        .build()
        .unwrap();
    let resp = client
        .embeddings(EmbeddingRequest::new("hello"))
        .await
        .unwrap();
    assert_eq!(resp.embeddings[0], vec![0.4, 0.5]);
}
