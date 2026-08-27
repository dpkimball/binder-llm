use std::time::Duration;

use serde_json::json;
use wiremock::matchers::{header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use binder_llm::{ChatMessage, ChatRequest, LlmClient, Provider, RetryPolicy};

#[tokio::test]
async fn messages_api() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "test-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "claude-sonnet-4-20250514",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 8, "output_tokens": 12}
        })))
        .mount(&server)
        .await;

    let client = LlmClient::builder()
        .provider(Provider::Anthropic)
        .api_key("test-key")
        .base_url(server.uri())
        .model("claude-sonnet-4-20250514")
        .retries(RetryPolicy {
            max_retries: 0,
            initial_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(1),
            multiplier: 2.0,
        })
        .build()
        .unwrap();
    let resp = client
        .chat(ChatRequest::new(vec![
            ChatMessage::system("sys"),
            ChatMessage::user("hi"),
        ]))
        .await
        .unwrap();
    assert_eq!(resp.provider, Provider::Anthropic);
    assert_eq!(resp.message.content(), "hello");
    assert_eq!(resp.message.tool_calls()[0].id, "toolu_1");
    assert_eq!(resp.finish_reason.to_string(), "tool_calls");
}
