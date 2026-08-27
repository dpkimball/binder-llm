use std::time::Duration;

use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use binder_llm::{ChatMessage, ChatRequest, LlmClient, Provider, RetryPolicy};

#[tokio::test]
async fn chat_completions() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "model": "grok-3",
            "choices": [{
                "message": {"role": "assistant", "content": "hello from grok"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6}
        })))
        .mount(&server)
        .await;

    let client = LlmClient::builder()
        .provider(Provider::Xai)
        .api_key("test-key")
        .base_url(server.uri())
        .model("grok-3")
        .retries(RetryPolicy {
            max_retries: 0,
            initial_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(1),
            multiplier: 2.0,
        })
        .build()
        .unwrap();
    let resp = client
        .chat(ChatRequest::new(vec![ChatMessage::user("hi")]))
        .await
        .unwrap();
    assert_eq!(resp.provider, Provider::Xai);
    assert_eq!(resp.message.content(), "hello from grok");
    assert_eq!(resp.usage.total_tokens, 6);
}
