use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;

use binder_llm::{
    ChatMessage, ChatRequest, ChatResponse, ChatStream, EmbeddingRequest, EmbeddingResponse,
    FinishReason, LlmClient, LlmConfig, MockProvider, ModelInfo, Provider, StreamChunk, Usage,
};

fn test_config() -> LlmConfig {
    LlmClient::builder()
        .provider(Provider::Ollama)
        .model("mock-model")
        .build_config()
        .expect("config")
}

#[tokio::test]
async fn mock_chat_and_json() {
    #[derive(serde::Deserialize, Debug, PartialEq)]
    struct Out {
        ok: bool,
    }

    let provider = MockProvider::new().with_chat(ChatResponse::new(
        Provider::Ollama,
        "mock-model",
        ChatMessage::assistant(r#"{"ok":true}"#),
    ));
    let client = LlmClient::from_provider(Arc::new(provider), test_config());
    let parsed: Out = client
        .chat_json(ChatRequest::new(vec![ChatMessage::user("hi")]))
        .await
        .unwrap();
    assert_eq!(parsed, Out { ok: true });
}

#[tokio::test]
async fn mock_stream() {
    let provider = MockProvider::new().with_stream(vec![
        StreamChunk::ContentDelta { text: "hel".into() },
        StreamChunk::ContentDelta { text: "lo".into() },
        StreamChunk::Finished {
            reason: FinishReason::Stop,
            message: None,
        },
    ]);
    let client = LlmClient::from_provider(Arc::new(provider), test_config());
    let mut stream: ChatStream = client
        .stream(ChatRequest::new(vec![ChatMessage::user("hi")]))
        .await
        .unwrap();
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        if let StreamChunk::ContentDelta { text: d } = chunk.unwrap() {
            text.push_str(&d);
        }
    }
    assert_eq!(text, "hello");
}

#[tokio::test]
async fn mock_embeddings_and_models() {
    let provider = MockProvider::new()
        .with_embeddings(EmbeddingResponse {
            embeddings: vec![vec![0.1, 0.2]],
            model: "mock-model".into(),
            provider: Provider::Ollama,
            usage: Usage::default(),
            latency: Duration::ZERO,
            retries: 0,
        })
        .with_models(vec![ModelInfo::new(Provider::Ollama, "mock-model")]);
    let client = LlmClient::from_provider(Arc::new(provider), test_config());
    let embed = client
        .embeddings(EmbeddingRequest::new("hello"))
        .await
        .unwrap();
    assert_eq!(embed.embeddings[0].len(), 2);
    let models = client.list_models().await.unwrap();
    assert_eq!(models.len(), 1);
}
