# binder-llm

[![CI](https://github.com/dpkimball/binder-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/dpkimball/binder-llm/actions/workflows/ci.yml)

Canonical LLM abstraction for the Binder platform.

No application, crate, plugin, or service should call OpenAI, Anthropic, Ollama, xAI, Gemini, or any future provider directly. All LLM traffic goes through `binder-llm`.

This crate is the replaceable **reasoning engine**: it runs model inference on **supplied** messages. It does not discover repositories, query bindb, own prompts, or orchestrate workflows. Those belong to bindb (facts) and binder-ai (orchestration). See binder-ai `docs/specs/responsibilities.md`.

This crate is infrastructure only: no prompts, no Binder business logic, no agents, no RAG, no tool execution.

## Role

- Provider-agnostic `LlmClient`
- Strongly typed chat, streaming, tools, embeddings, and JSON mode
- Feature-flagged providers (`ollama` default)
- Unified errors and retries (429 / 5xx, exponential backoff)
- `tracing` metadata only (never prompts, completions, or secrets)

## Usage

```rust
use binder_llm::{ChatMessage, ChatRequest, LlmClient, Provider};

# async fn demo() -> binder_llm::Result<()> {
let client = LlmClient::builder()
    .provider(Provider::Ollama)
    .base_url("http://localhost:11434")
    .model("qwen3-coder:30b")
    .build()?;

let response = client
    .chat(ChatRequest::new(vec![ChatMessage::user("hello")]))
    .await?;
# let _ = response;
# Ok(())
# }
```

Swap providers by configuration only:

```rust
# use binder_llm::{LlmClient, Provider};
# fn prod() -> binder_llm::Result<LlmClient> {
LlmClient::builder()
    .provider(Provider::Openai)
    .api_key(std::env::var("OPENAI_API_KEY").unwrap())
    .model("gpt-4.1")
    .build()
# }
```

Consumers should depend on `LlmClient` and the provider-neutral request/response types. Tests can inject [`MockProvider`](https://docs.rs/binder-llm) via `LlmClient::from_provider`.

## Features

| Feature | Default | Provider |
|---------|---------|----------|
| `ollama` | yes | Ollama `/api/chat`, `/api/generate`, `/api/tags`, `/api/embed` |
| `openai` | | OpenAI Responses API, embeddings, streaming, tools, structured outputs |
| `anthropic` | | Anthropic Messages API, streaming, tool use |
| `xai` | | xAI (OpenAI-compatible chat completions) |
| `gemini` | | Gemini `generateContent` / embeddings |
| `all-providers` | | All of the above |

## Develop

Sibling checkout with shared build tooling (optional):

```text
workspace/
  dpk-build/   # or any BUILD_ROOT that provides Makefile.common
  binder-llm/
```

```sh
cargo test
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
make b   # lint + test; publishes to private Kellnr when configured
```

**Embed path:** private Kellnr (`publish = ["dpk"]`) — not crates.io.  
**Public binary path:** tag `v*` → [`binder-llm-releases`](https://github.com/dpkimball/binder-llm-releases). See [`docs/specs/private-source-public-binary.md`](docs/specs/private-source-public-binary.md).

## Out of scope

Prompt templates, agents, planning, memory, vector search, RAG, MCP, GitHub integration, tool execution, workflow orchestration, conversation persistence.

## License

See repository license / parent workspace policy.
