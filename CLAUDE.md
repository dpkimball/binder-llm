# CLAUDE.md

Guidance for agents working in `binder-llm`.

## What this crate is

Infrastructure-only LLM transport for Binder. The rest of the ecosystem must depend on `LlmClient` and provider-neutral types. Do not add prompts, agents, RAG, MCP, Binder domain logic, bindb queries, or repository discovery here. Architectural context is supplied by binder-ai.

## Layout

```text
src/
  client.rs, config.rs, provider.rs   # LlmClient + LlmProvider
  message.rs, request.rs, response.rs, stream.rs, tool.rs
  providers/{ollama,openai,anthropic,xai,gemini}/
```

New providers get their own module behind a Cargo feature. Map provider JSON in `mapper.rs`; never leak it from the public API.

## Build

```sh
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt -- --check
make b
```

Default feature is `ollama` only. CI clippy uses `--all-features`.

## Logging

`tracing` metadata only: provider, model, token counts, latency, retries, error kind. Never log prompts, completions, API keys, or Authorization headers.

## Publish

- **Crate (embed):** private Kellnr (`CARGO_REGISTRY=dpk`). `env.sh` sets `PUBLISH_CRATE=true` for local `make b`.
- **Public binary:** cargo-dist → `dpkimball/binder-llm-releases` (see `docs/specs/private-source-public-binary.md`).
