# binder-llm

github_issue: https://github.com/dpkimball/binder-llm/issues/1

Provider-agnostic LLM infrastructure crate for Binder.

Applications, crates, plugins, and services must not call OpenAI, Anthropic, Ollama, xAI, Gemini, or future providers directly. All LLM traffic flows through `LlmClient`.

## In scope

- Provider-neutral chat, streaming, embeddings, tool calling, JSON mode
- Feature-flagged providers: `ollama` (default), `openai`, `anthropic`, `xai`, `gemini`
- Unified errors, infrastructure retries (429/5xx), tracing metadata, call metrics types
- Easy mocks via `MockProvider`

## Out of scope

Prompts, agents, planning, memory, vector search, RAG, MCP, GitHub integration, tool execution, workflow orchestration, conversation persistence, Binder-specific intelligence.

## Swap contract

Consumers change only configuration (`Provider`, credentials, base URL, model). Request/response types stay the same.
