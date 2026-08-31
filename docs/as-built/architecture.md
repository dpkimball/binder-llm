---
title: binder-llm as-built
description: As-built architecture of the binder-llm crate, describing LlmClient, providers, and the data flow from applications to inference.
category: architecture
---

# binder-llm as-built

`binder-llm` is a Rust library. **binder-ai** constructs [`LlmClient`]. Applications do not call providers or this crate directly.

`LlmClient` is the swap point for the reasoning engine. It runs inference on supplied messages. It does not query bindb or walk repositories.

```mermaid
flowchart LR
  apps[Binder applications]
  ai[binder-ai orchestration]
  client[LlmClient]
  trait[LlmProvider]
  ollama[Ollama]
  openai[OpenAI Responses]
  anthropic[Anthropic Messages]
  xai[xAI chat completions]
  gemini[Gemini generateContent]

  apps --> ai --> client --> trait
  trait --> ollama
  trait --> openai
  trait --> anthropic
  trait --> xai
  trait --> gemini
```

Default feature: `ollama`. Cloud providers are opt-in. Configuration (`Provider`, API key, base URL, model) is the swap point — request/response types do not change.
