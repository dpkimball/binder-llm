# binder-llm as-built

`binder-llm` is a Rust library. `LlmClient` is the only type applications should construct. Providers implement `LlmProvider` behind Cargo features.

```mermaid
flowchart LR
  app[Binder apps / plugins / services]
  client[LlmClient]
  trait[LlmProvider]
  ollama[Ollama]
  openai[OpenAI Responses]
  anthropic[Anthropic Messages]
  xai[xAI chat completions]
  gemini[Gemini generateContent]

  app --> client --> trait
  trait --> ollama
  trait --> openai
  trait --> anthropic
  trait --> xai
  trait --> gemini
```

Default feature: `ollama`. Cloud providers are opt-in. Configuration (`Provider`, API key, base URL, model) is the swap point — request/response types do not change.
