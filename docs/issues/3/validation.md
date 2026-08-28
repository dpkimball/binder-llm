# Validation — issue #3

Documentation only. No runtime change.

- README states binder-llm is the replaceable reasoning engine and does not walk repos or query bindb.
- Spec `docs/specs/binder-llm.md` links binder-ai#14 / `responsibilities.md`.
- As-built diagram: applications → binder-ai → `LlmClient` → providers.
- CLAUDE.md forbids bindb queries and repository discovery.

Out of scope (prompts, agents, RAG, workflows) unchanged.
