---
id: private-source-public-binary
title: binder-llm — private source / public binary distribution
status: superseded
created: 2026-09-02
updated: 2026-10-05
owner: binder-llm
github_issue: https://github.com/dpkimball/binder-llm/issues/6
related:
  - https://github.com/dpkimball/binder-llm/issues/6
  - https://github.com/dpkimball/bindb/issues/97
  - https://github.com/dpkimball/binder-workspace/issues/194
---

# binder-llm — private source / public binary distribution

> **Superseded for source licensing.** The source repository is MIT (`LICENSE`). This document describes the earlier private-source / public-binary channel. The readiness checklist lives in the codeintel repo at `docs/public-release-readiness.md`.

## Purpose

Publish a **public compiled binary** for install/verify without exposing the private source repository. Mirror bindb Phase 1 ([bindb#97](https://github.com/dpkimball/bindb/issues/97)).

## Desired behavior

### Public executable contract

```text
binder-llm --version
binder-llm --help
```

`--version` prints (stdout):

```text
binder-llm <semver>
commit: <7-char sha or unknown>
built: <YYYY-MM-DD or unknown>
```

No private repository URLs. The CLI does **not** perform inference in Phase 1 — applications embed the crate (Kellnr). The binary exists so distribution/verification can be proven.

### Release automation

On push of tag `v*`, [dist](https://github.com/axodotdev/cargo-dist) runs `.github/workflows/release.yml`:

1. Build for `aarch64-apple-darwin`, `x86_64-unknown-linux-gnu`
2. Package archives with checksums + `packaging/LICENSE.txt` + `packaging/README.txt`
3. Publish to **`dpkimball/binder-llm-releases`** only when all platforms succeed

### Operator setup

1. Public repo: [`dpkimball/binder-llm-releases`](https://github.com/dpkimball/binder-llm-releases)
2. Fine-grained PAT with Contents: Read and write on that repo only
3. Actions secret **`GH_RELEASES_TOKEN`** on private `dpkimball/binder-llm`
4. Tag matching `Cargo.toml` version

Public install docs must match cargo-dist naming: `binder-llm-<rust-target-triple>.tar.xz`, per-archive `*.tar.xz.sha256`, and `sha256.sum`.

### cargo-dist announcement URL drift

`github-releases-repo` uploads to the public repo, but cargo-dist still embeds **private** `dpkimball/binder-llm` download URLs in the release body and `dist-manifest.json` hosting block (upstream: [axodotdev/cargo-dist#2252](https://github.com/axodotdev/cargo-dist/issues/2252)).

Mitigation: `.github/workflows/fix-binder-llm-releases-announcement.yml` rewrites those URLs after a successful `Release` (requires `GH_RELEASES_TOKEN`).

## Out of scope

Open-sourcing the private repo; expanding CLI into a full chat client; changing Kellnr publish for DPK embeds.
