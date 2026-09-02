binder-llm
==========

Official compiled release of binder-llm.

binder-llm is a provider-agnostic LLM transport library. This archive contains
the public executable for install/verify. Source code is privately maintained
and is not included. Applications should embed the Rust crate for inference.

Installation
------------

1. Verify the archive against the matching `*.tar.xz.sha256` file from the
   GitHub Release (or use the release’s `sha256.sum`).
2. Extract this archive (`tar -xJf …` for `.tar.xz`).
3. Place `binder-llm` (or `binder-llm.exe`) on your PATH.

Verify
------

    binder-llm --version
    binder-llm --help

Phase 1 of the public binary supports only `--version` and `--help`.
It does not run inference. Embed the private Rust crate for LLM transport.

License
-------

See LICENSE.txt in this archive. This is proprietary binary distribution,
not open-source software.

Support
-------

See https://github.com/dpkimball/binder-llm-releases
