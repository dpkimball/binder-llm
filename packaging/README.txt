binder-llm
==========

Official compiled release of binder-llm.

binder-llm is a provider-agnostic LLM transport library. This archive contains
the compiled executable. Source is MIT licensed in
https://github.com/dpkimball/binder-llm.

IMPORTANT
---------

This binary does NOT run inference (--version / --help only).
Applications embed the private Rust crate, or use binder-ai-plan from
https://github.com/dpkimball/binder-ai-releases

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

Docs
----

https://github.com/dpkimball/binder-llm-releases

License
-------

See LICENSE.txt in this archive (MIT).

Support
-------

https://github.com/dpkimball/binder-llm-releases
