# Project Guidelines

## Build & Test Commands

- `cargo build` — build the project
- `cargo test` — run all tests
- `cargo clippy --all-targets -- -D warnings` — lint (must pass with zero warnings)
- `cargo fmt --check` — check formatting

## Pre-commit Checks

Before committing, always run:

```sh
cargo fmt && cargo clippy --all-targets -- -D warnings && cargo test
```
