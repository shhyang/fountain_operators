# Committed operation traces (Phase 4)

JSON traces captured from real encode/decode runs. Regenerate:

```bash
cargo run -p fountain_operators --example capture_traces --release
# optional RaptorQ trace:
cargo run -p fountain_operators --example capture_traces --release --features raptor_q
```

Files are named `{scheme}_k{K}_t{T}.json` and consumed by `operator_replay` benches and `tests/trace_replay.rs`.
