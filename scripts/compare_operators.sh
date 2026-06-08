#!/usr/bin/env bash
# Phase 4 operator comparison (plan §8.4). Run from repository root.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "== Regenerate traces (if needed) =="
cargo run -p fountain_operators --example capture_traces --release

echo "== Unit tests =="
cargo test -p fountain_operators
cargo test -p fountain_operators --features simd

echo "== Layer 1: operator replay (Criterion) =="
cargo bench -p fountain_operators --features bench --release --bench operator_replay

echo "== Layer 2: e2e encode/decode =="
cargo bench -p fountain_operators --features bench --release --bench e2e_encode_decode

echo "== Kernel micro =="
cargo bench -p fountain_operators --features bench --release --bench kernel_micro

echo "== JSONL report =="
cargo run -p fountain_operators --example operators_benchmark --release | tee test_results/operators_benchmark.jsonl

if [[ -d ref/raptorq ]]; then
  echo "== compare-refs (optional) =="
  cargo bench -p fountain_operators --features bench,compare-refs --release --bench kernel_micro
else
  echo "skip compare-refs: clone ref/raptorq per ref/README.md"
fi
