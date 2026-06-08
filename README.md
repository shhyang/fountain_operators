# fountain_operators

High-performance [`DataOperator`](https://docs.rs/fountain_engine/1.3.0/fountain_engine/traits/trait.DataOperator.html) implementations for [fountain_engine](https://crates.io/crates/fountain_engine) **1.3+**.

Provides slab-backed and SIMD-accelerated operators, portable GF(256) block kernels, operation replay, and testing helpers for fountain code encode/decode pipelines.

Project docs: [fountain_docs](https://shhyang.github.io/fountain_docs/) · Operator guide: [doc-operators.pdf](https://shhyang.github.io/fountain_docs/docs/doc-operators.pdf)

## Components

| Module | Role |
|--------|------|
| `gf_kernels` | Portable, AVX2, and NEON GF(256) XOR / multiply kernels |
| `SlabDataOperator` | Slab-allocated in-memory operator |
| `SimdDataOperator` | Slab operator with runtime kernel dispatch (`simd` feature) |
| `replay` | Replay delayed encoder operation logs |
| `operator_testing` | Differential testing against reference operators |
| `trace_format` | JSON trace capture for benchmarks |

## Usage

Add to `Cargo.toml`:

```toml
fountain_engine = "1.3"
fountain_operators = { version = "1.0", features = ["simd"] }
```

```rust
use fountain_engine::traits::DataOperator;
use fountain_operators::{replay_operations, default_kernel, GfBlockKernel, SlabDataOperator};

// Replay a delayed encoder operation log
replay_operations(&mut my_operator, &ops);

// Portable GF(256) kernels (used by slab/SIMD operators)
let kernel = default_kernel();
let gf = fountain_engine::algebra::finite_field::GF256::new_with_primitive_polynomial(0x11D);
let mut dst = vec![1u8, 2, 3];
kernel.xor_inplace(&mut dst, &[4, 5, 6]);
```

## Testing

```bash
cargo test
cargo test --features simd
```

Differential tests compare `SlabDataOperator` / `SimdDataOperator` against `VecDataOperater` on the same random operation stream (GF2 and GF256).

## Benchmarks

```bash
# Regenerate committed traces (requires dev-deps: fountain_scheme, fountain_utility)
cargo run --example capture_traces --release

# Criterion (Layer 1–2)
cargo bench --features bench

# JSONL report (Layer 2: replay_wall_ms + e2e_wall_ms)
cargo run --example operators_benchmark --release

# Full local workflow
./scripts/compare_operators.sh
```

Traces live in `bench_data/traces/`. Layer 1 replays **encoder** ops (precoding + encoding) from the delayed `Encoder::new` log.

## Features

- `tooling` (default) — replay, trace format, operator/scheme testing helpers
- `simd` — `select_kernel()`: **AVX2** on x86_64 (Haswell+), **NEON** on aarch64/arm; portable fallback otherwise
- `bench` — Criterion targets `operator_replay`, `e2e_encode_decode`, `kernel_micro`
- `compare-refs` — optional throughput vs external raptorq in `kernel_micro` (local checkout only; not required for normal use)

### SIMD platform support (`--features simd`)

| Target | Runtime feature | Kernel |
|--------|-----------------|--------|
| `x86_64` / `x86` | AVX2 | 32-byte AVX2 XOR / nibble-mul |
| `aarch64` / `arm` | NEON | 16-byte NEON XOR / `vqtbl1q` mul |
| Other / feature off | — | Portable scalar |

LUTs are built from session [`GF256`](https://docs.rs/fountain_engine/1.3.0/fountain_engine/algebra/finite_field/struct.GF256.html) (`mul_lookup`). `default_kernel()` is always portable; `SimdDataOperator` uses `select_kernel()`.

## License

MIT — see [LICENSE-MIT](LICENSE-MIT).
