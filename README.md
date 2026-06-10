# fountain_operators

High-performance [`DataOperator`](https://docs.rs/fountain_engine/latest/fountain_engine/traits/trait.DataOperator.html) implementations for the Fountain Engine.

## Status

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | `gf_kernels` (portable), `replay`, `operator_testing` | **Done** |
| 2 | `SlabStorage`, `SlabDataOperator` | **Done** |
| 2b | Scheme roundtrip + delayed replay tests | **Done** |
| 3a | `SimdDataOperator`, AVX2 `gf_kernels` (`simd` feature) | **Done** |
| 3b | NEON `gf_kernels` on aarch64/arm (`simd` feature) | **Done** |
| 4 | Traces, Criterion benches, JSONL report | **Done** |
| 5 | Agent skill, doc polish | **Done** |

See [docs/plans/data_operators_dev.md](../docs/plans/data_operators_dev.md) (milestones) and [docs/doc-operators.org](../docs/doc-operators.org) (user guide).

## Usage

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
cargo test -p fountain_operators
cargo test -p fountain_operators --features simd
```

Differential tests compare `SlabDataOperator` / `SimdDataOperator` against `VecDataOperater` on the same random operation stream (GF2 and GF256).

## Benchmarks (phase 4)

```bash
# Regenerate committed traces (optional: --features raptor_q)
cargo run -p fountain_operators --example capture_traces --release

# Criterion (Layer 1–2)
cargo bench -p fountain_operators --features bench

# JSONL report (Layer 2: replay_wall_ms + e2e_wall_ms)
cargo run -p fountain_operators --example operators_benchmark --release

# Full local workflow
./fountain_operators/scripts/compare_operators.sh
```

Traces live in `bench_data/traces/`. Layer 1 replays **encoder** ops (precoding + encoding) from the delayed `Encoder::new` log.

## Features

- `simd` — `select_kernel()`: **AVX2** on x86_64 (Haswell+), **NEON** on aarch64/arm; portable fallback otherwise
- `bench` — Criterion targets `operator_replay`, `e2e_encode_decode`, `kernel_micro`
- `compare-refs` — optional throughput vs `ref/raptorq` in `kernel_micro` (requires `ref/raptorq` checkout)
- `raptor_q` — `capture_traces` example also writes a RaptorQ trace

### SIMD platform support (`--features simd`)

| Target | Runtime feature | Kernel |
|--------|-----------------|--------|
| `x86_64` / `x86` | AVX2 | 32-byte AVX2 XOR / nibble-mul |
| `aarch64` / `arm` | NEON | 16-byte NEON XOR / `vqtbl1q` mul |
| Other / feature off | — | Portable scalar |

LUTs are built from session [`GF256`](fountain_engine) (`mul_lookup`), not raptorq `Octet` tables.
`default_kernel()` is always portable; `SimdDataOperator` uses `select_kernel()`.

## End-to-end examples (Raptor crates)

The published Raptor scheme crates ship **performance examples** that exercise `SlabDataOperator` and `SimdDataOperator` alongside `fountain_utility::VecDataOperater`. They use [`real_symbol_benchmark`](https://docs.rs/fountain_utility/latest/fountain_utility/) to compare on-the-fly vs deferred encode/decode wall times at realistic symbol sizes.

Add this crate as a **dev-dependency** (Git; not on Crates.io):

```toml
[dev-dependencies]
fountain_operators = { git = "https://github.com/shhyang/fountain_operators", features = ["simd"] }
```

Then run from a checkout of the Raptor crate:

| Example | Crate | What it benchmarks |
|---------|-------|-------------------|
| [`raptor_q_performance`](https://github.com/wutongabc/fountain_raptor_q/blob/main/examples/raptor_q_performance.rs) | [`fountain_raptor_q`](https://github.com/wutongabc/fountain_raptor_q) | RFC 6330 RaptorQ; padded codec @ K = 5008; symbol sizes 128 and 1500 |
| [`raptor_10_performance`](https://github.com/wutongabc/fountain_raptor_10/blob/main/examples/raptor_10_performance.rs) | [`fountain_raptor_10`](https://github.com/wutongabc/fountain_raptor_10) | RFC 5053 Raptor-10; systematic codec @ k = 5000; symbol sizes 128 and 1500 |

```bash
# RaptorQ (clone fountain_raptor_q, then from its root)
cargo run --example raptor_q_performance --release

# Raptor-10 (clone fountain_raptor_10, then from its root)
cargo run --example raptor_10_performance --release
```

Each example builds operator factories for `VecDataOperater`, `SlabDataOperator`, and `SimdDataOperator`, then prints a comparison table via `print_real_symbol_benchmark_table`. See the example sources for `OperatorFactory` wiring and `RealSymbolBenchConfig` setup.

**Note:** These crates depend on `fountain_engine` **1.3.1+**, `fountain_utility` **1.3.0+**, and the matching scheme crate on Crates.io. The operators crate remains **experimental**; API may change.

## License

MIT
