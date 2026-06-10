# Changelog

All notable changes to the [`fountain_operators`](https://github.com/shhyang/fountain_operators) library are documented here.

## [1.0.0] - 2026-06-10

### Added

Initial release for [`fountain_engine`](https://github.com/shhyang/fountain_engine) **1.3+** (distributed via [GitHub](https://github.com/shhyang/fountain_operators), not crates.io).

- **`gf_kernels`** — portable GF(256) block XOR / multiply; optional **AVX2** (x86_64) and **NEON** (aarch64) via `simd` feature (`select_kernel()`, `default_kernel()`).
- **`SlabDataOperator`** / **`SlabStorage`** — slab-allocated in-memory `DataOperator` matching `VecDataOperater` semantics.
- **`SimdDataOperator`** — slab operator with runtime kernel dispatch (`simd` feature).
- **`tooling` feature** (default):
  - `replay` — replay delayed-encoder operation logs.
  - `operator_testing` — differential testing against a reference operator.
  - `scheme_operator_testing` — on-the-fly and delayed scheme roundtrip helpers.
  - `trace_format` — JSON trace capture for benchmarks.
  - `bench_report` — median / throughput helpers for JSONL reports.

### Dependencies

- Requires `fountain_engine` **1.3.1** (published legacy solver API).

### Packaging

- Library `src/` and integration **tests** ship from the GitHub repo (`autobenches` / `autoexamples` disabled so repo-local Criterion targets are not part of the default library build).
- Criterion **benches**, **examples** (`capture_traces`, `operators_benchmark`), and `bench_data/` remain in the [GitHub repo](https://github.com/shhyang/fountain_operators) for local development (use monorepo or standalone repo with `bench` feature when added).

### Notes

- Used as an optional **dev-dependency** by `fountain_raptor_q` / `fountain_raptor_10` performance examples (`SlabDataOperator`, `SimdDataOperator`).
