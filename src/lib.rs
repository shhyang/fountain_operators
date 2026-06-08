// Copyright (c) 2025 Shenghao Yang. All rights reserved.
// Licensed under the MIT License. See LICENSE-MIT for details.

//! Compute-first GF(256) acceleration and optional [`DataOperator`](fountain_engine::traits::DataOperator) implementations.
//!
//! Phase 1: portable [`gf_kernels`](gf_kernels), [`replay_operations`](replay::replay_operations),
//! and differential testing helpers ([`operator_testing`]).
//! Phase 2: [`SlabDataOperator`](slab_data_operator::SlabDataOperator) on [`SlabStorage`](slab_storage::SlabStorage).
//! Phase 3a: [`SimdDataOperator`](simd_data_operator::SimdDataOperator) with [`select_kernel()`](gf_kernels::select_kernel).
//! Phase 4: [`CapturedTrace`](trace_format::CapturedTrace), Criterion benches (`bench` feature), `operators_benchmark` JSONL.
//!
//! See `docs/plans/data_operators_dev.md` and `docs/doc-operators.org`.

pub mod gf_kernels;
pub mod slab_data_operator;
pub mod slab_storage;
pub mod simd_data_operator;
#[cfg(feature = "tooling")]
pub mod bench_report;
#[cfg(feature = "tooling")]
pub mod operator_testing;
#[cfg(feature = "tooling")]
pub mod replay;
#[cfg(feature = "tooling")]
pub mod scheme_operator_testing;
#[cfg(feature = "tooling")]
pub mod trace_format;

pub use gf_kernels::{
    available_kernel_kinds, default_kernel, select_kernel, selected_kernel_kind, GfBlockKernel,
    KernelKind, PortableKernel,
};
#[cfg(all(
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub use gf_kernels::Avx2Kernel;
#[cfg(all(
    feature = "simd",
    any(target_arch = "aarch64", target_arch = "arm")
))]
pub use gf_kernels::NeonKernel;
pub use simd_data_operator::SimdDataOperator;
pub use slab_data_operator::SlabDataOperator;
pub use slab_storage::SlabStorage;
#[cfg(feature = "tooling")]
pub use bench_report::{median_ms, throughput_mib_s, BenchRecord};
#[cfg(feature = "tooling")]
pub use operator_testing::{
    assert_operators_equivalent, differential_test_random_ops, generate_random_ops,
    OperatorSnapshot,
};
#[cfg(feature = "tooling")]
pub use replay::replay_operations;
#[cfg(feature = "tooling")]
pub use scheme_operator_testing::{
    assert_roundtrip_matches_reference, fountain_coded_ids, lt_coded_ids,
    make_test_messages, roundtrip_delayed_replay, roundtrip_on_the_fly, OperatorFactory,
    RoundtripResult,
};
#[cfg(feature = "tooling")]
pub use trace_format::{CapturedTrace, OperationJson, TraceError};
