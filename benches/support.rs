// Copyright (c) 2025 Shenghao Yang. See LICENSE-MIT for details.

use std::path::PathBuf;
use std::time::Instant;

use fountain_engine::traits::DataOperator;
use fountain_operators::{replay_operations, CapturedTrace, SlabDataOperator, SimdDataOperator};
use fountain_utility::VecDataOperater;

pub fn bench_data_traces_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench_data/traces")
}

pub fn load_trace(name: &str) -> CapturedTrace {
    let path = bench_data_traces_dir().join(format!("{name}.json"));
    CapturedTrace::load_json(&path).unwrap_or_else(|e| panic!("load {}: {e}", path.display()))
}

pub fn make_vec(symbol_size: usize) -> Box<dyn DataOperator> {
    Box::new(VecDataOperater::new(symbol_size))
}

pub fn make_slab(symbol_size: usize) -> Box<dyn DataOperator> {
    Box::new(SlabDataOperator::new(symbol_size))
}

pub fn make_simd(symbol_size: usize) -> Box<dyn DataOperator> {
    Box::new(SimdDataOperator::new(symbol_size))
}

/// Replay encoder-side ops only (precoding + encoding). Decoding ops need decoder state.
pub fn replay_encoder_trace(op: &mut dyn DataOperator, trace: &CapturedTrace) {
    if let Some(pp) = trace.field_pp {
        op.config_finite_field(pp);
    }
    let k = trace.k;
    let t = trace.symbol_size;
    for i in 0..k {
        let mut row = vec![0u8; t];
        for (j, b) in row.iter_mut().enumerate() {
            *b = ((i * 7 + j * 13) % 256) as u8;
        }
        op.insert_vector(&row, i);
    }
    let precoding = trace.precoding_operations();
    let encoding = trace.encoding_operations();
    if precoding.is_empty() {
        // Delayed encoder log: precoding is folded into `encoding` (plan §7.3).
        replay_operations(op, &encoding);
    } else {
        replay_operations(op, &precoding);
        replay_operations(op, &encoding);
    }
}

pub fn time_replay_ms(
    factory: fn(usize) -> Box<dyn DataOperator>,
    trace: &CapturedTrace,
) -> f64 {
    let symbol_size = trace.symbol_size;
    let mut op = factory(symbol_size);
    let start = Instant::now();
    replay_encoder_trace(op.as_mut(), trace);
    start.elapsed().as_secs_f64() * 1000.0
}
