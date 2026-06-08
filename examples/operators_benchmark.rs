//! Layer 2 JSONL benchmark report (plan §8.6).
//!
//! ```bash
//! cargo run -p fountain_operators --example operators_benchmark --release
//! cargo run -p fountain_operators --example operators_benchmark --release --features simd
//! ```

use std::io::{self, Write};
use std::time::Instant;

use fountain_operators::{
    bench_report::{median_ms, throughput_mib_s, BenchRecord},
    lt_coded_ids, make_test_messages, replay_operations, roundtrip_on_the_fly, CapturedTrace,
    SlabDataOperator, SimdDataOperator,
};
use fountain_scheme::validation::pseudo_rand::XorShift64;
use fountain_scheme::RandomLTCode;
use fountain_utility::VecDataOperater;

const WARMUP: usize = 2;
const SAMPLES: usize = 12;

fn vec_factory(symbol_size: usize) -> Box<dyn fountain_engine::traits::DataOperator> {
    Box::new(VecDataOperater::new(symbol_size))
}

fn slab_factory(symbol_size: usize) -> Box<dyn fountain_engine::traits::DataOperator> {
    Box::new(SlabDataOperator::new(symbol_size))
}

fn simd_factory(symbol_size: usize) -> Box<dyn fountain_engine::traits::DataOperator> {
    Box::new(SimdDataOperator::new(symbol_size))
}

fn replay_ms(
    factory: fn(usize) -> Box<dyn fountain_engine::traits::DataOperator>,
    trace: &CapturedTrace,
) -> f64 {
    let symbol_size = trace.symbol_size;
    let mut op = factory(symbol_size);
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
    let start = Instant::now();
    replay_operations(op.as_mut(), &trace.precoding_operations());
    replay_operations(op.as_mut(), &trace.encoding_operations());
    start.elapsed().as_secs_f64() * 1000.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let k = 15;
    let t = 64;
    let scheme = RandomLTCode::new_from_robust_soliton(k, XorShift64::new(0x00C0_FFEE));
    let messages = make_test_messages(k, t);
    let coded_ids = lt_coded_ids(k);

    let trace_path = format!(
        "{}/bench_data/traces/random_lt_k{k}_t{t}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let trace = CapturedTrace::load_json(&trace_path)?;

    let mut out = io::stdout().lock();

    // E2E on-the-fly (Vec)
    let mut e2e_samples = Vec::with_capacity(SAMPLES + WARMUP);
    for _ in 0..(SAMPLES + WARMUP) {
        let start = Instant::now();
        let _ = roundtrip_on_the_fly(scheme.clone(), &messages, &coded_ids, vec_factory);
        e2e_samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let e2e_ms = median_ms(&e2e_samples[WARMUP..]);
    let payload_bytes = k * t;
    let record = BenchRecord {
        operator: "VecDataOperater".into(),
        scheme: "RandomLTCode".into(),
        k,
        symbol_size: t,
        phase: "all".into(),
        mode: "on_the_fly".into(),
        replay_wall_ms: 0.0,
        e2e_wall_ms: Some(e2e_ms),
        throughput_mib_s: Some(throughput_mib_s(payload_bytes, e2e_ms)),
        trace: None,
    };
    writeln!(out, "{}", record.to_jsonl_line()?)?;

    for (name, factory) in [
        ("VecDataOperater", vec_factory as fn(usize) -> _),
        ("SlabDataOperator", slab_factory),
        ("SimdDataOperator", simd_factory),
    ] {
        let mut samples = Vec::with_capacity(SAMPLES + WARMUP);
        for _ in 0..(SAMPLES + WARMUP) {
            samples.push(replay_ms(factory, &trace));
        }
        let replay_ms = median_ms(&samples[WARMUP..]);
        let n_ops = trace.precoding.len() + trace.encoding.len();
        let op_bytes: usize = n_ops * t;
        let record = BenchRecord {
            operator: name.into(),
            scheme: trace.scheme.clone(),
            k: trace.k,
            symbol_size: trace.symbol_size,
            phase: "replay".into(),
            mode: "delayed_replay".into(),
            replay_wall_ms: replay_ms,
            e2e_wall_ms: Some(e2e_ms),
            throughput_mib_s: Some(throughput_mib_s(op_bytes.max(payload_bytes), replay_ms)),
            trace: Some(trace.name.clone()),
        };
        writeln!(out, "{}", record.to_jsonl_line()?)?;
    }

    Ok(())
}
