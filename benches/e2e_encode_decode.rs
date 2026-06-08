//! Layer 2: on-the-fly vs replay-only wall time (plan §8.1–8.2).

mod support;

use std::time::Instant;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fountain_engine::traits::CodeScheme;
use fountain_operators::{
    fountain_coded_ids, lt_coded_ids, make_test_messages, roundtrip_on_the_fly, RoundtripResult,
};
use fountain_scheme::validation::pseudo_rand::XorShift64;
use fountain_scheme::{HDPCLTCode, RandomLTCode};
use support::{make_vec, replay_encoder_trace};

fn bench_e2e(c: &mut Criterion) {
    let k = 15;
    let t = 64;
    let lt_scheme = RandomLTCode::new_from_robust_soliton(k, XorShift64::new(0x00C0_FFEE));
    let messages = make_test_messages(k, t);
    let coded_ids = lt_coded_ids(k);

    let mut group = c.benchmark_group("e2e_lt");
    group.sample_size(20);

    group.bench_function("on_the_fly_vec", |b| {
        b.iter(|| {
            black_box(roundtrip_on_the_fly(
                lt_scheme.clone(),
                &messages,
                &coded_ids,
                make_vec,
            ));
        });
    });

    let reference: RoundtripResult = roundtrip_on_the_fly(
        lt_scheme.clone(),
        &messages,
        &coded_ids,
        make_vec,
    );
    let trace = fountain_operators::CapturedTrace::from_roundtrip(
        "random_lt",
        "RandomLTCode",
        k,
        t,
        Some(fountain_engine::types::GF2_FIELD_POLY),
        &reference.precoding_ops,
        &reference.encoding_ops,
        &reference.decoding_ops,
    );

    group.bench_function("replay_only_vec", |b| {
        b.iter(|| {
            let mut op = make_vec(t);
            replay_encoder_trace(op.as_mut(), black_box(&trace));
        });
    });

    group.bench_with_input(BenchmarkId::new("replay_only", "Slab"), &trace, |b, trace| {
        b.iter(|| {
            let mut op = support::make_slab(t);
            replay_encoder_trace(op.as_mut(), black_box(trace));
        });
    });

    group.finish();

    // GF(256) HDPC — smaller symbol size
    let k = 15;
    let t = 1;
    let hdpc = HDPCLTCode::new_with_ideal_soliton(k, XorShift64::new(0x00C0_FFEE));
    let params = hdpc.get_params();
    let coded_ids = fountain_coded_ids(&params, k, 0);
    let messages = make_test_messages(k, t);

    let mut group = c.benchmark_group("e2e_hdpc_gf256");
    group.sample_size(20);
    group.bench_function("on_the_fly_vec", |b| {
        b.iter(|| {
            black_box(roundtrip_on_the_fly(
                hdpc.clone(),
                &messages,
                &coded_ids,
                make_vec,
            ));
        });
    });
    group.finish();
}

criterion_group!(benches, bench_e2e);
criterion_main!(benches);
