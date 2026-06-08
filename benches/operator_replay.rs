//! Layer 1: replay captured scheme traces on Vec / Slab / Simd (plan §8.1).

mod support;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use support::{load_trace, make_slab, make_simd, make_vec, replay_encoder_trace};

fn bench_operator_replay(c: &mut Criterion) {
    let traces = [
        ("random_lt", load_trace("random_lt_k15_t64")),
        ("hdpc_gf256", load_trace("hdpc_gf256_k14_t1")),
        ("binary_hdpc", load_trace("binary_hdpc_k14_t1")),
    ];

    let mut group = c.benchmark_group("operator_replay");
    group.sample_size(30);

    for (label, trace) in traces {
        let symbol_size = trace.symbol_size;
        for (op_name, factory) in [
            ("Vec", make_vec as fn(usize) -> _),
            ("Slab", make_slab),
            ("Simd", make_simd),
        ] {
            group.bench_with_input(
                BenchmarkId::new(op_name, label),
                &(factory, &trace),
                |b, (factory, trace)| {
                    b.iter(|| {
                        let mut op = factory(symbol_size);
                        replay_encoder_trace(op.as_mut(), black_box(trace));
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_operator_replay);
criterion_main!(benches);
