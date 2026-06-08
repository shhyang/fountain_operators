//! Micro-benchmarks for GF kernels (Layer 1 primitives). Optional `compare-refs` throughput.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fountain_engine::algebra::finite_field::GF256;
use fountain_operators::{default_kernel, select_kernel, GfBlockKernel};

fn bench_kernel_xor(c: &mut Criterion) {
    let len = 16_384usize;
    let src: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
    let mut dst = vec![0u8; len];

    let mut group = c.benchmark_group("kernel_xor");
    for (name, kernel) in [
        ("portable", default_kernel() as &dyn GfBlockKernel),
        ("selected", select_kernel()),
    ] {
        group.bench_with_input(BenchmarkId::new(name, len), &src, |b, src| {
            b.iter(|| {
                dst.copy_from_slice(&vec![0u8; len]);
                kernel.xor_inplace(black_box(&mut dst), black_box(src));
            });
        });
    }
    group.finish();
}

fn bench_kernel_mul_add(c: &mut Criterion) {
    let len = 16_384usize;
    let gf = GF256::new_with_primitive_polynomial(0x11D);
    let src: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
    let mut dst = vec![1u8; len];
    let scalar = 0xA7u8;

    let mut group = c.benchmark_group("kernel_mul_add");
    for (name, kernel) in [
        ("portable", default_kernel() as &dyn GfBlockKernel),
        ("selected", select_kernel()),
    ] {
        group.bench_with_input(BenchmarkId::new(name, len), &src, |b, src| {
            b.iter(|| {
                dst.copy_from_slice(&vec![1u8; len]);
                kernel.mul_add_inplace(black_box(&gf), black_box(&mut dst), black_box(src), scalar);
            });
        });
    }
    group.finish();
}

#[cfg(feature = "compare-refs")]
fn bench_compare_raptorq_xor(c: &mut Criterion) {
    // Throughput-only comparison; requires `ref/raptorq` checkout (see ref/README.md).
    use raptorq_ref::octets::Octets;

    let len = 16_384usize;
    let a: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
    let b: Vec<u8> = (1..=len).map(|i| (i % 253) as u8).collect();

    let mut group = c.benchmark_group("compare_refs_xor");
    group.bench_function("fountain_operators_portable", |bench| {
        let kernel = default_kernel();
        let mut dst = vec![0u8; len];
        bench.iter(|| {
            dst.copy_from_slice(&a);
            kernel.xor_inplace(black_box(&mut dst), black_box(&b));
        });
    });
    group.bench_function("raptorq_octets_add_assign", |bench| {
        bench.iter(|| {
            let mut oct_a = Octets::new(a.clone());
            let oct_b = Octets::new(b.clone());
            oct_a.add_assign(black_box(&oct_b));
            black_box(oct_a);
        });
    });
    group.finish();
}

#[cfg(not(feature = "compare-refs"))]
fn bench_compare_raptorq_xor(_c: &mut Criterion) {}

criterion_group!(
    kernels,
    bench_kernel_xor,
    bench_kernel_mul_add,
    bench_compare_raptorq_xor
);
criterion_main!(kernels);
