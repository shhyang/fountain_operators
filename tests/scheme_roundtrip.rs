// Copyright (c) 2025 Shenghao Yang. All rights reserved.
// Licensed under the MIT License. See LICENSE-MIT for details.

//! Phase 2b: on-the-fly scheme roundtrips (plan §7.2.3).

mod common;

use common::{make_simd_operator, make_slab_operator, make_vec_operator};
use fountain_engine::traits::CodeScheme;
use fountain_operators::scheme_operator_testing::{
    assert_roundtrip_matches_reference, fountain_coded_ids, lt_coded_ids, make_test_messages,
    roundtrip_on_the_fly,
};
use fountain_scheme::validation::pseudo_rand::XorShift64;
use fountain_scheme::{BinaryHDPCLTCode, HDPCLTCode, LDPCLTCode, RandomLTCode};

fn assert_slab_matches_vec<C: CodeScheme + Clone>(
    scheme: C,
    k: usize,
    symbol_size: usize,
    coded_ids: Vec<usize>,
) {
    let messages = make_test_messages(k, symbol_size);
    assert_roundtrip_matches_reference(
        scheme,
        &messages,
        &coded_ids,
        make_slab_operator,
        make_vec_operator,
    );
}

#[test]
fn random_lt_vec() {
    let k = 15;
    let t = 3;
    let scheme = RandomLTCode::new_from_robust_soliton(k, XorShift64::new(0x00C0_FFEE));
    let messages = make_test_messages(k, t);
    let result = roundtrip_on_the_fly(scheme, &messages, &lt_coded_ids(k), make_vec_operator);
    assert!(result.success(), "{result:?}");
}

fn assert_simd_matches_vec<C: CodeScheme + Clone>(
    scheme: C,
    k: usize,
    symbol_size: usize,
    coded_ids: Vec<usize>,
) {
    let messages = make_test_messages(k, symbol_size);
    assert_roundtrip_matches_reference(
        scheme,
        &messages,
        &coded_ids,
        make_simd_operator,
        make_vec_operator,
    );
}

#[test]
fn random_lt_simd_vs_vec() {
    let k = 15;
    let t = 3;
    let scheme = RandomLTCode::new_from_robust_soliton(k, XorShift64::new(0x00C0_FFEE));
    assert_simd_matches_vec(scheme, k, t, lt_coded_ids(k));
}

#[test]
fn random_lt_slab_vs_vec() {
    let k = 15;
    let t = 3;
    let scheme = RandomLTCode::new_from_robust_soliton(k, XorShift64::new(0x00C0_FFEE));
    assert_slab_matches_vec(scheme, k, t, lt_coded_ids(k));
}

#[test]
fn hdpc_lt_vec_and_slab() {
    for k in 12..=15 {
        let scheme = HDPCLTCode::new_with_ideal_soliton(k, XorShift64::new(0x00C0_FFEE));
        let params = scheme.get_params();
        let coded_ids = fountain_coded_ids(&params, k, 0);
        assert_slab_matches_vec(scheme, k, 1, coded_ids);
    }
}

#[test]
fn binary_hdpc_lt_vec_and_slab() {
    for k in 12..=15 {
        let scheme = BinaryHDPCLTCode::new_with_ideal_soliton(k, XorShift64::new(0x00C0_FFEE));
        let params = scheme.get_params();
        let coded_ids = fountain_coded_ids(&params, k, 0);
        assert_slab_matches_vec(scheme, k, 1, coded_ids);
    }
}

#[test]
fn ldpc_lt_vec_and_slab() {
    let k = 23;
    let t = 3;
    let scheme = LDPCLTCode::new_with_ideal_soliton(k, XorShift64::new(0x00C0_FFEE));
    let params = scheme.get_params();
    let coded_ids: Vec<usize> = (params.num_total()..params.num_total() + k * 3).collect();
    assert_slab_matches_vec(scheme, k, t, coded_ids);
}
