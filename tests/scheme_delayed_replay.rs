// Copyright (c) 2025 Shenghao Yang. All rights reserved.
// Licensed under the MIT License. See LICENSE-MIT for details.

//! Phase 2b: delayed encoder replay + decode (plan §7.3).

mod common;

use common::{make_simd_operator, make_slab_operator, make_vec_operator};
use fountain_engine::traits::CodeScheme;
use fountain_engine::types::GF2_FIELD_POLY;
use fountain_operators::scheme_operator_testing::{
    fountain_coded_ids, lt_coded_ids, make_test_messages, roundtrip_delayed_replay,
};
use fountain_scheme::validation::pseudo_rand::XorShift64;
use fountain_scheme::{BinaryHDPCLTCode, HDPCLTCode, RandomLTCode};

#[test]
fn random_lt_delayed_slab() {
    let k = 15;
    let t = 3;
    let scheme = RandomLTCode::new_from_robust_soliton(k, XorShift64::new(0x00C0_FFEE));
    let messages = make_test_messages(k, t);
    let result = roundtrip_delayed_replay(
        scheme,
        &messages,
        &lt_coded_ids(k),
        GF2_FIELD_POLY,
        make_slab_operator,
        make_vec_operator,
    );
    assert!(result.success(), "{result:?}");
}

#[test]
fn random_lt_delayed_simd() {
    let k = 15;
    let t = 3;
    let scheme = RandomLTCode::new_from_robust_soliton(k, XorShift64::new(0x00C0_FFEE));
    let messages = make_test_messages(k, t);
    let result = roundtrip_delayed_replay(
        scheme,
        &messages,
        &lt_coded_ids(k),
        GF2_FIELD_POLY,
        make_simd_operator,
        make_vec_operator,
    );
    assert!(result.success(), "{result:?}");
}

#[test]
fn hdpc_lt_delayed_slab() {
    let k = 14;
    let scheme = HDPCLTCode::new_with_ideal_soliton(k, XorShift64::new(0x00C0_FFEE));
    let params = scheme.get_params();
    let messages = make_test_messages(k, 1);
    let result = roundtrip_delayed_replay(
        scheme,
        &messages,
        &fountain_coded_ids(&params, k, 0),
        0x11D,
        make_slab_operator,
        make_vec_operator,
    );
    assert!(result.success(), "{result:?}");
}

#[test]
fn binary_hdpc_lt_delayed_slab() {
    let k = 14;
    let scheme = BinaryHDPCLTCode::new_with_ideal_soliton(k, XorShift64::new(0x00C0_FFEE));
    let params = scheme.get_params();
    let messages = make_test_messages(k, 1);
    let result = roundtrip_delayed_replay(
        scheme,
        &messages,
        &fountain_coded_ids(&params, k, 0),
        GF2_FIELD_POLY,
        make_slab_operator,
        make_vec_operator,
    );
    assert!(result.success(), "{result:?}");
}
