// Copyright (c) 2025 Shenghao Yang. All rights reserved.
// Licensed under the MIT License. See LICENSE-MIT for details.

//! Scheme-driven encode/decode tests for [`DataOperator`] implementations (plan §7.2).

use std::collections::HashMap;

use fountain_engine::traits::{CodeScheme, DataOperator};
use fountain_engine::types::{CodeParams, DecodeStatus, Operation};
use fountain_engine::{Decoder, Encoder};

use crate::replay::replay_operations;

/// Builds a fresh in-memory operator for the given symbol length.
pub type OperatorFactory = fn(usize) -> Box<dyn DataOperator>;

/// Outcome of an on-the-fly or delayed-replay scheme roundtrip.
#[derive(Debug, Clone)]
pub struct RoundtripResult {
    pub num_mismatches: usize,
    pub decoded: bool,
    pub precoding_ops: Vec<Operation>,
    pub encoding_ops: Vec<Operation>,
    pub decoding_ops: Vec<Operation>,
}

impl RoundtripResult {
    #[must_use]
    pub fn success(&self) -> bool {
        self.decoded && self.num_mismatches == 0
    }

    /// Concatenated operation log (precoding → encoding → decoding).
    #[must_use]
    pub fn all_operations(&self) -> Vec<Operation> {
        let mut ops = self.precoding_ops.clone();
        ops.extend(self.encoding_ops.clone());
        ops.extend(self.decoding_ops.clone());
        ops
    }
}

/// Deterministic message payloads for tests (`code_testing`-style).
#[must_use]
pub fn make_test_messages(k: usize, symbol_size: usize) -> Vec<Vec<u8>> {
    let mut message_vectors = vec![vec![0u8; symbol_size]; k];
    for (i, row) in message_vectors.iter_mut().enumerate() {
        for (j, byte) in row.iter_mut().enumerate() {
            *byte = ((i * 7 + j * 13) % 256) as u8;
        }
    }
    message_vectors
}

/// LT-only coded IDs (`k..3k`), matching [`fountain_scheme::tests::lt_integration_test`].
#[must_use]
pub fn lt_coded_ids(k: usize) -> Vec<usize> {
    (k..k * 3).collect()
}

/// Fountain-symbol coded IDs (`num_total + offset ..`), matching HDPC integration tests.
#[must_use]
pub fn fountain_coded_ids(params: &CodeParams, k: usize, offset: usize) -> Vec<usize> {
    let total = params.num_total();
    (total + offset..total + 3 * k + offset).collect()
}

/// On-the-fly encode/decode (same flow as `fountain_scheme` integration tests).
pub fn roundtrip_on_the_fly<C: CodeScheme + Clone>(
    scheme: C,
    message_vectors: &[Vec<u8>],
    coded_ids: &[usize],
    make_operator: OperatorFactory,
) -> RoundtripResult {
    let k = message_vectors.len();
    assert!(k > 0);
    let symbol_size = message_vectors[0].len();

    let mut enc_operator = make_operator(symbol_size);
    for (i, vector) in message_vectors.iter().enumerate() {
        enc_operator.insert_vector(vector, i);
    }

    let mut encoder = Encoder::new_with_operator(&scheme.clone(), enc_operator);
    let precoding_ops = encoder.manager.move_new_operations();

    for &coded_id in coded_ids {
        encoder.encode_coded_vector(coded_id);
    }
    let encoding_ops = encoder.manager.move_new_operations();

    let mut decoder = Decoder::new_with_operator(&scheme, make_operator(symbol_size));
    let mut decoded = false;
    for &coded_id in coded_ids {
        let payload = encoder.manager.get_coded_vector(coded_id);
        if matches!(
            decoder.add_coded_vector(coded_id, &payload),
            DecodeStatus::Decoded
        ) {
            decoded = true;
            break;
        }
    }
    let decoding_ops = decoder.manager.move_new_operations();

    let dec_operator = decoder.manager.move_operator();
    let num_mismatches = count_message_mismatches(&*dec_operator, message_vectors, decoded);

    RoundtripResult {
        num_mismatches,
        decoded,
        precoding_ops,
        encoding_ops,
        decoding_ops,
    }
}

/// Delayed encoder log + replay on `make_operator`, then decode using replayed coded payloads.
///
/// Coded-vector bytes for the channel are taken from a reference on-the-fly run with
/// `reference_factory` (typically [`VecDataOperater`](fountain_utility::VecDataOperater)).
///
/// `field_pp` must match the scheme HDPC field ([`GF2_FIELD_POLY`](fountain_engine::types::GF2_FIELD_POLY)
/// or e.g. `0x11D` for RQ-style HDPC). It is applied before replay because `config_finite_field`
/// is not recorded in the op log.
pub fn roundtrip_delayed_replay<C: CodeScheme + Clone>(
    scheme: C,
    message_vectors: &[Vec<u8>],
    coded_ids: &[usize],
    field_pp: u16,
    make_operator: OperatorFactory,
    reference_factory: OperatorFactory,
) -> RoundtripResult {
    let reference = roundtrip_on_the_fly(
        scheme.clone(),
        message_vectors,
        coded_ids,
        reference_factory,
    );

    let coded_payloads =
        capture_coded_vectors_on_the_fly(scheme.clone(), message_vectors, coded_ids, reference_factory);

    let mut delayed_encoder = Encoder::new(&scheme.clone());
    let precoding_ops = delayed_encoder.manager.move_new_operations();

    let mut mappings = HashMap::new();
    for &coded_id in coded_ids {
        if let Some(data_id) = delayed_encoder.encode_coded_vector(coded_id) {
            mappings.insert(coded_id, data_id);
        }
    }
    let encoding_ops = delayed_encoder.manager.move_new_operations();

    let symbol_size = message_vectors[0].len();
    let mut replay_operator = make_operator(symbol_size);
    replay_operator.config_finite_field(field_pp);
    for (i, vector) in message_vectors.iter().enumerate() {
        replay_operator.insert_vector(vector, i);
    }
    replay_operations(&mut *replay_operator, &precoding_ops);
    replay_operations(&mut *replay_operator, &encoding_ops);

    assert_replayed_coded_vectors_match(
        replay_operator.as_ref(),
        &mappings,
        &coded_payloads,
        "delayed encoder replay",
    );

  // Decode with a fresh operator, feeding payloads from the replayed encoder state.
    let mut decoder = Decoder::new_with_operator(&scheme, make_operator(symbol_size));
    let mut decoded = false;
    for &coded_id in coded_ids {
        let data_id = *mappings
            .get(&coded_id)
            .unwrap_or_else(|| panic!("missing mapping for coded_id {coded_id}"));
        let payload = replay_operator.get_vector(data_id);
        if matches!(
            decoder.add_coded_vector(coded_id, payload),
            DecodeStatus::Decoded
        ) {
            decoded = true;
            break;
        }
    }
    let decoding_ops = decoder.manager.move_new_operations();
    let dec_operator = decoder.manager.move_operator();
    let num_mismatches = count_message_mismatches(&*dec_operator, message_vectors, decoded);

    assert_eq!(
        reference.num_mismatches, num_mismatches,
        "delayed replay mismatch count differs from on-the-fly reference"
    );
    assert_eq!(reference.decoded, decoded, "delayed decode status differs from reference");

    RoundtripResult {
        num_mismatches,
        decoded,
        precoding_ops,
        encoding_ops,
        decoding_ops,
    }
}

/// Assert `candidate` roundtrip matches `reference_factory` on the same inputs.
pub fn assert_roundtrip_matches_reference<C: CodeScheme + Clone>(
    scheme: C,
    message_vectors: &[Vec<u8>],
    coded_ids: &[usize],
    candidate_factory: OperatorFactory,
    reference_factory: OperatorFactory,
) {
    let reference = roundtrip_on_the_fly(
        scheme.clone(),
        message_vectors,
        coded_ids,
        reference_factory,
    );
    assert!(
        reference.success(),
        "reference operator failed: decoded={} mismatches={}",
        reference.decoded,
        reference.num_mismatches
    );

    let candidate = roundtrip_on_the_fly(scheme, message_vectors, coded_ids, candidate_factory);
    assert!(
        candidate.success(),
        "candidate operator failed: decoded={} mismatches={}",
        candidate.decoded,
        candidate.num_mismatches
    );
}

fn capture_coded_vectors_on_the_fly<C: CodeScheme + Clone>(
    scheme: C,
    message_vectors: &[Vec<u8>],
    coded_ids: &[usize],
    make_operator: OperatorFactory,
) -> HashMap<usize, Vec<u8>> {
    let symbol_size = message_vectors[0].len();
    let mut enc_operator = make_operator(symbol_size);
    for (i, vector) in message_vectors.iter().enumerate() {
        enc_operator.insert_vector(vector, i);
    }

    let mut encoder = Encoder::new_with_operator(&scheme, enc_operator);
    let _ = encoder.manager.move_new_operations();

    let mut payloads = HashMap::new();
    for &coded_id in coded_ids {
        encoder.encode_coded_vector(coded_id);
        payloads.insert(coded_id, encoder.manager.get_coded_vector(coded_id));
    }
    payloads
}

fn assert_replayed_coded_vectors_match(
    operator: &dyn DataOperator,
    mappings: &HashMap<usize, usize>,
    expected: &HashMap<usize, Vec<u8>>,
    context: &str,
) {
    for (coded_id, expected_payload) in expected {
        let data_id = mappings
            .get(coded_id)
            .unwrap_or_else(|| panic!("{context}: missing mapping for coded_id {coded_id}"));
        assert_eq!(
            operator.get_vector(*data_id),
            expected_payload.as_slice(),
            "{context}: coded_id {coded_id}"
        );
    }
}

fn count_message_mismatches(
    operator: &dyn DataOperator,
    message_vectors: &[Vec<u8>],
    decoded: bool,
) -> usize {
    if !decoded {
        return message_vectors.len();
    }
    message_vectors
        .iter()
        .enumerate()
        .filter(|(i, expected)| operator.get_vector(*i) != expected.as_slice())
        .count()
}
