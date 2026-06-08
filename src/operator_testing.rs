// Copyright (c) 2025 Shenghao Yang. All rights reserved.
// Licensed under the MIT License. See LICENSE-MIT for details.

//! Differential testing helpers: compare two [`DataOperator`] implementations on the same op log.

use std::collections::{HashMap, HashSet};

use fountain_engine::traits::DataOperator;
use fountain_engine::types::{Operation, GF2_FIELD_POLY};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Snapshot of all tracked data IDs and their payloads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorSnapshot {
    vectors: HashMap<usize, Vec<u8>>,
}

impl OperatorSnapshot {
    /// Builds a snapshot from an operator for the given IDs (panics if a vector is missing).
    #[must_use]
    pub fn capture(operator: &dyn DataOperator, ids: &HashSet<usize>) -> Self {
        let mut vectors = HashMap::new();
        for &id in ids {
            vectors.insert(id, operator.get_vector(id).to_vec());
        }
        Self { vectors }
    }
}

/// Panics if the two operators differ after the same setup and operation sequence.
pub fn assert_operators_equivalent(
    left: &mut dyn DataOperator,
    right: &mut dyn DataOperator,
    vector_len: usize,
    initial_vectors: &[(usize, Vec<u8>)],
    ops: &[Operation],
) {
    let mut tracked = HashSet::new();
    for (id, data) in initial_vectors {
        assert_eq!(data.len(), vector_len);
        left.insert_vector(data, *id);
        right.insert_vector(data, *id);
        tracked.insert(*id);
    }

    apply_ops(left, &mut tracked, ops);
    apply_ops(right, &mut tracked, ops);

    let snap_left = OperatorSnapshot::capture(left, &tracked);
    let snap_right = OperatorSnapshot::capture(right, &tracked);
    assert_eq!(
        snap_left, snap_right,
        "operator states diverged after {} operations",
        ops.len()
    );
}

/// Runs a seeded random operation sequence; both operators must stay byte-identical.
pub fn differential_test_random_ops(
    left: &mut dyn DataOperator,
    right: &mut dyn DataOperator,
    vector_len: usize,
    num_ids: usize,
    num_ops: usize,
    seed: u64,
    gf256_pp: u16,
) {
    if gf256_pp == GF2_FIELD_POLY {
        left.config_finite_field(GF2_FIELD_POLY);
        right.config_finite_field(GF2_FIELD_POLY);
    } else {
        left.config_finite_field(gf256_pp);
        right.config_finite_field(gf256_pp);
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let ids: Vec<usize> = (0..num_ids).collect();
    let mut initial = Vec::new();
    for &id in &ids {
        let mut v = vec![0u8; vector_len];
        rng.fill(&mut v[..]);
        initial.push((id, v));
    }

    let ops = generate_random_ops(&mut rng, &ids, num_ops, gf256_pp);
    assert_operators_equivalent(left, right, vector_len, &initial, &ops);
}

fn apply_ops(operator: &mut dyn DataOperator, tracked: &mut HashSet<usize>, ops: &[Operation]) {
    for op in ops {
        track_op_ids(tracked, op);
        operator.execute(op);
        if let Operation::Remove { id } = op {
            tracked.remove(id);
        }
    }
}

fn track_op_ids(tracked: &mut HashSet<usize>, op: &Operation) {
    apply_op_to_live_set(tracked, op);
}

/// Generates random operations referencing only `ids` (all vectors assumed to exist initially).
///
/// Maintains a set of live IDs while generating so every op is valid for [`VecDataOperater`]
/// (targets must exist; `MoveTo` removes `src_id` from the live set).
pub fn generate_random_ops(
    rng: &mut StdRng,
    ids: &[usize],
    count: usize,
    gf256_pp: u16,
) -> Vec<Operation> {
    let mut live: HashSet<usize> = ids.iter().copied().collect();
    let mut ops = Vec::with_capacity(count);
    for _ in 0..count {
        let op = generate_one_random_op(rng, ids, &mut live, gf256_pp);
        apply_op_to_live_set(&mut live, &op);
        ops.push(op);
    }
    ops
}

fn pick_live(rng: &mut StdRng, live: &HashSet<usize>) -> usize {
    let idx = rng.gen_range(0..live.len());
    *live.iter().nth(idx).expect("non-empty live set")
}

fn generate_one_random_op(
    rng: &mut StdRng,
    ids: &[usize],
    live: &mut HashSet<usize>,
    gf256_pp: u16,
) -> Operation {
    if live.is_empty() {
        return Operation::EnsureZero {
            list_id: ids.to_vec(),
        };
    }

    let gf256 = gf256_pp != GF2_FIELD_POLY;
    let alpha = 0x02u8;
    let nontrivial_scalars: &[u8] = if gf256 {
        &[0, 1, 2, 7, alpha, 0xFF]
    } else {
        &[0, 1]
    };

    let id_a = pick_live(rng, live);
    let id_b = pick_live(rng, live);

    if gf256 {
        match rng.gen_range(0..10) {
            0 => Operation::EnsureZero {
                list_id: vec![id_a],
            },
            1 => Operation::MultiplyAlpha { id: id_a },
            2 => Operation::MultiplyScalar {
                scalar: nontrivial_scalars[rng.gen_range(0..nontrivial_scalars.len())],
                id: id_a,
            },
            3 => Operation::AddToVector {
                list_id: vec![id_a],
                target_id: id_b,
            },
            4 => Operation::BroadcastAdd {
                src_id: id_a,
                target_ids: vec![id_b],
            },
            5 => Operation::MulAdd {
                src_id: id_a,
                scalar: nontrivial_scalars[rng.gen_range(0..nontrivial_scalars.len())],
                target_id: id_b,
            },
            6 if id_a != id_b && live.len() > 1 => Operation::MoveTo {
                src_id: id_a,
                target_id: id_b,
            },
            7 => Operation::CopyTo {
                src_id: id_a,
                target_id: id_b,
            },
            8 if live.len() > 1 => Operation::Remove { id: id_a },
            _ => Operation::InfoCodedVector {
                coded_id: rng.gen_range(0..1000),
                data_id: id_a,
            },
        }
    } else {
        match rng.gen_range(0..7) {
            0 => Operation::EnsureZero {
                list_id: vec![id_a],
            },
            1 => Operation::AddToVector {
                list_id: vec![id_a],
                target_id: id_b,
            },
            2 => Operation::BroadcastAdd {
                src_id: id_a,
                target_ids: vec![id_b],
            },
            3 => Operation::MulAdd {
                src_id: id_a,
                scalar: nontrivial_scalars[rng.gen_range(0..nontrivial_scalars.len())],
                target_id: id_b,
            },
            4 if id_a != id_b && live.len() > 1 => Operation::MoveTo {
                src_id: id_a,
                target_id: id_b,
            },
            5 => Operation::CopyTo {
                src_id: id_a,
                target_id: id_b,
            },
            6 if live.len() > 1 => Operation::Remove { id: id_a },
            _ => Operation::EnsureZero {
                list_id: vec![id_a],
            },
        }
    }
}

fn apply_op_to_live_set(live: &mut HashSet<usize>, op: &Operation) {
    match op {
        Operation::EnsureZero { list_id } => {
            live.extend(list_id.iter().copied());
        }
        Operation::MultiplyAlpha { id }
        | Operation::MultiplyScalar { id, .. } => {
            live.insert(*id);
        }
        Operation::AddOneToVector { src_id, target_id } => {
            live.insert(*src_id);
            live.insert(*target_id);
        }
        Operation::AddTwoToVector { s0, s1, target_id } => {
            live.insert(*s0);
            live.insert(*s1);
            live.insert(*target_id);
        }
        Operation::AddThreeToVector {
            s0,
            s1,
            s2,
            target_id,
        } => {
            live.insert(*s0);
            live.insert(*s1);
            live.insert(*s2);
            live.insert(*target_id);
        }
        Operation::AddToVector { list_id, target_id } => {
            live.extend(list_id.iter().copied());
            live.insert(*target_id);
        }
        Operation::BroadcastAdd { src_id, target_ids } => {
            live.insert(*src_id);
            live.extend(target_ids.iter().copied());
        }
        Operation::MulAdd {
            src_id,
            target_id,
            ..
        } => {
            live.insert(*src_id);
            live.insert(*target_id);
        }
        Operation::MoveTo { src_id, target_id } => {
            live.remove(src_id);
            live.insert(*target_id);
        }
        Operation::CopyTo { src_id, target_id } => {
            live.insert(*src_id);
            live.insert(*target_id);
        }
        Operation::Remove { id } => {
            live.remove(id);
        }
        Operation::InfoCodedVector { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SlabDataOperator;
    use fountain_engine::types::Operation;
    use fountain_utility::VecDataOperater;

    fn fresh_pair(len: usize, pp: u16) -> (VecDataOperater, VecDataOperater) {
        (
            VecDataOperater::new_with_gf256(len, pp),
            VecDataOperater::new_with_gf256(len, pp),
        )
    }

    #[test]
    fn vec_vs_vec_identity_gf256() {
        let (mut a, mut b) = fresh_pair(64, 0x11D);
        differential_test_random_ops(&mut a, &mut b, 64, 6, 200, 42, 0x11D);
    }

    #[test]
    fn vec_vs_vec_identity_gf2() {
        let mut a = VecDataOperater::new(16);
        let mut b = VecDataOperater::new(16);
        differential_test_random_ops(&mut a, &mut b, 16, 5, 150, 99, GF2_FIELD_POLY);
    }

    #[test]
    fn slab_vs_vec_gf256() {
        let len = 64;
        let mut slab = SlabDataOperator::new_with_gf256(len, 0x11D);
        let mut vec = VecDataOperater::new_with_gf256(len, 0x11D);
        differential_test_random_ops(&mut slab, &mut vec, len, 6, 200, 42, 0x11D);
    }

    #[test]
    fn slab_vs_vec_gf2() {
        let len = 16;
        let mut slab = SlabDataOperator::new(len);
        let mut vec = VecDataOperater::new(len);
        differential_test_random_ops(&mut slab, &mut vec, len, 5, 150, 99, GF2_FIELD_POLY);
    }

    #[test]
    fn slab_vs_vec_vector_lengths() {
        for len in [1usize, 3, 64] {
            let mut slab = SlabDataOperator::new_with_gf256(len, 0x11D);
            let mut vec = VecDataOperater::new_with_gf256(len, 0x11D);
            differential_test_random_ops(&mut slab, &mut vec, len, 4, 80, 1000 + len as u64, 0x11D);
        }
    }

    #[test]
    fn simd_vs_vec_gf256() {
        let len = 64;
        let mut simd = crate::SimdDataOperator::new_with_gf256(len, 0x11D);
        let mut vec = VecDataOperater::new_with_gf256(len, 0x11D);
        differential_test_random_ops(&mut simd, &mut vec, len, 6, 200, 43, 0x11D);
    }

    #[test]
    fn simd_vs_slab_gf256() {
        let len = 128;
        let mut simd = crate::SimdDataOperator::new_with_gf256(len, 0x11D);
        let mut slab = SlabDataOperator::new_with_gf256(len, 0x11D);
        differential_test_random_ops(&mut simd, &mut slab, len, 6, 200, 44, 0x11D);
    }

    #[test]
    fn simd_vs_vec_gf2() {
        let len = 16;
        let mut simd = crate::SimdDataOperator::new(len);
        let mut vec = VecDataOperater::new(len);
        differential_test_random_ops(&mut simd, &mut vec, len, 5, 150, 100, GF2_FIELD_POLY);
    }

    #[test]
    fn hand_crafted_operation_sequence() {
        let (mut left, mut right) = fresh_pair(4, 0x11D);
        let initial = vec![
            (0, vec![1, 2, 3, 4]),
            (1, vec![5, 6, 7, 8]),
            (2, vec![0, 0, 0, 0]),
        ];
        let ops = vec![
            Operation::MulAdd {
                src_id: 0,
                scalar: 3,
                target_id: 1,
            },
            Operation::BroadcastAdd {
                src_id: 2,
                target_ids: vec![1],
            },
            Operation::MoveTo {
                src_id: 1,
                target_id: 3,
            },
        ];
        assert_operators_equivalent(&mut left, &mut right, 4, &initial, &ops);
        assert_eq!(left.get_vector(3), right.get_vector(3));
    }
}
