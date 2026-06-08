//! Load committed bench traces; replay encoder ops and compare operator state.

mod common;

use common::{make_simd_operator, make_slab_operator, make_vec_operator};
use fountain_operators::{replay_operations, CapturedTrace};

fn trace_path(name: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("bench_data/traces")
        .join(format!("{name}.json"))
}

fn prepare_messages(trace: &CapturedTrace, op: &mut dyn fountain_engine::traits::DataOperator) {
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
}

fn replay_encoder(op: &mut dyn fountain_engine::traits::DataOperator, trace: &CapturedTrace) {
    prepare_messages(trace, op);
    let precoding = trace.precoding_operations();
    let encoding = trace.encoding_operations();
    if precoding.is_empty() {
        replay_operations(op, &encoding);
    } else {
        replay_operations(op, &precoding);
        replay_operations(op, &encoding);
    }
}

fn encoder_data_ids(trace: &CapturedTrace) -> Vec<usize> {
    let mut ids = std::collections::BTreeSet::new();
    for op in trace.encoder_operations().iter() {
        match op {
            fountain_engine::types::Operation::EnsureZero { list_id } => {
                ids.extend(list_id);
            }
            fountain_engine::types::Operation::MultiplyAlpha { id }
            | fountain_engine::types::Operation::MultiplyScalar { id, .. }
            | fountain_engine::types::Operation::Remove { id } => {
                ids.insert(*id);
            }
            fountain_engine::types::Operation::AddOneToVector { src_id, target_id } => {
                ids.insert(*src_id);
                ids.insert(*target_id);
            }
            fountain_engine::types::Operation::AddTwoToVector { s0, s1, target_id } => {
                ids.insert(*s0);
                ids.insert(*s1);
                ids.insert(*target_id);
            }
            fountain_engine::types::Operation::AddThreeToVector {
                s0,
                s1,
                s2,
                target_id,
            } => {
                ids.insert(*s0);
                ids.insert(*s1);
                ids.insert(*s2);
                ids.insert(*target_id);
            }
            fountain_engine::types::Operation::AddToVector { list_id, target_id } => {
                ids.extend(list_id);
                ids.insert(*target_id);
            }
            fountain_engine::types::Operation::BroadcastAdd { src_id, target_ids } => {
                ids.insert(*src_id);
                ids.extend(target_ids);
            }
            fountain_engine::types::Operation::MulAdd {
                src_id,
                target_id,
                ..
            } => {
                ids.insert(*src_id);
                ids.insert(*target_id);
            }
            fountain_engine::types::Operation::MoveTo { src_id, target_id }
            | fountain_engine::types::Operation::CopyTo { src_id, target_id } => {
                ids.insert(*src_id);
                ids.insert(*target_id);
            }
            fountain_engine::types::Operation::InfoCodedVector { data_id, .. } => {
                ids.insert(*data_id);
            }
        }
    }
    ids.into_iter().collect()
}

#[test]
fn replay_hdpc_encoder_trace_on_vec_and_slab() {
    let trace = CapturedTrace::load_json(trace_path("hdpc_gf256_k14_t1")).expect("load trace");
    let mut vec_op = make_vec_operator(trace.symbol_size);
    let mut slab_op = make_slab_operator(trace.symbol_size);
    replay_encoder(vec_op.as_mut(), &trace);
    replay_encoder(slab_op.as_mut(), &trace);
}

#[test]
fn replay_binary_hdpc_encoder_trace_on_vec_and_slab() {
    let trace = CapturedTrace::load_json(trace_path("binary_hdpc_k14_t1")).expect("load trace");
    let mut vec_op = make_vec_operator(trace.symbol_size);
    let mut slab_op = make_slab_operator(trace.symbol_size);
    replay_encoder(vec_op.as_mut(), &trace);
    replay_encoder(slab_op.as_mut(), &trace);
}

fn assert_operator_bytes_match(
    trace: &CapturedTrace,
    a: &dyn fountain_engine::traits::DataOperator,
    b: &dyn fountain_engine::traits::DataOperator,
) {
    for id in encoder_data_ids(trace) {
        assert_eq!(a.get_vector(id), b.get_vector(id), "data_id {id}");
    }
}

#[test]
fn replay_random_lt_encoder_trace_simd_matches_slab() {
    let trace = CapturedTrace::load_json(trace_path("random_lt_k15_t64")).expect("load trace");
    let mut slab_op = make_slab_operator(trace.symbol_size);
    let mut simd_op = make_simd_operator(trace.symbol_size);
    replay_encoder(slab_op.as_mut(), &trace);
    replay_encoder(simd_op.as_mut(), &trace);
    assert_operator_bytes_match(&trace, slab_op.as_ref(), simd_op.as_ref());
}
