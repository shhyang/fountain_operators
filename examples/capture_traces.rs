//! Regenerate scheme operation traces (plan §7.4).
//!
//! ```bash
//! cargo run -p fountain_operators --example capture_traces
//! ```
//!
//! Writes to `bench_data/traces/` (committed fixtures) and `test_results/traces/` (local, gitignored).

use std::path::PathBuf;

use fountain_engine::traits::{CodeScheme, DataOperator};
use fountain_engine::types::GF2_FIELD_POLY;
use fountain_engine::Encoder;
use fountain_operators::{
    fountain_coded_ids, lt_coded_ids, make_test_messages, CapturedTrace, TraceError,
};
use fountain_scheme::validation::pseudo_rand::XorShift64;
use fountain_scheme::{BinaryHDPCLTCode, HDPCLTCode, RandomLTCode};
use fountain_utility::VecDataOperater;

fn vec_factory(symbol_size: usize) -> Box<dyn DataOperator> {
    Box::new(VecDataOperater::new(symbol_size))
}

/// Capture ops from a delayed encoder log (plan §7.3); decoding ops from a Vec on-the-fly run.
fn capture<C: CodeScheme + Clone>(
    name: &str,
    scheme_label: &str,
    scheme: C,
    messages: &[Vec<u8>],
    coded_ids: &[usize],
    field_pp: Option<u16>,
) -> CapturedTrace {
    use fountain_operators::roundtrip_on_the_fly;

    // Capture encoder ops before any on-the-fly run (LT degree RNG is stateful).
    let mut encoder = Encoder::new(&scheme.clone());
    let precoding_ops = encoder.manager.move_new_operations();
    for &coded_id in coded_ids {
        encoder.encode_coded_vector(coded_id);
    }
    let encoding_ops = encoder.manager.move_new_operations();

    let reference = roundtrip_on_the_fly(scheme, messages, coded_ids, vec_factory);
    assert!(
        reference.success(),
        "{name}: decoded={} mismatches={}",
        reference.decoded,
        reference.num_mismatches
    );

    CapturedTrace::from_roundtrip(
        name,
        scheme_label,
        messages.len(),
        messages[0].len(),
        field_pp,
        &precoding_ops,
        &encoding_ops,
        &reference.decoding_ops,
    )
}

fn write_trace(trace: &CapturedTrace, dir: &PathBuf) -> Result<(), TraceError> {
    std::fs::create_dir_all(dir)?;
    trace.save_json(dir.join(trace.suggested_filename()))?;
    Ok(())
}

fn main() -> Result<(), TraceError> {
    let bench_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("bench_data/traces");
    let local_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../test_results/traces");

    let mut traces = Vec::new();

    {
        let k = 15;
        let t = 64;
        let scheme = RandomLTCode::new_from_robust_soliton(k, XorShift64::new(0x00C0_FFEE));
        let messages = make_test_messages(k, t);
        traces.push(capture(
            "random_lt",
            "RandomLTCode",
            scheme,
            &messages,
            &lt_coded_ids(k),
            Some(GF2_FIELD_POLY),
        ));
    }

    {
        let k = 14;
        let t = 1;
        let scheme = HDPCLTCode::new_with_ideal_soliton(k, XorShift64::new(0x00C0_FFEE));
        let params = scheme.get_params();
        let messages = make_test_messages(k, t);
        traces.push(capture(
            "hdpc_gf256",
            "HDPCLTCode",
            scheme,
            &messages,
            &fountain_coded_ids(&params, k, 0),
            Some(0x11D),
        ));
    }

    {
        let k = 14;
        let t = 1;
        let scheme = BinaryHDPCLTCode::new_with_ideal_soliton(k, XorShift64::new(0x00C0_FFEE));
        let params = scheme.get_params();
        let messages = make_test_messages(k, t);
        traces.push(capture(
            "binary_hdpc",
            "BinaryHDPCLTCode",
            scheme,
            &messages,
            &fountain_coded_ids(&params, k, 0),
            Some(GF2_FIELD_POLY),
        ));
    }

    for trace in &traces {
        write_trace(trace, &bench_dir)?;
        let _ = write_trace(trace, &local_dir);
        println!("wrote {}", trace.suggested_filename());
    }

    Ok(())
}
