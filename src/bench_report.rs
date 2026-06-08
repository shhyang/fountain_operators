// Copyright (c) 2025 Shenghao Yang. See LICENSE-MIT for details.

//! JSONL benchmark records (plan §8.6).

use serde::{Deserialize, Serialize};

/// One benchmark sample line for Layer 1 or Layer 2 reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchRecord {
    pub operator: String,
    pub scheme: String,
    pub k: usize,
    pub symbol_size: usize,
    pub phase: String,
    pub mode: String,
    pub replay_wall_ms: f64,
    pub e2e_wall_ms: Option<f64>,
    pub throughput_mib_s: Option<f64>,
    pub trace: Option<String>,
}

impl BenchRecord {
    pub fn to_jsonl_line(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// Median of samples (empty slice → 0.0).
#[must_use]
pub fn median_ms(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f64> = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    if v.len() % 2 == 0 {
        (v[mid - 1] + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

/// Throughput in MiB/s for `bytes` processed in `wall_ms`.
#[must_use]
pub fn throughput_mib_s(bytes: usize, wall_ms: f64) -> f64 {
    if wall_ms <= 0.0 {
        return 0.0;
    }
    let mib = bytes as f64 / (1024.0 * 1024.0);
    mib / (wall_ms / 1000.0)
}
