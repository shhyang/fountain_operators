// Copyright (c) 2025 Shenghao Yang. All rights reserved.
// Licensed under the MIT License. See LICENSE-MIT for details.

//! JSON serialization for captured [`Operation`](fountain_engine::types::Operation) logs (plan §7.4).

use std::fs;
use std::path::Path;

use fountain_engine::types::Operation;
use serde::{Deserialize, Serialize};

/// JSON-friendly mirror of [`Operation`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum OperationJson {
    EnsureZero {
        list_id: Vec<usize>,
    },
    MultiplyAlpha {
        id: usize,
    },
    MultiplyScalar {
        scalar: u8,
        id: usize,
    },
    AddToVector {
        list_id: Vec<usize>,
        target_id: usize,
    },
    BroadcastAdd {
        src_id: usize,
        target_ids: Vec<usize>,
    },
    MulAdd {
        src_id: usize,
        scalar: u8,
        target_id: usize,
    },
    MoveTo {
        src_id: usize,
        target_id: usize,
    },
    CopyTo {
        src_id: usize,
        target_id: usize,
    },
    Remove {
        id: usize,
    },
    InfoCodedVector {
        coded_id: usize,
        data_id: usize,
    },
}

impl From<&Operation> for OperationJson {
    fn from(op: &Operation) -> Self {
        match op {
            Operation::EnsureZero { list_id } => OperationJson::EnsureZero {
                list_id: list_id.clone(),
            },
            Operation::MultiplyAlpha { id } => OperationJson::MultiplyAlpha { id: *id },
            Operation::MultiplyScalar { scalar, id } => OperationJson::MultiplyScalar {
                scalar: *scalar,
                id: *id,
            },
            Operation::AddOneToVector { src_id, target_id } => OperationJson::AddToVector {
                list_id: vec![*src_id],
                target_id: *target_id,
            },
            Operation::AddTwoToVector { s0, s1, target_id } => OperationJson::AddToVector {
                list_id: vec![*s0, *s1],
                target_id: *target_id,
            },
            Operation::AddThreeToVector {
                s0,
                s1,
                s2,
                target_id,
            } => OperationJson::AddToVector {
                list_id: vec![*s0, *s1, *s2],
                target_id: *target_id,
            },
            Operation::AddToVector { list_id, target_id } => OperationJson::AddToVector {
                list_id: list_id.clone(),
                target_id: *target_id,
            },
            Operation::BroadcastAdd { src_id, target_ids } => OperationJson::BroadcastAdd {
                src_id: *src_id,
                target_ids: target_ids.clone(),
            },
            Operation::MulAdd {
                src_id,
                scalar,
                target_id,
            } => OperationJson::MulAdd {
                src_id: *src_id,
                scalar: *scalar,
                target_id: *target_id,
            },
            Operation::MoveTo { src_id, target_id } => OperationJson::MoveTo {
                src_id: *src_id,
                target_id: *target_id,
            },
            Operation::CopyTo { src_id, target_id } => OperationJson::CopyTo {
                src_id: *src_id,
                target_id: *target_id,
            },
            Operation::Remove { id } => OperationJson::Remove { id: *id },
            Operation::InfoCodedVector { coded_id, data_id } => OperationJson::InfoCodedVector {
                coded_id: *coded_id,
                data_id: *data_id,
            },
        }
    }
}

impl From<OperationJson> for Operation {
    fn from(json: OperationJson) -> Self {
        match json {
            OperationJson::EnsureZero { list_id } => Operation::EnsureZero { list_id },
            OperationJson::MultiplyAlpha { id } => Operation::MultiplyAlpha { id },
            OperationJson::MultiplyScalar { scalar, id } => Operation::MultiplyScalar { scalar, id },
            OperationJson::AddToVector { list_id, target_id } => {
                Operation::AddToVector { list_id, target_id }
            }
            OperationJson::BroadcastAdd { src_id, target_ids } => {
                Operation::BroadcastAdd { src_id, target_ids }
            }
            OperationJson::MulAdd {
                src_id,
                scalar,
                target_id,
            } => Operation::MulAdd {
                src_id,
                scalar,
                target_id,
            },
            OperationJson::MoveTo { src_id, target_id } => Operation::MoveTo { src_id, target_id },
            OperationJson::CopyTo { src_id, target_id } => Operation::CopyTo { src_id, target_id },
            OperationJson::Remove { id } => Operation::Remove { id },
            OperationJson::InfoCodedVector { coded_id, data_id } => {
                Operation::InfoCodedVector { coded_id, data_id }
            }
        }
    }
}

fn to_json_ops(ops: &[Operation]) -> Vec<OperationJson> {
    ops.iter().map(OperationJson::from).collect()
}

fn from_json_ops(ops: &[OperationJson]) -> Vec<Operation> {
    ops.iter().cloned().map(Operation::from).collect()
}

/// Persistent encode/decode operation log (plan §7.4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedTrace {
    pub name: String,
    pub scheme: String,
    pub k: usize,
    pub symbol_size: usize,
    /// `None` for GF(2); `Some(pp)` for GF(256).
    pub field_pp: Option<u16>,
    pub precoding: Vec<OperationJson>,
    pub encoding: Vec<OperationJson>,
    pub decoding: Vec<OperationJson>,
}

impl CapturedTrace {
    #[must_use]
    pub fn all_operations(&self) -> Vec<Operation> {
        let mut ops = from_json_ops(&self.precoding);
        ops.extend(from_json_ops(&self.encoding));
        ops.extend(from_json_ops(&self.decoding));
        ops
    }

    #[must_use]
    pub fn precoding_operations(&self) -> Vec<Operation> {
        from_json_ops(&self.precoding)
    }

    #[must_use]
    pub fn encoding_operations(&self) -> Vec<Operation> {
        from_json_ops(&self.encoding)
    }

    #[must_use]
    pub fn decoding_operations(&self) -> Vec<Operation> {
        from_json_ops(&self.decoding)
    }

    /// Precoding + encoding (valid on a fresh operator with messages `0..k-1` inserted).
    #[must_use]
    pub fn encoder_operations(&self) -> Vec<Operation> {
        let mut ops = self.precoding_operations();
        ops.extend(self.encoding_operations());
        ops
    }

    /// Suggested filename: `scheme_k{K}_t{T}.json` with optional name prefix.
    #[must_use]
    pub fn suggested_filename(&self) -> String {
        format!("{}_k{}_t{}.json", self.name, self.k, self.symbol_size)
    }

    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<(), TraceError> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    pub fn load_json(path: impl AsRef<Path>) -> Result<Self, TraceError> {
        let text = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }

    pub fn from_roundtrip(
        name: impl Into<String>,
        scheme: impl Into<String>,
        k: usize,
        symbol_size: usize,
        field_pp: Option<u16>,
        precoding: &[Operation],
        encoding: &[Operation],
        decoding: &[Operation],
    ) -> Self {
        Self {
            name: name.into(),
            scheme: scheme.into(),
            k,
            symbol_size,
            field_pp,
            precoding: to_json_ops(precoding),
            encoding: to_json_ops(encoding),
            decoding: to_json_ops(decoding),
        }
    }
}

#[derive(Debug)]
pub enum TraceError {
    Io(std::io::Error),
    Json(serde_json::Error),
}

impl std::fmt::Display for TraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TraceError::Io(e) => write!(f, "io error: {e}"),
            TraceError::Json(e) => write!(f, "json error: {e}"),
        }
    }
}

impl std::error::Error for TraceError {}

impl From<std::io::Error> for TraceError {
    fn from(e: std::io::Error) -> Self {
        TraceError::Io(e)
    }
}

impl From<serde_json::Error> for TraceError {
    fn from(e: serde_json::Error) -> Self {
        TraceError::Json(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fountain_engine::types::Operation;

    #[test]
    fn operation_json_roundtrip() {
        let ops = vec![
            Operation::EnsureZero {
                list_id: vec![1, 2],
            },
            Operation::MulAdd {
                src_id: 0,
                scalar: 3,
                target_id: 1,
            },
        ];
        let json: Vec<OperationJson> = ops.iter().map(OperationJson::from).collect();
        let back: Vec<Operation> = json.iter().cloned().map(Operation::from).collect();
        let json2: Vec<OperationJson> = back.iter().map(OperationJson::from).collect();
        assert_eq!(json, json2);
    }
}
