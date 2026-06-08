// Copyright (c) 2025 Shenghao Yang. All rights reserved.
// Licensed under the MIT License. See LICENSE-MIT for details.

//! [`DataOperator`] with SIMD kernels selected once at construction (phase 3a).

use fountain_engine::algebra::finite_field::GF256;
use fountain_engine::traits::DataOperator;
use fountain_engine::types::Operation;

use crate::gf_kernels::select_kernel;
use crate::slab_data_operator::SlabDataOperator;
use crate::slab_storage::SlabStorage;

/// Slab-backed operator using [`select_kernel()`] (AVX2 when `feature = "simd"` and CPU supports it).
///
/// On hosts without AVX2, behavior matches [`SlabDataOperator`] with the portable kernel.
pub struct SimdDataOperator(SlabDataOperator);

impl SimdDataOperator {
    #[must_use]
    pub fn new(vector_len: usize) -> Self {
        Self(SlabDataOperator::with_kernel(vector_len, None, select_kernel()))
    }

    #[must_use]
    pub fn new_with_gf256(vector_len: usize, pp: u16) -> Self {
        Self(SlabDataOperator::with_kernel(
            vector_len,
            Some(GF256::new_with_primitive_polynomial(pp)),
            select_kernel(),
        ))
    }

    #[must_use]
    pub fn gf256(&self) -> Option<&GF256> {
        self.0.gf256()
    }

    #[must_use]
    pub fn storage(&self) -> &SlabStorage {
        self.0.storage()
    }

    #[must_use]
    pub fn storage_mut(&mut self) -> &mut SlabStorage {
        self.0.storage_mut()
    }
}

impl DataOperator for SimdDataOperator {
    fn config_finite_field(&mut self, pp: u16) {
        self.0.config_finite_field(pp);
    }

    fn config_finite_field_from(&mut self, gf: &GF256) {
        self.0.config_finite_field_from(gf);
    }

    fn insert_vector(&mut self, src: &[u8], data_id: usize) {
        self.0.insert_vector(src, data_id);
    }

    fn get_vector(&self, data_id: usize) -> &[u8] {
        self.0.get_vector(data_id)
    }

    fn execute(&mut self, operation: &Operation) {
        self.0.execute(operation);
    }
}
