// Copyright (c) 2025 Shenghao Yang. All rights reserved.
// Licensed under the MIT License. See LICENSE-MIT for details.

//! [`DataOperator`] backed by contiguous [`SlabStorage`].

use fountain_engine::algebra::finite_field::GF256;
use fountain_engine::traits::DataOperator;
use fountain_engine::types::{Operation, GF2_FIELD_POLY};

use crate::gf_kernels::{default_kernel, GfBlockKernel};
use crate::slab_storage::SlabStorage;

/// Slab-backed operator: one contiguous buffer, `data_id → slot` map.
///
/// Operation semantics match [`VecDataOperater`](fountain_utility::VecDataOperater) (golden reference).
/// Uses the portable kernel via [`default_kernel()`]; see [`SimdDataOperator`](crate::simd_data_operator::SimdDataOperator) for SIMD.
pub struct SlabDataOperator {
    storage: SlabStorage,
    gf256: Option<GF256>,
    kernel: &'static dyn GfBlockKernel,
}

impl SlabDataOperator {
    #[must_use]
    pub fn new(vector_len: usize) -> Self {
        Self::with_kernel(vector_len, None, default_kernel())
    }

    #[must_use]
    pub fn new_with_gf256(vector_len: usize, pp: u16) -> Self {
        Self::with_kernel(
            vector_len,
            Some(GF256::new_with_primitive_polynomial(pp)),
            default_kernel(),
        )
    }

    /// Construct with an explicit kernel (used by [`SimdDataOperator`](crate::simd_data_operator::SimdDataOperator)).
    #[must_use]
    pub fn with_kernel(
        vector_len: usize,
        gf256: Option<GF256>,
        kernel: &'static dyn GfBlockKernel,
    ) -> Self {
        Self {
            storage: SlabStorage::new(vector_len),
            gf256,
            kernel,
        }
    }

    #[must_use]
    pub fn gf256(&self) -> Option<&GF256> {
        self.gf256.as_ref()
    }

    #[must_use]
    pub fn storage(&self) -> &SlabStorage {
        &self.storage
    }

    #[must_use]
    pub fn storage_mut(&mut self) -> &mut SlabStorage {
        &mut self.storage
    }

    fn ensure_zero(&mut self, list_id: &[usize]) {
        for &id in list_id {
            self.storage.ensure_zero_by_id(id);
        }
    }

    fn multiply_alpha(&mut self, vector_id: usize) {
        let gf = self
            .gf256
            .as_ref()
            .expect("GF(256) is not set on SlabDataOperator");
        let slot = self.storage.slot_for_id(vector_id);
        self.kernel
            .mul_alpha_inplace(gf, self.storage.slot_slice_mut(slot));
    }

    fn multiply_scalar(&mut self, scalar: u8, vector_id: usize) {
        let slot = self.storage.slot_for_id(vector_id);
        match scalar {
            0 => self.storage.slot_slice_mut(slot).fill(0),
            1 => {}
            _ => {
                let gf = self
                    .gf256
                    .as_ref()
                    .expect("GF(256) is not set on SlabDataOperator");
                self.kernel.mul_scalar_inplace(
                    gf,
                    self.storage.slot_slice_mut(slot),
                    scalar,
                );
            }
        }
    }

    fn add_to_vector(&mut self, list_id: &[usize], target_id: usize) {
        let target_slot = self.storage.slot_for_id(target_id);
        match list_id.len() {
            0 => {}
            1 => {
                if let Some(src_slot) = self.storage.slot_for_id_opt(list_id[0]) {
                    if src_slot != target_slot {
                        let (dst, src) = self.storage.get_pair_mut(target_slot, src_slot);
                        self.kernel.xor_inplace(dst, src);
                    }
                }
            }
            2 => {
                let s0 = list_id[0];
                let s1 = list_id[1];
                if s0 == s1 {
                    if let Some(src_slot) = self.storage.slot_for_id_opt(s0) {
                        if src_slot != target_slot {
                            let (dst, src) = self.storage.get_pair_mut(target_slot, src_slot);
                            self.kernel.xor_inplace(dst, src);
                        }
                    }
                    return;
                }
                let s0_slot = self.storage.slot_for_id_opt(s0);
                let s1_slot = self.storage.slot_for_id_opt(s1);
                if let (Some(s0_slot), Some(s1_slot)) = (s0_slot, s1_slot) {
                    if s0_slot == target_slot {
                        let (dst, src) = self.storage.get_pair_mut(target_slot, s1_slot);
                        self.kernel.xor_inplace(dst, src);
                    } else if s1_slot == target_slot {
                        let (dst, src) = self.storage.get_pair_mut(target_slot, s0_slot);
                        self.kernel.xor_inplace(dst, src);
                    } else {
                        self.storage
                            .xor_two_slots(target_slot, s0_slot, s1_slot);
                    }
                }
            }
            3 => {
                let s0 = list_id[0];
                let s1 = list_id[1];
                let s2 = list_id[2];
                if let (Some(s0_slot), Some(s1_slot), Some(s2_slot)) = (
                    self.storage.slot_for_id_opt(s0),
                    self.storage.slot_for_id_opt(s1),
                    self.storage.slot_for_id_opt(s2),
                ) {
                    self.storage
                        .xor_three_slots(target_slot, s0_slot, s1_slot, s2_slot);
                }
            }
            _ => {
                for &id in list_id {
                    if let Some(src_slot) = self.storage.slot_for_id_opt(id) {
                        if src_slot != target_slot {
                            let (dst, src) = self.storage.get_pair_mut(target_slot, src_slot);
                            self.kernel.xor_inplace(dst, src);
                        }
                    }
                }
            }
        }
    }

    fn broadcast_add(&mut self, source_id: usize, target_ids: &[usize]) {
        let src_slot = self.storage.slot_for_id(source_id);
        for &target_id in target_ids {
            let target_slot = self.storage.slot_for_id(target_id);
            if src_slot == target_slot {
                continue;
            }
            let (dst, src) = self.storage.get_pair_mut(target_slot, src_slot);
            self.kernel.xor_inplace(dst, src);
        }
    }

    fn mul_add(&mut self, source_id: usize, scalar: u8, target_id: usize) {
        let target_slot = self.storage.slot_for_id(target_id);
        let src_slot = self.storage.slot_for_id(source_id);
        match scalar {
            0 => self.storage.slot_slice_mut(target_slot).fill(0),
            1 => self.add_to_vector(&[source_id], target_id),
            _ => {
                let gf = self
                    .gf256
                    .as_ref()
                    .expect("GF(256) is not set on SlabDataOperator");
                if src_slot == target_slot {
                    let dst = self.storage.slot_slice_mut(target_slot);
                    for b in dst.iter_mut() {
                        *b ^= gf.mul_lookup(scalar, *b);
                    }
                } else {
                    let (dst, src) = self.storage.get_pair_mut(target_slot, src_slot);
                    self.kernel.mul_add_inplace(gf, dst, src, scalar);
                }
            }
        }
    }

    fn move_to(&mut self, src_id: usize, target_id: usize) {
        self.storage.move_id(src_id, target_id);
    }

    fn copy_to(&mut self, src_id: usize, target_id: usize) {
        self.storage.copy_id(src_id, target_id);
    }

    fn remove(&mut self, id: usize) {
        self.storage.remove_id(id);
    }
}

impl DataOperator for SlabDataOperator {
    fn config_finite_field(&mut self, pp: u16) {
        if pp == GF2_FIELD_POLY {
            self.gf256 = None;
            return;
        }
        if self
            .gf256
            .as_ref()
            .is_some_and(|gf| gf.primitive_polynomial() == pp)
        {
            return;
        }
        self.gf256 = Some(GF256::new_with_primitive_polynomial(pp));
    }

    fn config_finite_field_from(&mut self, gf: &GF256) {
        if self
            .gf256
            .as_ref()
            .is_some_and(|f| f.primitive_polynomial() == gf.primitive_polynomial())
        {
            return;
        }
        self.gf256 = Some(gf.clone());
    }

    fn insert_vector(&mut self, src: &[u8], data_id: usize) {
        self.storage.insert_by_id(data_id, src);
    }

    fn get_vector(&self, data_id: usize) -> &[u8] {
        self.storage.get_by_id(data_id)
    }

    fn execute(&mut self, operation: &Operation) {
        match operation {
            Operation::EnsureZero { list_id } => self.ensure_zero(list_id),
            Operation::MultiplyAlpha { id } => self.multiply_alpha(*id),
            Operation::MultiplyScalar { scalar, id } => self.multiply_scalar(*scalar, *id),
            Operation::AddOneToVector { src_id, target_id } => {
                self.add_to_vector(&[*src_id], *target_id);
            }
            Operation::AddTwoToVector { s0, s1, target_id } => {
                self.add_to_vector(&[*s0, *s1], *target_id);
            }
            Operation::AddThreeToVector {
                s0,
                s1,
                s2,
                target_id,
            } => {
                self.add_to_vector(&[*s0, *s1, *s2], *target_id);
            }
            Operation::AddToVector { list_id, target_id } => {
                self.add_to_vector(list_id, *target_id);
            }
            Operation::BroadcastAdd { src_id, target_ids } => {
                self.broadcast_add(*src_id, target_ids);
            }
            Operation::MulAdd {
                src_id,
                scalar,
                target_id,
            } => self.mul_add(*src_id, *scalar, *target_id),
            Operation::MoveTo { src_id, target_id } => self.move_to(*src_id, *target_id),
            Operation::CopyTo { src_id, target_id } => self.copy_to(*src_id, *target_id),
            Operation::Remove { id } => self.remove(*id),
            Operation::InfoCodedVector { .. } => {}
        }
    }
}
