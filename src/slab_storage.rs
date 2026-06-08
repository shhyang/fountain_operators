// Copyright (c) 2025 Shenghao Yang. All rights reserved.
// Licensed under the MIT License. See LICENSE-MIT for details.

//! Contiguous byte slab with `data_id → slot` mapping (raptorq-style layout).

use std::collections::HashMap;

/// One `Vec<u8>` backing store; symbol `slot` occupies `[slot * vector_len .. (slot + 1) * vector_len)`.
///
/// Slot allocation matches [`VecDataOperater`](fountain_utility::VecDataOperater): new IDs always
/// get a fresh slot at the end. [`Self::remove_id`] only unmaps the ID (bytes may remain in the slab).
#[derive(Debug)]
pub struct SlabStorage {
    data: Vec<u8>,
    vector_len: usize,
    num_slots: usize,
    data_id_to_slot: HashMap<usize, usize>,
    /// Reserved for future reuse; not used for allocation while matching Vec golden behavior.
    #[allow(dead_code)]
    free_slots: Vec<usize>,
}

impl SlabStorage {
    #[must_use]
    pub fn new(vector_len: usize) -> Self {
        Self {
            data: Vec::new(),
            vector_len,
            num_slots: 0,
            data_id_to_slot: HashMap::new(),
            free_slots: Vec::new(),
        }
    }

    #[must_use]
    pub fn vector_len(&self) -> usize {
        self.vector_len
    }

    #[must_use]
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    #[must_use]
    pub fn contains_id(&self, data_id: usize) -> bool {
        self.data_id_to_slot.contains_key(&data_id)
    }

    #[must_use]
    pub fn slot_for_id_opt(&self, data_id: usize) -> Option<usize> {
        self.data_id_to_slot.get(&data_id).copied()
    }

    pub fn slot_for_id(&self, data_id: usize) -> usize {
        *self
            .data_id_to_slot
            .get(&data_id)
            .unwrap_or_else(|| panic!("Vector with ID {data_id} does not exist"))
    }

    /// Borrows the payload for a mapped `data_id`.
    pub fn get_by_id(&self, data_id: usize) -> &[u8] {
        let slot = *self
            .data_id_to_slot
            .get(&data_id)
            .unwrap_or_else(|| panic!("Vector with ID {data_id} does not exist"));
        self.slot_slice(slot)
    }

    /// Writes `src` into the slot for `data_id`, allocating a slot if needed.
    pub fn insert_by_id(&mut self, data_id: usize, src: &[u8]) {
        assert_eq!(
            src.len(),
            self.vector_len,
            "vector length mismatch: {} != {}",
            src.len(),
            self.vector_len
        );
        let slot = self.ensure_id_exists(data_id);
        self.slot_slice_mut(slot).copy_from_slice(src);
    }

    /// Zeros an existing slot or allocates a new zeroed slot for `data_id`.
    pub fn ensure_zero_by_id(&mut self, data_id: usize) {
        let slot = if let Some(&slot) = self.data_id_to_slot.get(&data_id) {
            self.zero_slot(slot);
            slot
        } else {
            self.append_zero_slot(data_id)
        };
        let _ = slot;
    }

    /// Returns the slot for `data_id`, allocating a zeroed slot if missing.
    pub fn ensure_id_exists(&mut self, data_id: usize) -> usize {
        if let Some(&slot) = self.data_id_to_slot.get(&data_id) {
            slot
        } else {
            self.append_zero_slot(data_id)
        }
    }

    /// Unmaps `data_id` only (slot bytes are retained), matching `VecDataOperater::remove`.
    pub fn remove_id(&mut self, data_id: usize) {
        if self.data_id_to_slot.remove(&data_id).is_none() {
            panic!("Vector with ID {data_id} does not exist");
        }
        // Intentionally do not shrink `data` or reuse slots (golden §6.0).
    }

    /// Remaps `src_id` → `target_id` to the same slot; no payload copy.
    pub fn move_id(&mut self, src_id: usize, target_id: usize) {
        if src_id == target_id {
            return;
        }
        let slot = *self
            .data_id_to_slot
            .get(&src_id)
            .unwrap_or_else(|| panic!("Vector with ID {src_id} does not exist"));
        self.data_id_to_slot.remove(&src_id);
        self.data_id_to_slot.insert(target_id, slot);
    }

    /// Copies payload from `src_id` into `target_id` (allocates `target_id` if needed).
    pub fn copy_id(&mut self, src_id: usize, target_id: usize) {
        if src_id == target_id {
            return;
        }
        let src_slot = *self
            .data_id_to_slot
            .get(&src_id)
            .unwrap_or_else(|| panic!("Vector with ID {src_id} does not exist"));
        let target_slot = self.ensure_id_exists(target_id);
        let len = self.vector_len;
        let src_start = src_slot * len;
        let target_start = target_slot * len;
        let payload = self.data[src_start..src_start + len].to_vec();
        self.data[target_start..target_start + len].copy_from_slice(&payload);
    }

    #[inline]
    pub fn slot_slice(&self, slot: usize) -> &[u8] {
        let start = slot * self.vector_len;
        &self.data[start..start + self.vector_len]
    }

    #[inline]
    pub fn slot_slice_mut(&mut self, slot: usize) -> &mut [u8] {
        let start = slot * self.vector_len;
        &mut self.data[start..start + self.vector_len]
    }

    /// `dest[i] ^= src[i]`; panics if `dest_slot == src_slot`.
    pub fn xor_slots(&mut self, dest_slot: usize, src_slot: usize) {
        if dest_slot == src_slot {
            return;
        }
        let (dest, src) = self.get_pair_mut(dest_slot, src_slot);
        for (d, s) in dest.iter_mut().zip(src) {
            *d ^= s;
        }
    }

    /// `dest ^= a ^ b` in one pass (distinct slots).
    pub fn xor_two_slots(&mut self, dest_slot: usize, a_slot: usize, b_slot: usize) {
        if dest_slot == a_slot {
            self.xor_slots(dest_slot, b_slot);
            return;
        }
        if dest_slot == b_slot {
            self.xor_slots(dest_slot, a_slot);
            return;
        }
        if a_slot == b_slot {
            self.xor_slots(dest_slot, a_slot);
            return;
        }
        let len = self.vector_len;
        let (d_start, a_start, b_start) = (
            dest_slot * len,
            a_slot * len,
            b_slot * len,
        );
        unsafe {
            let ptr = self.data.as_mut_ptr();
            let dst = std::slice::from_raw_parts_mut(ptr.add(d_start), len);
            let a = std::slice::from_raw_parts(ptr.add(a_start), len);
            let b = std::slice::from_raw_parts(ptr.add(b_start), len);
            let mut i = 0;
            while i + 8 <= len {
                let d = u64::from_ne_bytes(dst[i..i + 8].try_into().unwrap());
                let x = u64::from_ne_bytes(a[i..i + 8].try_into().unwrap());
                let y = u64::from_ne_bytes(b[i..i + 8].try_into().unwrap());
                dst[i..i + 8].copy_from_slice(&(d ^ x ^ y).to_ne_bytes());
                i += 8;
            }
            while i < len {
                dst[i] ^= a[i] ^ b[i];
                i += 1;
            }
        }
    }

    /// `dest ^= a ^ b ^ c` in one pass (distinct slots).
    pub fn xor_three_slots(
        &mut self,
        dest_slot: usize,
        a_slot: usize,
        b_slot: usize,
        c_slot: usize,
    ) {
        let slots = [dest_slot, a_slot, b_slot, c_slot];
        if slots.iter().any(|&s| slots.iter().filter(|&&t| t == s).count() > 1) {
            // Overlap or duplicate: fall back to sequential XOR.
            if dest_slot != a_slot {
                self.xor_slots(dest_slot, a_slot);
            }
            if dest_slot != b_slot {
                self.xor_slots(dest_slot, b_slot);
            }
            if dest_slot != c_slot {
                self.xor_slots(dest_slot, c_slot);
            }
            return;
        }
        let len = self.vector_len;
        let (d_start, a_start, b_start, c_start) = (
            dest_slot * len,
            a_slot * len,
            b_slot * len,
            c_slot * len,
        );
        unsafe {
            let ptr = self.data.as_mut_ptr();
            let dst = std::slice::from_raw_parts_mut(ptr.add(d_start), len);
            let a = std::slice::from_raw_parts(ptr.add(a_start), len);
            let b = std::slice::from_raw_parts(ptr.add(b_start), len);
            let c = std::slice::from_raw_parts(ptr.add(c_start), len);
            let mut i = 0;
            while i + 8 <= len {
                let d = u64::from_ne_bytes(dst[i..i + 8].try_into().unwrap());
                let w0 = u64::from_ne_bytes(a[i..i + 8].try_into().unwrap());
                let w1 = u64::from_ne_bytes(b[i..i + 8].try_into().unwrap());
                let w2 = u64::from_ne_bytes(c[i..i + 8].try_into().unwrap());
                dst[i..i + 8].copy_from_slice(&(d ^ w0 ^ w1 ^ w2).to_ne_bytes());
                i += 8;
            }
            while i < len {
                dst[i] ^= a[i] ^ b[i] ^ c[i];
                i += 1;
            }
        }
    }

    /// Mutable `dest` and shared `src` slices for distinct slots.
    pub fn get_pair_mut(&mut self, dest_slot: usize, src_slot: usize) -> (&mut [u8], &[u8]) {
        assert_ne!(dest_slot, src_slot, "dest and src slots must differ");
        assert!(dest_slot < self.num_slots, "dest slot out of range");
        assert!(src_slot < self.num_slots, "src slot out of range");

        let len = self.vector_len;
        let dest_start = dest_slot * len;
        let src_start = src_slot * len;

        // SAFETY: disjoint symbol ranges within one `Vec<u8>`.
        unsafe {
            let ptr = self.data.as_mut_ptr();
            let dest_slice = std::slice::from_raw_parts_mut(ptr.add(dest_start), len);
            let src_slice = std::slice::from_raw_parts(ptr.add(src_start), len);
            (dest_slice, src_slice)
        }
    }

    fn append_zero_slot(&mut self, data_id: usize) -> usize {
        let slot = self.num_slots;
        self.num_slots += 1;
        self.data.resize(self.num_slots * self.vector_len, 0);
        self.data_id_to_slot.insert(data_id, slot);
        slot
    }

    fn zero_slot(&mut self, slot: usize) {
        self.slot_slice_mut(slot).fill(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get_roundtrip() {
        let mut slab = SlabStorage::new(4);
        slab.insert_by_id(0, &[1, 2, 3, 4]);
        assert_eq!(slab.get_by_id(0), &[1, 2, 3, 4]);
    }

    #[test]
    fn remove_unmaps_only() {
        let mut slab = SlabStorage::new(2);
        slab.insert_by_id(0, &[1, 2]);
        slab.remove_id(0);
        assert!(!slab.contains_id(0));
        assert_eq!(slab.num_slots(), 1);
    }

    #[test]
    fn get_pair_mut_xor() {
        let mut slab = SlabStorage::new(3);
        slab.insert_by_id(0, &[1, 2, 3]);
        slab.ensure_id_exists(1);
        slab.slot_slice_mut(1).copy_from_slice(&[4, 5, 6]);
        slab.xor_slots(1, 0);
        assert_eq!(slab.slot_slice(1), &[5, 7, 5]);
    }
}
