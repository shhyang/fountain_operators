// Copyright (c) 2025 Shenghao Yang. All rights reserved.
// Licensed under the MIT License. See LICENSE-MIT for details.

//! NEON GF(256) kernels using per-call nibble LUTs from session [`GF256`] tables.

use super::portable;
use super::GfBlockKernel;
use fountain_engine::algebra::finite_field::GF256;

/// NEON-backed kernel; selected at operator construction when `feature = "simd"` and CPU supports NEON.
#[derive(Debug, Clone, Copy, Default)]
pub struct NeonKernel;

pub static NEON_KERNEL: NeonKernel = NeonKernel;

const CHUNK: usize = 16;

/// Build 16-byte table registers for `vqtbl1q_u8` (low/high nibble of each byte).
#[inline]
fn mul_shuffle_tables(gf: &GF256, scalar: u8) -> ([u8; CHUNK], [u8; CHUNK]) {
    let mut low = [0u8; CHUNK];
    let mut hi = [0u8; CHUNK];
    for j in 0..16 {
        low[j] = gf.mul_lookup(scalar, j as u8);
        hi[j] = gf.mul_lookup(scalar, (j as u8) << 4);
    }
    (low, hi)
}

#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
#[allow(unsafe_op_in_unsafe_fn)]
mod arm {
    use super::*;

    #[target_feature(enable = "neon")]
    pub unsafe fn xor_neon(dst: &mut [u8], src: &[u8]) {
        #[cfg(target_arch = "aarch64")]
        use std::arch::aarch64::*;
        #[cfg(target_arch = "arm")]
        use std::arch::arm::*;

        debug_assert_eq!(dst.len(), src.len());
        let len = dst.len();
        let mut i = 0;
        while i + CHUNK <= len {
            let d = vld1q_u8(dst.as_ptr().add(i));
            let s = vld1q_u8(src.as_ptr().add(i));
            vst1q_u8(dst.as_mut_ptr().add(i), veorq_u8(d, s));
            i += CHUNK;
        }
        while i < len {
            *dst.get_unchecked_mut(i) ^= *src.get_unchecked(i);
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn mul_scalar_neon(
        dst: &mut [u8],
        low: &[u8; CHUNK],
        hi: &[u8; CHUNK],
        gf: &GF256,
        scalar: u8,
    ) {
        #[cfg(target_arch = "aarch64")]
        use std::arch::aarch64::*;
        #[cfg(target_arch = "arm")]
        use std::arch::arm::*;

        let mask = vdupq_n_u8(0x0F);
        let low_table = vld1q_u8(low.as_ptr());
        let hi_table = vld1q_u8(hi.as_ptr());

        let len = dst.len();
        let ptr = dst.as_mut_ptr();
        let mut i = 0;
        while i + CHUNK <= len {
            let self_vec = vld1q_u8(ptr.add(i));
            let low_n = vandq_u8(self_vec, mask);
            let low_result = vqtbl1q_u8(low_table, low_n);
            let high_n = vshrq_n_u8(self_vec, 4);
            let high_n = vandq_u8(high_n, mask);
            let high_result = vqtbl1q_u8(hi_table, high_n);
            vst1q_u8(ptr.add(i), veorq_u8(high_result, low_result));
            i += CHUNK;
        }
        while i < len {
            let b = *ptr.add(i);
            *ptr.add(i) = gf.mul_lookup(scalar, b);
            i += 1;
        }
    }

    #[target_feature(enable = "neon")]
    pub unsafe fn mul_add_neon(
        dst: &mut [u8],
        src: &[u8],
        low: &[u8; CHUNK],
        hi: &[u8; CHUNK],
        gf: &GF256,
        scalar: u8,
    ) {
        #[cfg(target_arch = "aarch64")]
        use std::arch::aarch64::*;
        #[cfg(target_arch = "arm")]
        use std::arch::arm::*;

        debug_assert_eq!(dst.len(), src.len());
        let mask = vdupq_n_u8(0x0F);
        let low_table = vld1q_u8(low.as_ptr());
        let hi_table = vld1q_u8(hi.as_ptr());

        let len = dst.len();
        let d_ptr = dst.as_mut_ptr();
        let s_ptr = src.as_ptr();
        let mut i = 0;
        while i + CHUNK <= len {
            let other_vec = vld1q_u8(s_ptr.add(i));
            let low_n = vandq_u8(other_vec, mask);
            let low_result = vqtbl1q_u8(low_table, low_n);
            let high_n = vshrq_n_u8(other_vec, 4);
            let high_n = vandq_u8(high_n, mask);
            let high_result = vqtbl1q_u8(hi_table, high_n);
            let scaled = veorq_u8(high_result, low_result);

            let self_vec = vld1q_u8(d_ptr.add(i));
            vst1q_u8(d_ptr.add(i), veorq_u8(self_vec, scaled));
            i += CHUNK;
        }
        while i < len {
            *d_ptr.add(i) ^= gf.mul_lookup(scalar, *s_ptr.add(i));
            i += 1;
        }
    }
}

impl GfBlockKernel for NeonKernel {
    fn xor_inplace(&self, dst: &mut [u8], src: &[u8]) {
        assert_eq!(dst.len(), src.len());
        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
        {
            if dst.len() >= CHUNK {
                unsafe {
                    arm::xor_neon(dst, src);
                }
                return;
            }
        }
        portable::PORTABLE_KERNEL.xor_inplace(dst, src);
    }

    fn mul_add_inplace(&self, gf: &GF256, dst: &mut [u8], src: &[u8], scalar: u8) {
        assert_eq!(dst.len(), src.len());
        match scalar {
            0 => {}
            1 => self.xor_inplace(dst, src),
            _ => {
                let (low, hi) = mul_shuffle_tables(gf, scalar);
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                {
                    if dst.len() >= CHUNK {
                        unsafe {
                            arm::mul_add_neon(dst, src, &low, &hi, gf, scalar);
                        }
                        return;
                    }
                }
                GfBlockKernel::mul_add_inplace(
                    &portable::PORTABLE_KERNEL,
                    gf,
                    dst,
                    src,
                    scalar,
                );
            }
        }
    }

    fn mul_scalar_inplace(&self, gf: &GF256, dst: &mut [u8], scalar: u8) {
        match scalar {
            0 => dst.fill(0),
            1 => {}
            _ => {
                let (low, hi) = mul_shuffle_tables(gf, scalar);
                #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
                {
                    if dst.len() >= CHUNK {
                        unsafe {
                            arm::mul_scalar_neon(dst, &low, &hi, gf, scalar);
                        }
                        return;
                    }
                }
                GfBlockKernel::mul_scalar_inplace(
                    &portable::PORTABLE_KERNEL,
                    gf,
                    dst,
                    scalar,
                );
            }
        }
    }

    fn mul_alpha_inplace(&self, gf: &GF256, dst: &mut [u8]) {
        self.mul_scalar_inplace(gf, dst, gf.primitive_element());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gf_kernels::{default_kernel, GfBlockKernel};

    #[test]
    fn neon_matches_portable_gf256() {
        let gf = GF256::new_with_primitive_polynomial(0x11D);
        let portable = default_kernel();
        let neon = &NEON_KERNEL;

        for len in [1usize, 3, 15, 16, 64, 1024] {
            let mut a = vec![0u8; len];
            let b = vec![7u8; len];
            let mut rng = 1u64;
            for x in a.iter_mut() {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                *x = (rng >> 33) as u8;
            }

            let mut p = a.clone();
            let mut n = a.clone();
            portable.xor_inplace(&mut p, &b);
            neon.xor_inplace(&mut n, &b);
            assert_eq!(p, n, "xor len={len}");

            let scalar = 0x9Du8;
            p = a.clone();
            n = a.clone();
            portable.mul_scalar_inplace(&gf, &mut p, scalar);
            neon.mul_scalar_inplace(&gf, &mut n, scalar);
            assert_eq!(p, n, "mul_scalar len={len}");

            p = a.clone();
            n = a.clone();
            portable.mul_add_inplace(&gf, &mut p, &b, scalar);
            neon.mul_add_inplace(&gf, &mut n, &b, scalar);
            assert_eq!(p, n, "mul_add len={len}");

            p = a.clone();
            n = a.clone();
            portable.mul_alpha_inplace(&gf, &mut p);
            neon.mul_alpha_inplace(&gf, &mut n);
            assert_eq!(p, n, "mul_alpha len={len}");
        }
    }
}
