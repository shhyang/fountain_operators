// Copyright (c) 2025 Shenghao Yang. All rights reserved.
// Licensed under the MIT License. See LICENSE-MIT for details.

//! AVX2 GF(256) kernels using per-call nibble LUTs from session [`GF256`] tables.

use super::GfBlockKernel;
use super::portable::PortableKernel;
use fountain_engine::algebra::finite_field::GF256;

/// AVX2-backed kernel; selected at operator construction when `feature = "simd"` and CPU supports AVX2.
#[derive(Debug, Clone, Copy, Default)]
pub struct Avx2Kernel;

pub static AVX2_KERNEL: Avx2Kernel = Avx2Kernel;

const CHUNK: usize = 32;

/// Build 32-byte shuffle tables for `_mm256_shuffle_epi8` (low/high nibble of each byte).
#[inline]
fn mul_shuffle_tables(gf: &GF256, scalar: u8) -> ([u8; CHUNK], [u8; CHUNK]) {
    let mut low = [0u8; CHUNK];
    let mut hi = [0u8; CHUNK];
    for j in 0..16 {
        let lj = gf.mul_lookup(scalar, j as u8);
        let hj = gf.mul_lookup(scalar, (j as u8) << 4);
        low[j] = lj;
        low[j + 16] = lj;
        hi[j] = hj;
        hi[j + 16] = hj;
    }
    (low, hi)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    use super::*;

    #[target_feature(enable = "avx2")]
    pub unsafe fn xor_avx2(dst: &mut [u8], src: &[u8]) {
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;

        debug_assert_eq!(dst.len(), src.len());
        let len = dst.len();
        let mut i = 0;
        while i + CHUNK <= len {
            let d_ptr = dst.as_mut_ptr().add(i);
            let s_ptr = src.as_ptr().add(i);
            let d = _mm256_loadu_si256(d_ptr as *const __m256i);
            let s = _mm256_loadu_si256(s_ptr as *const __m256i);
            _mm256_storeu_si256(d_ptr as *mut __m256i, _mm256_xor_si256(d, s));
            i += CHUNK;
        }
        while i < len {
            *dst.get_unchecked_mut(i) ^= *src.get_unchecked(i);
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn mul_scalar_avx2(
        dst: &mut [u8],
        low: &[u8; CHUNK],
        hi: &[u8; CHUNK],
        gf: &GF256,
        scalar: u8,
    ) {
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;

        let low_mask = _mm256_set1_epi8(0x0F);
        let hi_mask = _mm256_set1_epi8(0xF0u8 as i8);
        let low_table = _mm256_loadu_si256(low.as_ptr() as *const __m256i);
        let hi_table = _mm256_loadu_si256(hi.as_ptr() as *const __m256i);

        let len = dst.len();
        let ptr = dst.as_mut_ptr();
        let mut i = 0;
        while i + CHUNK <= len {
            let self_vec = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
            let low_n = _mm256_and_si256(self_vec, low_mask);
            let low_result = _mm256_shuffle_epi8(low_table, low_n);
            let high_n = _mm256_and_si256(self_vec, hi_mask);
            let high_n = _mm256_srli_epi64(high_n, 4);
            let high_result = _mm256_shuffle_epi8(hi_table, high_n);
            let result = _mm256_xor_si256(high_result, low_result);
            _mm256_storeu_si256(ptr.add(i) as *mut __m256i, result);
            i += CHUNK;
        }
        while i < len {
            let b = *ptr.add(i);
            *ptr.add(i) = gf.mul_lookup(scalar, b);
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn mul_add_avx2(
        dst: &mut [u8],
        src: &[u8],
        low: &[u8; CHUNK],
        hi: &[u8; CHUNK],
        gf: &GF256,
        scalar: u8,
    ) {
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;

        debug_assert_eq!(dst.len(), src.len());
        let low_mask = _mm256_set1_epi8(0x0F);
        let hi_mask = _mm256_set1_epi8(0xF0u8 as i8);
        let low_table = _mm256_loadu_si256(low.as_ptr() as *const __m256i);
        let hi_table = _mm256_loadu_si256(hi.as_ptr() as *const __m256i);

        let len = dst.len();
        let d_ptr = dst.as_mut_ptr();
        let s_ptr = src.as_ptr();
        let mut i = 0;
        while i + CHUNK <= len {
            let other_vec = _mm256_loadu_si256(s_ptr.add(i) as *const __m256i);
            let low_n = _mm256_and_si256(other_vec, low_mask);
            let low_result = _mm256_shuffle_epi8(low_table, low_n);
            let high_n = _mm256_and_si256(other_vec, hi_mask);
            let high_n = _mm256_srli_epi64(high_n, 4);
            let high_result = _mm256_shuffle_epi8(hi_table, high_n);
            let scaled = _mm256_xor_si256(high_result, low_result);

            let self_vec = _mm256_loadu_si256(d_ptr.add(i) as *const __m256i);
            _mm256_storeu_si256(
                d_ptr.add(i) as *mut __m256i,
                _mm256_xor_si256(self_vec, scaled),
            );
            i += CHUNK;
        }
        while i < len {
            *d_ptr.add(i) ^= gf.mul_lookup(scalar, *s_ptr.add(i));
            i += 1;
        }
    }
}

impl GfBlockKernel for Avx2Kernel {
    fn xor_inplace(&self, dst: &mut [u8], src: &[u8]) {
        assert_eq!(dst.len(), src.len());
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if dst.len() >= CHUNK {
                // SAFETY: caller ensures AVX2 via `select_kernel`; intrinsics match `target_feature`.
                unsafe {
                    x86::xor_avx2(dst, src);
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
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    if dst.len() >= CHUNK {
                        unsafe {
                            x86::mul_add_avx2(dst, src, &low, &hi, gf, scalar);
                        }
                        return;
                    }
                }
                portable::PORTABLE_KERNEL.mul_add_inplace(&portable::PORTABLE_KERNEL, gf, dst, src, scalar);
            }
        }
    }

    fn mul_scalar_inplace(&self, gf: &GF256, dst: &mut [u8], scalar: u8) {
        match scalar {
            0 => dst.fill(0),
            1 => {}
            _ => {
                let (low, hi) = mul_shuffle_tables(gf, scalar);
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    if dst.len() >= CHUNK {
                        unsafe {
                            x86::mul_scalar_avx2(dst, &low, &hi, gf, scalar);
                        }
                        return;
                    }
                }
                portable::PORTABLE_KERNEL.mul_scalar_inplace(&portable::PORTABLE_KERNEL, gf, dst, scalar);
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
    fn avx2_matches_portable_gf256() {
        let gf = GF256::new_with_primitive_polynomial(0x11D);
        let portable = default_kernel();
        let avx2 = &AVX2_KERNEL;

        for len in [1usize, 3, 31, 32, 64, 1024] {
            let mut a = vec![0u8; len];
            let b = vec![7u8; len];
            let mut rng = 1u64;
            for x in a.iter_mut() {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                *x = (rng >> 33) as u8;
            }

            let mut p = a.clone();
            let mut s = a.clone();
            portable.xor_inplace(&mut p, &b);
            avx2.xor_inplace(&mut s, &b);
            assert_eq!(p, s, "xor len={len}");

            let scalar = 0x9Du8;
            p = a.clone();
            s = a.clone();
            portable.mul_scalar_inplace(&gf, &mut p, scalar);
            avx2.mul_scalar_inplace(&gf, &mut s, scalar);
            assert_eq!(p, s, "mul_scalar len={len}");

            p = a.clone();
            s = a.clone();
            portable.mul_add_inplace(&gf, &mut p, &b, scalar);
            avx2.mul_add_inplace(&gf, &mut s, &b, scalar);
            assert_eq!(p, s, "mul_add len={len}");

            p = a.clone();
            s = a.clone();
            portable.mul_alpha_inplace(&gf, &mut p);
            avx2.mul_alpha_inplace(&gf, &mut s);
            assert_eq!(p, s, "mul_alpha len={len}");
        }
    }
}
