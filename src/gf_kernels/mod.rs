// Copyright (c) 2025 Shenghao Yang. All rights reserved.
// Licensed under the MIT License. See LICENSE-MIT for details.

//! GF(256) block kernels for vector operations (portable baseline; SIMD in phase 3).

mod dispatch;
mod portable;

#[cfg(all(
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
mod avx2;

#[cfg(all(
    feature = "simd",
    any(target_arch = "aarch64", target_arch = "arm")
))]
mod neon;

pub use portable::PortableKernel;

#[cfg(all(
    feature = "simd",
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub use avx2::Avx2Kernel;

#[cfg(all(
    feature = "simd",
    any(target_arch = "aarch64", target_arch = "arm")
))]
pub use neon::NeonKernel;

use fountain_engine::algebra::finite_field::GF256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
    Portable,
    Avx2,
    Neon,
}

impl KernelKind {
    #[must_use]
    pub fn is_simd(self) -> bool {
        matches!(self, Self::Avx2 | Self::Neon)
    }
}

/// Slice-oriented GF(256) primitives used by slab/SIMD operators.
pub trait GfBlockKernel: Send + Sync {
    /// `dst[i] ^= src[i]` for all `i`.
    fn xor_inplace(&self, dst: &mut [u8], src: &[u8]);

    /// `dst[i] ^= scalar * src[i]`.
    fn mul_add_inplace(&self, gf: &GF256, dst: &mut [u8], src: &[u8], scalar: u8);

    /// `dst[i] = scalar * dst[i]`.
    fn mul_scalar_inplace(&self, gf: &GF256, dst: &mut [u8], scalar: u8);

    /// `dst[i] = alpha * dst[i]` (primitive element of `gf`).
    fn mul_alpha_inplace(&self, gf: &GF256, dst: &mut [u8]);
}

/// Portable-only kernel (phases 1–2). Never selects SIMD silently.
#[must_use]
pub fn default_kernel() -> &'static PortableKernel {
    &portable::PORTABLE_KERNEL
}

/// Best kernel for this process: AVX2 or NEON when `feature = "simd"` and CPU supports it, else portable.
#[must_use]
pub fn select_kernel() -> &'static dyn GfBlockKernel {
    #[cfg(all(
        feature = "simd",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    {
        if dispatch::avx2_available() {
            return &avx2::AVX2_KERNEL;
        }
    }
    #[cfg(all(
        feature = "simd",
        any(target_arch = "aarch64", target_arch = "arm")
    ))]
    {
        if dispatch::neon_available() {
            return &neon::NEON_KERNEL;
        }
    }
    &portable::PORTABLE_KERNEL
}

/// Reports which kernel will be selected in the current process.
#[must_use]
pub fn selected_kernel_kind() -> KernelKind {
    #[cfg(all(
        feature = "simd",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    {
        if dispatch::avx2_available() {
            return KernelKind::Avx2;
        }
    }
    #[cfg(all(
        feature = "simd",
        any(target_arch = "aarch64", target_arch = "arm")
    ))]
    {
        if dispatch::neon_available() {
            return KernelKind::Neon;
        }
    }
    KernelKind::Portable
}

/// Reports all kernels available for this target build.
#[must_use]
pub fn available_kernel_kinds() -> Vec<KernelKind> {
    if cfg!(all(
        feature = "simd",
        any(target_arch = "x86", target_arch = "x86_64")
    )) {
        vec![KernelKind::Portable, KernelKind::Avx2]
    } else if cfg!(all(
        feature = "simd",
        any(target_arch = "aarch64", target_arch = "arm")
    )) {
        vec![KernelKind::Portable, KernelKind::Neon]
    } else {
        vec![KernelKind::Portable]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fountain_engine::GF256;

    #[test]
    fn portable_matches_gf256_reference() {
        let gf = GF256::new_with_primitive_polynomial(0x11D);
        let kernel = default_kernel();
        let mut a = vec![1u8, 2, 3, 200];
        let b = vec![4u8, 5, 6, 7];
        let mut expect = a.clone();

        for i in 0..expect.len() {
            expect[i] ^= b[i];
        }
        kernel.xor_inplace(&mut a, &b);
        assert_eq!(a, expect);

        let scalar = 7u8;
        let mut c = vec![1u8, 2, 3, 4];
        expect = c.clone();
        for x in &mut expect {
            *x = gf.mul_lookup(*x, scalar);
        }
        kernel.mul_scalar_inplace(&gf, &mut c, scalar);
        assert_eq!(c, expect);

        let mut d = vec![10u8, 20, 30, 40];
        let e = vec![1u8, 2, 3, 4];
        expect = d.clone();
        for i in 0..expect.len() {
            expect[i] ^= gf.mul_lookup(scalar, e[i]);
        }
        kernel.mul_add_inplace(&gf, &mut d, &e, scalar);
        assert_eq!(d, expect);

        let mut f = vec![1u8, 128, 255, 0];
        expect = f.clone();
        for x in &mut expect {
            *x = gf.mul_alpha(*x);
        }
        kernel.mul_alpha_inplace(&gf, &mut f);
        assert_eq!(f, expect);
    }

    #[test]
    fn select_kernel_matches_portable_behavior() {
        let gf = GF256::new_with_primitive_polynomial(0x11D);
        let selected = select_kernel();
        let portable = default_kernel();
        let src = vec![1u8, 2, 3, 200, 17, 31, 32, 33];

        let mut via_selected = src.clone();
        let mut via_portable = src.clone();
        selected.xor_inplace(&mut via_selected, &src);
        portable.xor_inplace(&mut via_portable, &src);
        assert_eq!(via_selected, via_portable);

        let scalar = 0x9Du8;
        selected.mul_scalar_inplace(&gf, &mut via_selected, scalar);
        portable.mul_scalar_inplace(&gf, &mut via_portable, scalar);
        assert_eq!(via_selected, via_portable);

        #[cfg(all(
            feature = "simd",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        if dispatch::avx2_available() {
            let avx2 = &avx2::AVX2_KERNEL;
            let mut via_avx2 = src.clone();
            avx2.xor_inplace(&mut via_avx2, &src);
            assert_eq!(via_avx2, via_portable);
        }

        #[cfg(all(
            feature = "simd",
            any(target_arch = "aarch64", target_arch = "arm")
        ))]
        if dispatch::neon_available() {
            let neon = &neon::NEON_KERNEL;
            let mut via_neon = src.clone();
            neon.xor_inplace(&mut via_neon, &src);
            assert_eq!(via_neon, via_portable);
        }
    }

    #[test]
    fn selected_kernel_is_reported_in_available_set() {
        let selected = selected_kernel_kind();
        assert!(available_kernel_kinds().contains(&selected));
    }
}
