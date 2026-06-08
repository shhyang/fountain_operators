// Copyright (c) 2025 Shenghao Yang. All rights reserved.
// Licensed under the MIT License. See LICENSE-MIT for details.

//! One-time CPU feature detection for kernel selection (phases 3a/3b).

/// True when AVX2 is available at runtime (x86/x86_64 with `simd` feature only).
#[must_use]
#[allow(dead_code)]
pub fn avx2_available() -> bool {
    #[cfg(all(
        feature = "simd",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    {
        cpufeatures::new!(has_avx2, "avx2");
        return has_avx2::get();
    }
    #[cfg(not(all(
        feature = "simd",
        any(target_arch = "x86", target_arch = "x86_64")
    )))]
    {
        false
    }
}

/// True when NEON is available at runtime (aarch64/arm with `simd` feature only).
#[must_use]
#[allow(dead_code)]
pub fn neon_available() -> bool {
    #[cfg(all(
        feature = "simd",
        any(target_arch = "aarch64", target_arch = "arm")
    ))]
    {
        cpufeatures::new!(has_neon, "neon");
        return has_neon::get();
    }
    #[cfg(not(all(
        feature = "simd",
        any(target_arch = "aarch64", target_arch = "arm")
    )))]
    {
        false
    }
}
