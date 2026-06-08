// Copyright (c) 2025 Shenghao Yang. All rights reserved.
// Licensed under the MIT License. See LICENSE-MIT for details.

use super::GfBlockKernel;
use fountain_engine::algebra::finite_field::GF256;

/// Scalar portable implementation using [`GF256`] lookup tables from `fountain_engine`.
#[derive(Debug, Clone, Copy, Default)]
pub struct PortableKernel;

pub static PORTABLE_KERNEL: PortableKernel = PortableKernel;

impl GfBlockKernel for PortableKernel {
    fn xor_inplace(&self, dst: &mut [u8], src: &[u8]) {
        assert_eq!(dst.len(), src.len());
        for (d, s) in dst.iter_mut().zip(src) {
            *d ^= s;
        }
    }

    fn mul_add_inplace(&self, gf: &GF256, dst: &mut [u8], src: &[u8], scalar: u8) {
        assert_eq!(dst.len(), src.len());
        match scalar {
            0 => {}
            1 => self.xor_inplace(dst, src),
            _ => {
                for (d, s) in dst.iter_mut().zip(src) {
                    *d ^= gf.mul_lookup(scalar, *s);
                }
            }
        }
    }

    fn mul_scalar_inplace(&self, gf: &GF256, dst: &mut [u8], scalar: u8) {
        match scalar {
            0 => {
                dst.fill(0);
            }
            1 => {}
            _ => {
                for b in dst.iter_mut() {
                    *b = gf.mul_lookup(*b, scalar);
                }
            }
        }
    }

    fn mul_alpha_inplace(&self, gf: &GF256, dst: &mut [u8]) {
        for b in dst.iter_mut() {
            *b = gf.mul_alpha(*b);
        }
    }
}
