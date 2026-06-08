// Copyright (c) 2025 Shenghao Yang. All rights reserved.
// Licensed under the MIT License. See LICENSE-MIT for details.

use fountain_engine::traits::DataOperator;
use fountain_operators::SlabDataOperator;
use fountain_operators::SimdDataOperator;
use fountain_utility::VecDataOperater;

pub fn make_vec_operator(symbol_size: usize) -> Box<dyn DataOperator> {
    Box::new(VecDataOperater::new(symbol_size))
}

pub fn make_slab_operator(symbol_size: usize) -> Box<dyn DataOperator> {
    Box::new(SlabDataOperator::new(symbol_size))
}

pub fn make_simd_operator(symbol_size: usize) -> Box<dyn DataOperator> {
    Box::new(SimdDataOperator::new(symbol_size))
}
