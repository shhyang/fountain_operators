// Copyright (c) 2025 Shenghao Yang. All rights reserved.
// Licensed under the MIT License. See LICENSE-MIT for details.

use fountain_engine::traits::DataOperator;
use fountain_engine::types::Operation;

/// Replays a recorded operation log on a [`DataOperator`].
///
/// The caller must align the finite field (e.g. via [`DataOperator::config_finite_field`])
/// and insert message vectors before replay when using a delayed encoder log.
pub fn replay_operations(operator: &mut dyn DataOperator, ops: &[Operation]) {
    for op in ops {
        operator.execute(op);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fountain_engine::types::Operation;
    use fountain_utility::VecDataOperater;

    #[test]
    fn replay_matches_direct_execute() {
        let len = 8;
        let mut direct = VecDataOperater::new_with_gf256(len, 0x11D);
        let mut replayed = VecDataOperater::new_with_gf256(len, 0x11D);

        direct.insert_vector(&[1, 2, 3, 4, 5, 6, 7, 8], 0);
        replayed.insert_vector(&[1, 2, 3, 4, 5, 6, 7, 8], 0);

        let ops = vec![
            Operation::EnsureZero {
                list_id: vec![1, 2],
            },
            Operation::MulAdd {
                src_id: 0,
                scalar: 2,
                target_id: 1,
            },
            Operation::AddToVector {
                list_id: vec![0],
                target_id: 2,
            },
        ];

        for op in &ops {
            direct.execute(op);
        }
        replay_operations(&mut replayed, &ops);

        assert_eq!(direct.get_vector(1), replayed.get_vector(1));
        assert_eq!(direct.get_vector(2), replayed.get_vector(2));
    }
}
