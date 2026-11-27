"""Rules for exchanging adjacent circuit operations."""

from qmiep.rewriting.switch.switch_diagonal_through_r_gate import (
    switch_diagonal_through_r_gate_rule,
)
from qmiep.rewriting.switch.switch_multiplexed_permutation_through_r_gate import (
    switch_multiplexed_permutation_through_r_gate_rule,
)
from qmiep.rewriting.switch.switch_multiplexed_permutation_through_ucg_rz import (
    switch_multiplexed_permutation_through_ucg_rz_rule,
)

__all__ = [
    "switch_diagonal_through_r_gate_rule",
    "switch_multiplexed_permutation_through_r_gate_rule",
    "switch_multiplexed_permutation_through_ucg_rz_rule",
]
