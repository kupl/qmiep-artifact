"""Circuit rewriting rules."""

from qmiep.rewriting.block_zxz import block_zxz_operations, block_zxz_rule
from qmiep.rewriting.hadamard_cx import hadamard_cx_to_cz_rule
from qmiep.rewriting.merge_central_multiplexor import (
    merge_central_multiplexor_rule,
)
from qmiep.rewriting.migration import (
    pd_migration_through_r_rule,
    pd_migration_through_ucg_rule,
)
from qmiep.rewriting.multiplexed_unitary_decomposition_by_spectrum import (
    multiplexed_unitary_decomposition_by_spectrum_rule,
)
from qmiep.rewriting.relabel_gauge_unitaries import (
    relabel_gauge_unitaries_rule,
)
from qmiep.rewriting.switch import (
    switch_diagonal_through_r_gate_rule,
    switch_multiplexed_permutation_through_r_gate_rule,
    switch_multiplexed_permutation_through_ucg_rz_rule,
)

__all__ = [
    "block_zxz_operations",
    "block_zxz_rule",
    "multiplexed_unitary_decomposition_by_spectrum_rule",
    "hadamard_cx_to_cz_rule",
    "merge_central_multiplexor_rule",
    "pd_migration_through_r_rule",
    "pd_migration_through_ucg_rule",
    "relabel_gauge_unitaries_rule",
    "switch_diagonal_through_r_gate_rule",
    "switch_multiplexed_permutation_through_r_gate_rule",
    "switch_multiplexed_permutation_through_ucg_rz_rule",
]
