"""Migrate PD-modulus corrections through UCG(Rz) into its next unitary."""

from typing import Literal

import cirq

from qmiep.gates import UniformlyControlledRz
from qmiep.rewriting.migration._common import _pd_migration
from qmiep.rust import PDMode


@cirq.transformer
def pd_migration_through_ucg_rule(
    circuit: cirq.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
    direction: Literal["left", "right"] = "right",
    mode: Literal["analytic", "heuristic"] = "heuristic",
    kind: PDMode = "pd",
) -> cirq.Circuit:
    r"""Rewrite ``U -> F(Rz) -> V`` as ``U_PD -> F'(Rz) -> V'``.

    Fix tensor order |q0>|q1 ... qk>. F is UniformlyControlledRz targeting
    q0; both outer unitaries act on all selectors (q1, ..., qk).
    From U = P D U_PD, define D' = P D P^dagger. Circuit-time order is
    U_PD -> P -> D' -> F -> V. Since F = D_F direct-sum D_F^dagger is
    diagonal, D' commutes through F unchanged. Then use the existing UCG
    permutation switch and merge P -> D' -> V:

    .. math::

        F' &= (I\otimes P^\dagger)F(I\otimes P),\\
        V' &= VD'P=VPD,\\
        (I\otimes V)F(I\otimes U)
           &= (I\otimes V')F'(I\otimes U_{\rm PD}).

    Exactly three top-level operations are required; their tags are preserved
    and ignored tags are rejected. Synthesis uses F's selector order even if
    the outer operations use another order. X becomes X_PD; V' retains V's
    label as an unresolved unitary. F's label, tolerance, and CX boundary are
    preserved. Supported sizes follow PD synthesis. The input is not modified.

    With ``direction="left"``, synthesize V.T and transpose its factorization
    to obtain V = V_PD P_l D_l. Return U' -> F' -> V_PD, where

    .. math::

        F'=(I\otimes P_l)F(I\otimes P_l^\dagger),\qquad U'=P_lD_lU.

    A one-selector source is resolved exactly with identity corrections.
    ``mode`` selects the PD synthesis strategy, including recursive calls.
    ``kind="pd_further_D_optimized"`` uses improved D/PD recursion for the source.
    """
    return _pd_migration(
        circuit,
        rotation_type=UniformlyControlledRz,
        context=context,
        direction=direction,
        mode=mode,
        kind=kind,
    )


__all__ = ["pd_migration_through_ucg_rule"]
