"""Migrate PD-modulus corrections through an R gate into its next unitary."""

from typing import Literal

import cirq

from qmiep.gates import RGate
from qmiep.rewriting.migration._common import _pd_migration


@cirq.transformer
def pd_migration_through_r_rule(
    circuit: cirq.AbstractCircuit,
    *,
    context: cirq.TransformerContext | None = None,
    direction: Literal["left", "right"] = "right",
    mode: Literal["analytic", "heuristic"] = "heuristic",
) -> cirq.Circuit:
    r"""Rewrite ``U -> R -> V`` as ``U_PD -> R' -> V'``.

    In tensor order |q0>|q1 ... qk>, R targets q0 and both outer unitaries
    act on all selectors (q1, ..., qk). PD synthesis gives U = P D U_PD.
    Set D' = P D P^dagger to obtain circuit order U_PD -> P -> D' -> R -> V.
    Switch D' through R, then P, and merge P -> D' -> V:

    .. math::

        R' &= (I\otimes P^\dagger)R(I\otimes P),\\
        V' &= VD'P=VPD,\\
        (I\otimes V)R(I\otimes U)
           &= (I\otimes V')R'(I\otimes U_{\rm PD}).

    Exactly three top-level operations are required; their tags are preserved
    and ignored tags are rejected. Synthesis uses R's selector order even if
    the outer operations use another order. X becomes X_PD; V' retains V's
    label as an unresolved unitary. Supported sizes follow PD synthesis.
    With ``direction="left"``, synthesize V.T and transpose its factorization
    to obtain V = V_PD P_l D_l. Return U' -> R' -> V_PD, where

    .. math::

        R'=(I\otimes P_l)R(I\otimes P_l^\dagger),\qquad U'=P_lD_lU.

    A one-selector source is resolved exactly with identity corrections.
    ``mode`` selects the PD synthesis strategy, including recursive calls.
    The input circuit is never modified.
    """
    return _pd_migration(
        circuit,
        rotation_type=RGate,
        context=context,
        direction=direction,
        mode=mode,
    )


__all__ = ["pd_migration_through_r_rule"]
