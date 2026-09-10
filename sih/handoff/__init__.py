"""
sih.handoff
-----------
Seamless GNSS <-> INS Handoff State Machine for Intelligent Dead Reckoning.
Governs degradation, blackout entry, and smooth zero-jump reacquisition.
"""

from sih.handoff.integrity import (
    compute_position_nis,
    check_kinematic_feasibility,
    evaluate_signal_quality,
)
from sih.handoff.reconciliation import HermiteReconciler
from sih.handoff.manager import (
    SeamlessGNSSHandoffManager,
    HandoffState,
    HandoffConfig,
)

__all__ = [
    "compute_position_nis",
    "check_kinematic_feasibility",
    "evaluate_signal_quality",
    "HermiteReconciler",
    "SeamlessGNSSHandoffManager",
    "HandoffState",
    "HandoffConfig",
]
