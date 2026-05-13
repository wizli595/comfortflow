from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DriftScore:
    psi_value: float
    is_drifted: bool

    @staticmethod
    def from_psi(psi: float, threshold: float = 0.2) -> DriftScore:
        return DriftScore(psi_value=psi, is_drifted=psi > threshold)
