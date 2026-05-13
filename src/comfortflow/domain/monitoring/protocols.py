from __future__ import annotations

from typing import Protocol

from comfortflow.domain.monitoring.entities import DriftScore


class DriftDetector(Protocol):
    def detect(self, model_name: str) -> DriftScore: ...
