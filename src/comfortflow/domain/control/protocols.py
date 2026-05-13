from __future__ import annotations

from typing import Protocol

from comfortflow.domain.control.entities import BuildingState, HVACSetpoints


class ControlPolicy(Protocol):
    def select_action(self, state: BuildingState) -> HVACSetpoints: ...
