"""Rule-based controller — ASHRAE 55 fixed setpoints."""

from __future__ import annotations

from comfortflow.domain.control.entities import BuildingState, HVACSetpoints


class RuleBasedController:
    def select_action(self, state: BuildingState) -> HVACSetpoints:
        return HVACSetpoints(heating_celsius=21.0, cooling_celsius=25.0)
