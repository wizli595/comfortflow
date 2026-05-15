"""PID controller for zone temperature."""

from __future__ import annotations

from comfortflow.domain.control.entities import BuildingState, HVACSetpoints


class PIDController:
    def __init__(self, target: float = 23.0, kp: float = 1.0, ki: float = 0.1, kd: float = 0.05) -> None:
        self._target = target
        self._kp = kp
        self._ki = ki
        self._kd = kd
        self._cumulative_error = 0.0
        self._prev_error = 0.0

    def select_action(self, state: BuildingState) -> HVACSetpoints:
        error = self._target - state.indoor_temperature_celsius
        self._cumulative_error += error
        derivative = error - self._prev_error
        self._prev_error = error

        adjustment = self._kp * error + self._ki * self._cumulative_error + self._kd * derivative
        heating = max(15.0, min(22.5, 21.0 + max(adjustment, 0)))
        cooling = max(22.5, min(30.0, 24.0 + min(adjustment, 0)))

        return HVACSetpoints(heating_celsius=heating, cooling_celsius=cooling)
