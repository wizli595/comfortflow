"""Reward function for RL agent."""

from __future__ import annotations


def compute_reward(
    energy_kwh: float,
    pmv: float,
    alpha: float = 0.4,
    beta: float = 0.5,
    gamma: float = 0.1,
) -> float:
    energy_penalty = -alpha * energy_kwh
    comfort_penalty = -beta * abs(pmv)
    comfort_bonus = gamma if abs(pmv) < 0.5 else 0.0
    return energy_penalty + comfort_penalty + comfort_bonus
