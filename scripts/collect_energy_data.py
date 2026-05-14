"""Collect energy simulation data from Sinergym.

Run inside the Sinergym Docker container:
  docker compose --profile simulation run sinergym python /app/scripts/collect_energy_data.py
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
import pandas as pd
import sinergym

OUTPUT_PATH = "/app/output/energy_data.csv"
NUM_EPISODES = 5
ENV_ID = "Eplus-5zone-hot-continuous-stochastic-v1"


def main() -> None:
    env = gym.make(ENV_ID)
    all_rows = []

    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        terminated = truncated = False
        step = 0

        while not terminated and not truncated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            row = build_row(obs, action, reward, info, episode, step, env)
            all_rows.append(row)
            step += 1

        print(f"Episode {episode + 1}/{NUM_EPISODES}: {step} steps")

    env.close()

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df):,} rows to {OUTPUT_PATH}")


def build_row(obs, action, reward, info, episode, step, env) -> dict:
    obs_names = env.observation_space.get_attr("names") if hasattr(env.observation_space, "get_attr") else []
    row = {"episode": episode, "step": step, "reward": reward}

    for i, val in enumerate(obs):
        name = obs_names[i] if i < len(obs_names) else f"obs_{i}"
        row[name] = val

    for i, val in enumerate(action):
        row[f"action_{i}"] = val

    if isinstance(info, dict):
        for key, val in info.items():
            if isinstance(val, (int, float)):
                row[f"info_{key}"] = val

    return row


if __name__ == "__main__":
    main()
