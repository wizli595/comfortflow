"""Collect energy simulation data from Sinergym.

Run inside Sinergym container:
  docker compose --profile simulation run sinergym python /app/scripts/collect_energy_data.py
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
import sinergym

OUTPUT_PATH = "/app/output/energy_data.csv"
ENV_ID = "Eplus-5zone-hot-continuous-v1"
NUM_EPISODES = 3


def main() -> None:
    env = gym.make(ENV_ID)
    obs_names = list(env.observation_space.keys()) if hasattr(env.observation_space, "keys") else []
    all_rows = []

    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        terminated = truncated = False
        step = 0

        while not terminated and not truncated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            row = {"episode": episode, "step": step, "reward": reward}
            for i, val in enumerate(obs):
                name = obs_names[i] if i < len(obs_names) else f"obs_{i}"
                row[name] = float(val)
            for i, val in enumerate(action):
                row[f"action_{i}"] = float(val)
            all_rows.append(row)
            step += 1

        print(f"Episode {episode + 1}/{NUM_EPISODES}: {step} steps")

    env.close()
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df):,} rows, {len(df.columns)} cols to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
