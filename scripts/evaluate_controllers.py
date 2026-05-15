"""Evaluate controllers on Sinergym. Run inside Sinergym container:

docker compose --profile simulation run sinergym python /app/scripts/evaluate_controllers.py
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import sinergym

ENV_ID = "Eplus-5zone-hot-continuous-v1"


def evaluate_random(env, episodes: int = 1) -> dict:
    return _run_episodes(env, episodes, lambda _: env.action_space.sample())


def evaluate_fixed(env, episodes: int = 1) -> dict:
    return _run_episodes(env, episodes, lambda _: np.array([21.0, 24.0]))


def evaluate_ppo(env, model_path: str, episodes: int = 1) -> dict:
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    return _run_episodes(env, episodes, lambda obs: model.predict(obs, deterministic=True)[0])


def _run_episodes(env, episodes: int, action_fn) -> dict:
    total_reward, total_steps = 0.0, 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = action_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            total_steps += 1
            done = terminated or truncated
    return {"avg_reward": total_reward / episodes, "avg_steps": total_steps / episodes}


def main() -> None:
    env = gym.make(ENV_ID)

    print("=== Controller Evaluation ===\n")

    print("Random actions (baseline)...")
    r = evaluate_random(env)
    print(f"  Avg reward: {r['avg_reward']:.2f}, Steps: {r['avg_steps']:.0f}\n")

    print("Fixed setpoints (21/24 C)...")
    r = evaluate_fixed(env)
    print(f"  Avg reward: {r['avg_reward']:.2f}, Steps: {r['avg_steps']:.0f}\n")

    try:
        print("PPO agent...")
        r = evaluate_ppo(env, "/app/output/ppo_comfort/best_model")
        print(f"  Avg reward: {r['avg_reward']:.2f}, Steps: {r['avg_steps']:.0f}\n")
    except FileNotFoundError:
        print("  PPO model not found. Train first with train_ppo.py\n")

    env.close()


if __name__ == "__main__":
    main()
