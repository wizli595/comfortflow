"""Train PPO agent on Sinergym. Run inside Sinergym container:

docker compose --profile simulation run sinergym python /app/scripts/train_ppo.py
"""

from __future__ import annotations

import gymnasium as gym
import sinergym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

ENV_ID = "Eplus-5zone-hot-continuous-v1"
MODEL_PATH = "/app/output/ppo_comfort"
TIMESTEPS = 200_000


def main() -> None:
    env = gym.make(ENV_ID)
    eval_env = gym.make(ENV_ID)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env, best_model_save_path=MODEL_PATH,
        eval_freq=10_000, n_eval_episodes=1, verbose=1,
    )

    print(f"Training PPO for {TIMESTEPS:,} timesteps...")
    model.learn(total_timesteps=TIMESTEPS, callback=eval_cb)
    model.save(f"{MODEL_PATH}/final_model")

    print(f"Model saved to {MODEL_PATH}")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
