import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from environment.tic_tac_toe import TicTacToeEnv


def main():
    parser = argparse.ArgumentParser(description="Train a PPO Tic-Tac-Toe agent.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)

    env = Monitor(TicTacToeEnv())

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path="checkpoints",
        name_prefix="tic_tac_toe_ppo",
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        verbose=1,
        tensorboard_log="logs",
        seed=args.seed,
        device="auto",
    )

    print(f"Training for {args.timesteps:,} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        tb_log_name="tic_tac_toe_ppo",
    )
    model.save("models/tic_tac_toe_ppo")
    env.close()
    print("Done. Model saved to models/tic_tac_toe_ppo.zip")


if __name__ == "__main__":
    main()
