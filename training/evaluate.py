import argparse
import numpy as np
from sb3_contrib import MaskablePPO

from environment.tic_tac_toe import TicTacToeEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--model", default="models/tic_tac_toe_ppo")
    args = parser.parse_args()

    model = MaskablePPO.load(args.model)
    env = TicTacToeEnv()
    results = {"agent": 0, "opponent": 0, "draw": 0}

    for _ in range(args.games):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(
                obs,
                deterministic=True,
                action_masks=env.action_masks(),
            )
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        results[info["winner"]] += 1

    print(f"\nEvaluation over {args.games} games")
    for key, count in results.items():
        print(f"{key.title():10s}: {count:4d} ({100 * count / args.games:.1f}%)")
    env.close()


if __name__ == "__main__":
    main()
