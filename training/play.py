from sb3_contrib import MaskablePPO
from environment.tic_tac_toe import TicTacToeEnv


def main():
    model = MaskablePPO.load("models/tic_tac_toe_ppo")
    env = TicTacToeEnv()
    obs, _ = env.reset()
    done = False

    print("PPO agent is X. Random opponent is O.")
    while not done:
        action, _ = model.predict(
            obs, deterministic=True, action_masks=env.action_masks()
        )
        obs, reward, terminated, truncated, info = env.step(int(action))
        env.render()
        done = terminated or truncated

    print(f"Game result: {info['winner']} | Reward: {reward}")
    env.close()


if __name__ == "__main__":
    main()
