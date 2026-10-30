import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TicTacToeEnv(gym.Env):
    """Tic-Tac-Toe: the learning agent is X and a random opponent is O.

    One environment step consists of the agent's move and, if the game
    has not ended, the random opponent's response.
    """

    metadata = {"render_modes": ["human"]}

    WIN_LINES = (
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6),
    )

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(9,), dtype=np.int8
        )
        self.action_space = spaces.Discrete(9)
        self.agent_symbol = 1
        self.opponent_symbol = -1
        self.board = np.zeros(9, dtype=np.int8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(9, dtype=np.int8)
        return self.board.copy(), {}

    def action_masks(self):
        """True means the corresponding action is currently legal."""
        return self.board == 0

    def get_legal_actions(self):
        return np.flatnonzero(self.action_masks())

    def check_winner(self, player):
        return any(
            all(self.board[position] == player for position in line)
            for line in self.WIN_LINES
        )

    def check_draw(self):
        return bool(np.all(self.board != 0))

    def _result(self):
        if self.check_winner(self.agent_symbol):
            return "agent"
        if self.check_winner(self.opponent_symbol):
            return "opponent"
        if self.check_draw():
            return "draw"
        return None

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be an integer from 0 to 8; got {action!r}")

        action = int(action)

        # MaskablePPO should never select an illegal move. This is a safe
        # terminal fallback in case the environment is called incorrectly.
        if self.board[action] != 0:
            return self.board.copy(), -1.0, True, False, {
                "winner": "opponent",
                "invalid_action": True,
            }

        self.board[action] = self.agent_symbol

        result = self._result()
        if result == "agent":
            return self.board.copy(), 1.0, True, False, {"winner": "agent"}

        if result == "draw":
            return self.board.copy(), 0.5, True, False, {"winner": "draw"}

        # Random opponent selects from legal cells.
        legal = self.get_legal_actions()
        if len(legal):
            opponent_action = int(self.np_random.choice(legal))
            self.board[opponent_action] = self.opponent_symbol

        result = self._result()
        if result == "opponent":
            return self.board.copy(), -1.0, True, False, {"winner": "opponent"}

        if result == "draw":
            return self.board.copy(), 0.5, True, False, {"winner": "draw"}

        return self.board.copy(), 0.0, False, False, {"winner": None}

    def render(self):
        symbols = {1: "X", -1: "O", 0: " "}
        print()
        for row in range(3):
            cells = self.board[row * 3:(row + 1) * 3]
            print(" " + " | ".join(symbols[int(cell)] for cell in cells))
            if row < 2:
                print("---+---+---")
        print()

    def close(self):
        pass
