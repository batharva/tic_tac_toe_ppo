import streamlit as st
import numpy as np
from pathlib import Path
from sb3_contrib import MaskablePPO

from environment.tic_tac_toe import TicTacToeEnv

st.set_page_config(
    page_title="PPO Tic-Tac-Toe",
    page_icon="🎮",
    layout="centered",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "tic_tac_toe_ppo.zip"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return MaskablePPO.load(str(MODEL_PATH), device="cpu")


def check_winner(board, player):
    return any(
        all(board[i] == player for i in line)
        for line in TicTacToeEnv.WIN_LINES
    )


def new_game():
    st.session_state.board = np.zeros(9, dtype=np.int8)
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.message = "AI (X) starts. You play O."
    st.session_state.started = True

    # AI starts the game.
    make_ai_move()


def make_ai_move():
    model = load_model()
    if model is None:
        st.session_state.message = (
            "Trained model not found. Train locally and commit "
            "models/tic_tac_toe_ppo.zip to your GitHub repository."
        )
        st.session_state.game_over = True
        return

    board = st.session_state.board
    mask = board == 0

    if not mask.any():
        st.session_state.game_over = True
        st.session_state.winner = "draw"
        st.session_state.message = "It's a draw."
        return

    action, _ = model.predict(
        board.copy(),
        deterministic=True,
        action_masks=mask,
    )
    action = int(action)
    if not mask[action]:
        st.session_state.game_over = True
        st.session_state.message = "AI returned an invalid move."
        return

    board[action] = 1

    if check_winner(board, 1):
        st.session_state.game_over = True
        st.session_state.winner = "ai"
        st.session_state.message = "PPO wins! Try again."
    elif not (board == 0).any():
        st.session_state.game_over = True
        st.session_state.winner = "draw"
        st.session_state.message = "It's a draw."


def human_move(cell):
    if st.session_state.game_over or st.session_state.board[cell] != 0:
        return

    st.session_state.board[cell] = -1

    if check_winner(st.session_state.board, -1):
        st.session_state.game_over = True
        st.session_state.winner = "human"
        st.session_state.message = "You win! You defeated the PPO agent."
        return

    if not (st.session_state.board == 0).any():
        st.session_state.game_over = True
        st.session_state.winner = "draw"
        st.session_state.message = "It's a draw."
        return

    make_ai_move()


st.title("🎮 PPO Tic-Tac-Toe")
st.caption("A reinforcement-learning agent trained with Proximal Policy Optimization.")

model = load_model()

if model is None:
    st.warning(
        "The trained model file is missing. Run "
        "`python -m training.train` locally, then add "
        "`models/tic_tac_toe_ppo.zip` to your GitHub repository."
    )

if "started" not in st.session_state:
    st.session_state.started = False
    st.session_state.board = np.zeros(9, dtype=np.int8)
    st.session_state.game_over = True
    st.session_state.winner = None
    st.session_state.message = "Click New Game to begin."

col1, col2 = st.columns(2)
with col1:
    st.markdown("**🔵 AI Agent — X**")
with col2:
    st.markdown("**🟠 You — O**")

if st.button("＋ New Game", use_container_width=True, type="primary"):
    new_game()

st.info(st.session_state.message)

symbols = {1: "❌", -1: "⭕", 0: "　"}
for row in range(3):
    cols = st.columns(3, gap="small")
    for col in range(3):
        idx = row * 3 + col
        value = int(st.session_state.board[idx])
        label = symbols[value] if value else " "
        disabled = (
            not st.session_state.started
            or st.session_state.game_over
            or value != 0
        )
        with cols[col]:
            if st.button(
                label,
                key=f"cell_{idx}",
                use_container_width=True,
                disabled=disabled,
            ):
                human_move(idx)
                st.rerun()

if st.session_state.game_over and st.session_state.winner:
    if st.session_state.winner == "human":
        st.success("Congratulations! You won.")
    elif st.session_state.winner == "ai":
        st.error("The PPO agent won this round.")
    else:
        st.info("Draw game.")

with st.expander("About this project"):
    st.write(
        "The agent uses MaskablePPO from SB3-Contrib. "
        "The observation is a 9-element board vector: 1 = X, "
        "-1 = O, and 0 = empty. The action mask prevents the "
        "agent from selecting occupied cells."
    )
