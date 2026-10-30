from pathlib import Path
from threading import Lock

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sb3_contrib import MaskablePPO

from environment.tic_tac_toe import TicTacToeEnv

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
MODEL_PATH = ROOT / "models" / "tic_tac_toe_ppo.zip"

app = FastAPI(title="PPO Tic-Tac-Toe", version="1.0.0")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

game_lock = Lock()
model = None
board = np.zeros(9, dtype=np.int8)
game_over = False
winner = None


class MoveRequest(BaseModel):
    cell: int = Field(ge=0, le=8)


def load_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Trained model not found. Run: python -m training.train",
            )
        model = MaskablePPO.load(str(MODEL_PATH), device="cpu")
    return model


def has_won(cells, player):
    return any(
        all(cells[i] == player for i in line)
        for line in TicTacToeEnv.WIN_LINES
    )


def legal_mask():
    return board == 0


def state_payload(message=""):
    return {
        "board": board.astype(int).tolist(),
        "game_over": game_over,
        "winner": winner,
        "message": message,
        "human_symbol": "O",
        "ai_symbol": "X",
    }


def ai_move():
    global game_over, winner
    current_model = load_model()
    mask = legal_mask()
    if not mask.any():
        game_over, winner = True, "draw"
        return

    action, _ = current_model.predict(
        board.copy(),
        deterministic=True,
        action_masks=mask,
    )
    cell = int(action)
    if not mask[cell]:
        raise RuntimeError("The model selected an illegal action despite masking.")

    board[cell] = 1

    if has_won(board, 1):
        game_over, winner = True, "ai"
    elif not legal_mask().any():
        game_over, winner = True, "draw"


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/state")
def get_state():
    with game_lock:
        return state_payload()


@app.post("/api/new")
def new_game():
    global board, game_over, winner
    with game_lock:
        load_model()
        board = np.zeros(9, dtype=np.int8)
        game_over = False
        winner = None
        ai_move()  # AI (X) starts each game.
        message = "AI starts. Your turn (O)." if not game_over else "New game started."
        return state_payload(message)


@app.post("/api/move")
def make_move(request: MoveRequest):
    global game_over, winner
    with game_lock:
        load_model()
        if game_over:
            raise HTTPException(status_code=400, detail="The game is already over.")
        cell = request.cell
        if board[cell] != 0:
            raise HTTPException(status_code=400, detail="That cell is already occupied.")

        board[cell] = -1  # Human is O.

        if has_won(board, -1):
            game_over, winner = True, "human"
            return state_payload("You win!")
        if not legal_mask().any():
            game_over, winner = True, "draw"
            return state_payload("It's a draw.")

        ai_move()

        if winner == "ai":
            message = "PPO wins. Try again!"
        elif winner == "draw":
            message = "It's a draw."
        else:
            message = "Your turn (O)."
        return state_payload(message)
