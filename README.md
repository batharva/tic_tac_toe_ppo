# PPO Tic-Tac-Toe

A small web-based Tic-Tac-Toe game. A PPO agent trained with `sb3-contrib` plays
as **X** against a human player (**O**). The agent starts each game.

## Requirements

- Python 3.10 or newer
- pip
- Internet connection for the first dependency installation

## 1. Create and activate a virtual environment

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install packages

Run from the project root (the folder containing `requirements.txt`):

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Train the PPO agent

```bash
python -m training.train
```

By default, training runs for 200,000 timesteps. You can change this:

```bash
python -m training.train --timesteps 500000
```

The final model is saved as:

```text
models/tic_tac_toe_ppo.zip
```

Training checkpoints are saved in `checkpoints/`, and TensorBoard logs in `logs/`.

## 4. Play in the terminal (optional)

```bash
python -m training.play
```

## 5. Evaluate the model (optional)

```bash
python -m training.evaluate --games 1000
```

## 6. Launch the web game

Make sure the model has finished training. Then run:

```bash
uvicorn backend.main:app --reload
```

Open this address in your browser:

http://127.0.0.1:8000

Click **New game**. The PPO agent plays X and makes the first move; you play O.

Stop the server with `Ctrl+C`.

## 7. View TensorBoard (optional)

Open a second terminal, activate the same virtual environment, then run:

```bash
tensorboard --logdir logs
```

Open:

http://127.0.0.1:6006

## Project structure

```text
tic_tac_toe_ppo/
├── environment/
│   ├── __init__.py
│   └── tic_tac_toe.py
├── training/
│   ├── __init__.py
│   ├── train.py
│   ├── evaluate.py
│   └── play.py
├── models/
├── checkpoints/
├── logs/
├── backend/
│   ├── __init__.py
│   └── main.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── requirements.txt
└── README.md
```

## Notes

- This is a learning project and a baseline, not a perfect Tic-Tac-Toe solver.
- Training uses a random opponent. The agent always plays X in the training environment.
- The web backend keeps one game state in server memory. It is intended for local,
  single-user play, not simultaneous multiplayer sessions.
- The game uses action masking so the PPO policy only selects empty board cells.


## Deploy on Streamlit Community Cloud

1. Train the model locally:
   `python -m training.train`
2. Confirm that `models/tic_tac_toe_ppo.zip` exists.
3. Commit and push the project, including the trained model, to a GitHub repository.
4. Open https://share.streamlit.io/ and sign in with GitHub.
5. Select **Create app**, choose the repository and branch, and set the main file path to:
   `streamlit_app.py`
6. Deploy. Streamlit installs packages from `requirements.txt`.

The Streamlit app loads the trained model from `models/tic_tac_toe_ppo.zip`.
Do not deploy before the model file is present in the repository.
