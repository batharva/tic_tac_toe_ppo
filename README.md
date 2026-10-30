# 🎮 PPO Tic-Tac-Toe

### A Reinforcement Learning Project using Proximal Policy Optimization (PPO)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Environment-Gymnasium-orange)](https://gymnasium.farama.org/)
[![RL Algorithm](https://img.shields.io/badge/Algorithm-PPO-green)](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A web-based Tic-Tac-Toe game powered by **Reinforcement Learning (RL)**. The project implements a PPO-based AI agent that learns to play Tic-Tac-Toe through interaction with a custom Gymnasium environment.

Play against the trained AI directly in your browser!

<p align="center">
  <a href="https://iatharvab-tictactoeppo.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_PLAY_LIVE-Launch_App-success?style=for-the-badge" alt="Play Live"/>
  </a>
  <a href="https://github.com/batharva/tic_tac_toe_ppo">
    <img src="https://img.shields.io/badge/GitHub-Source_Code-black?style=for-the-badge&logo=github" alt="Source Code"/>
  </a>
</p>

---

## 🌐 Live Demo

**Play here:** https://iatharvab-tictactoeppo.streamlit.app/

- AI plays as **X**.
- Human player plays as **O**.
- The AI makes the first move.
- Start a new game and challenge the trained agent.

---

## 🧠 Project Overview

The objective of this project is to implement a Reinforcement Learning agent capable of learning a strategy for Tic-Tac-Toe using Proximal Policy Optimization.

Instead of using predefined rules to select moves, the agent learns a policy by interacting with the environment and receiving rewards based on game outcomes.

The project consists of three major components:

1. **Custom Gymnasium Environment:** Defines the game mechanics, state space, action space and reward system.
2. **PPO Agent:** Uses a neural network-based policy to learn action selection.
3. **Web Application:** Provides an interactive interface to play against the trained agent using Streamlit.

## ⚙️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| RL Algorithm | Proximal Policy Optimization (PPO) |
| RL Framework | Stable-Baselines3 |
| Action Masking | SB3-Contrib |
| Environment | Gymnasium |
| Deep Learning | PyTorch |
| Numerical Computation | NumPy |
| Web Interface | Streamlit |
| Training Visualization | TensorBoard |
| Deployment | Streamlit Community Cloud |
| Version Control | Git & GitHub |

---

## 🎯 Reinforcement Learning Formulation

The Tic-Tac-Toe problem is formulated as a Markov Decision Process (MDP).

### 1. State Space (Observation)

The board is represented as a one-dimensional array containing nine elements.

Each cell has one of three possible values:

| Value | Meaning |
|---|---|
| `1` | AI player (X) |
| `-1` | Human/opponent (O) |
| `0` | Empty cell |

Example:

```python
observation = [
    1,  0, -1,
    0,  1,  0,
   -1,  0,  0
]
```

The observation space is defined as:

```python
spaces.Box(
    low=-1,
    high=1,
    shape=(9,),
    dtype=np.int8
)
```

### 2. Action Space

The agent can select any of the nine board positions.

```python
spaces.Discrete(9)
```

Each action corresponds to one cell:

```text
  0 | 1 | 2
 ---+---+---
  3 | 4 | 5
 ---+---+---
  6 | 7 | 8
```

### 3. Reward Function

The agent receives rewards based on the outcome of the game.

| Game Outcome | Reward |
|---|---:|
| AI wins | +1 |
| AI loses | -1 |
| Draw | +0.5 |
| Valid move, game continues | 0 |
| Invalid move | -1 |

The objective is to learn a policy that maximizes the expected cumulative reward.

### 4. Action Masking

The project uses **Maskable PPO** from SB3-Contrib to prevent the agent from selecting occupied cells.

The action mask identifies legal moves:

```python
def action_masks(self):
    return self.board == 0
```

- `True`: Valid action.
- `False`: Invalid action.

This allows the policy to select only from currently available positions.

---

## 🤖 PPO Algorithm

Proximal Policy Optimization (PPO) is a policy-gradient reinforcement learning algorithm that improves a policy through constrained updates.

The implementation uses an Actor-Critic architecture.

**Actor Network**

- Takes the current board observation.
- Produces a probability distribution over the nine possible actions.
- Selects an action based on the learned policy.

**Critic Network**

- Estimates the value of the current state.
- Helps calculate the advantage function.
- Guides policy optimization.

### Neural Network Architecture

```text
       Board Observation
          (9 Inputs)
              |
              v
       Shared MLP Extractor
              |
       +------+------+
       |             |
       v             v
      Actor        Critic
       |             |
       v             v
   9 Actions     State Value
```

### PPO Objective

PPO uses a clipped surrogate objective:

\[
L^{CLIP}(\theta)=
\mathbb{E}_t[
\min(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)
\hat A_t
)]
\]

Where:

- \(r_t(\theta)\) is the probability ratio between new and old policies.
- \(\hat A_t\) is the estimated advantage.
- \(\epsilon\) controls the clipping range.

---

## 📁 Project Structure

```text
tic_tac_toe_ppo/
│
├── environment/
│   ├── __init__.py
│   └── tic_tac_toe.py
│
├── training/
│   ├── __init__.py
│   ├── train.py
│   ├── evaluate.py
│   ├── play.py
│   ├── test_environment.py
│   └── check_env.py
│
├── models/
│   └── tic_tac_toe_ppo.zip
│
├── checkpoints/
│
├── logs/
│
├── backend/
│   ├── __init__.py
│   └── main.py
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── streamlit_app.py
├── requirements.txt
├── README.md
└── LICENSE
```

### Directory Description

| Directory/File | Purpose |
|---|---|
| `environment/` | Custom Gymnasium game environment |
| `training/` | PPO training and evaluation scripts |
| `models/` | Saved trained PPO model |
| `checkpoints/` | Intermediate training checkpoints |
| `logs/` | TensorBoard training logs |
| `backend/` | FastAPI implementation |
| `frontend/` | HTML, CSS and JavaScript interface |
| `streamlit_app.py` | Streamlit web application |
| `requirements.txt` | Project dependencies |

---

## 🚀 Installation and Setup

### Prerequisites

- Python 3.10 or newer
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/batharva/tic_tac_toe_ppo.git

cd tic_tac_toe_ppo
```

### 2. Create a virtual environment

**Windows:**

```bash
py -m venv .venv

.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv .venv

source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

---

## 🏋️ Training the PPO Agent

To train the agent from scratch, execute:

```bash
python -m training.train
```

The default training configuration uses 200,000 timesteps.

To increase training duration:

```bash
python -m training.train --timesteps 500000
```

The trained model is saved at:

```text
models/tic_tac_toe_ppo.zip
```

Intermediate checkpoints are stored in the `checkpoints/` directory.

### Monitor Training

TensorBoard can be used to visualize training metrics.

Run:

```bash
tensorboard --logdir logs
```

Open:

```text
http://localhost:6006
```

### Evaluate the Agent

Evaluate the trained model over 1,000 games:

```bash
python -m training.evaluate --games 1000
```

### Play in the Terminal

```bash
python -m training.play
```

---

## 🌍 Run the Web Application Locally

The project provides a Streamlit-based interface.

After ensuring that the trained model exists, execute:

```bash
streamlit run streamlit_app.py
```

Open the application at:

```text
http://localhost:8501
```

---

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud**.

Deployment workflow:

1. Train the PPO agent.
2. Save the trained model in the `models/` directory.
3. Push the project files and model to GitHub.
4. Connect the GitHub repository to Streamlit Community Cloud.
5. Configure `streamlit_app.py` as the main application file.
6. Deploy the application.

**Live Application:** https://iatharvab-tictactoeppo.streamlit.app/

---

## 📊 Current Implementation and Limitations

- The agent is trained using a random opponent.
- The training agent always plays as X.
- The web application uses the trained model for deterministic action selection.
- Action masking prevents the PPO agent from selecting occupied cells.
- The current implementation is a learning baseline and is not guaranteed to play optimally.

### Future Improvements

- [ ] Implement self-play reinforcement learning.
- [ ] Introduce a minimax-based evaluation opponent.
- [ ] Compare PPO performance against random and optimal opponents.
- [ ] Add training progress visualization to the web interface.
- [ ] Experiment with different neural network architectures.
- [ ] Compare different reward functions and PPO hyperparameters.
- [ ] Add win-rate and cumulative-reward analytics.

---

## 📚 References

1. Schulman et al. (2017) — [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
2. [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/).
3. [SB3-Contrib Documentation](https://sb3-contrib.readthedocs.io/).
4. [Gymnasium Documentation](https://gymnasium.farama.org/).

---

## 👨‍💻 Author

**Atharva Bhosale**

- GitHub: [@batharva](https://github.com/batharva)
- Project Repository: [tic_tac_toe_ppo](https://github.com/batharva/tic_tac_toe_ppo)
- Live Application: [PPO Tic-Tac-Toe](https://iatharvab-tictactoeppo.streamlit.app/)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

⭐ If you find this project interesting, consider giving the repository a star!
