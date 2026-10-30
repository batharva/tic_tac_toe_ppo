const cells = [...document.querySelectorAll(".cell")];
const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");
const newGameButton = document.getElementById("new-game");

let gameOver = true;

function showError(message) {
  errorEl.textContent = message || "";
}

function render(state) {
  gameOver = state.game_over;
  cells.forEach((button, index) => {
    const value = state.board[index];
    button.textContent = value === 1 ? "X" : value === -1 ? "O" : "";
    button.classList.toggle("mark-x", value === 1);
    button.classList.toggle("mark-o", value === -1);
    button.disabled = gameOver || value !== 0;
  });

  if (state.message) statusEl.textContent = state.message;
  else if (gameOver && state.winner === "ai") statusEl.textContent = "PPO wins!";
  else if (gameOver && state.winner === "human") statusEl.textContent = "You win!";
  else if (gameOver && state.winner === "draw") statusEl.textContent = "It's a draw.";
  else statusEl.textContent = "Your turn (O).";
}

async function api(path, body) {
  const response = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Something went wrong.");
  return data;
}

async function startGame() {
  showError("");
  newGameButton.disabled = true;
  try {
    render(await api("/api/new", {}));
  } catch (error) {
    showError(error.message);
    statusEl.textContent = "Could not start the game.";
  } finally {
    newGameButton.disabled = false;
  }
}

async function makeMove(cell) {
  if (gameOver) return;
  showError("");
  cells.forEach(button => button.disabled = true);
  try {
    render(await api("/api/move", { cell }));
  } catch (error) {
    showError(error.message);
    try { render(await api("/api/state")); } catch (_) {}
  }
}

cells.forEach(button => {
  button.addEventListener("click", () => makeMove(Number(button.dataset.cell)));
});
newGameButton.addEventListener("click", startGame);

// Start with an empty board; user presses New game to let the AI make its first move.
api("/api/state").then(render).catch(error => showError(error.message));
