# CLAUDE.md

This file provides guidance for Claude Code when working with this codebase.

## Project Overview

AlphaGo Zero implementation for Gomoku (无禁手五子棋 - Five in a Row without forbidden moves). The AI learns to play entirely through self-play reinforcement learning.

## Key Files

| File | Purpose |
|------|---------|
| `game.py` | Pure Python game environment |
| `game_fast.py` | Numba JIT-accelerated game (preferred) |
| `network.py` | ResNet with policy/value heads |
| `mcts.py` | Monte Carlo Tree Search |
| `self_play.py` | Self-play data generation |
| `train.py` | Main training loop |
| `play.py` | Human vs AI interface |

## Architecture

### State Representation (4 channels)
- Channel 0: Current player's stones (1 where stone exists)
- Channel 1: Opponent's stones
- Channel 2: Last move position (one-hot)
- Channel 3: Current player indicator (all 1s if black, all 0s if white)

### Neural Network
- Input: `(batch, 4, board_size, board_size)`
- Output: `policy_logits (batch, board_size²)`, `value (batch, 1)`
- Policy is raw logits (apply softmax for probabilities)
- Value is tanh output in [-1, 1]

### MCTS
- Uses PUCT formula: `Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))`
- Dirichlet noise added at root during training
- Returns visit counts as policy target

## Common Commands

```bash
# Train (fast test)
python train.py --board-size 9 --mcts-sims 100 --num-channels 64

# Train (full)
python train.py --mcts-sims 800 --num-iterations 1000

# Play against AI
python play.py play --model checkpoints/latest.pt

# Evaluate models
python play.py evaluate --model1 path1.pt --model2 path2.pt
```

## Code Patterns

### Game State
```python
from game_fast import GomokuGame
game = GomokuGame(board_size=15)
game.reset()
state, reward, done = game.step(action)  # action = row * 15 + col
valid = game.get_valid_moves()  # (225,) float32 mask
```

### Neural Network Inference
```python
from network import AlphaZeroNetwork
net = AlphaZeroNetwork(board_size=15)
policy, value = net.predict(state_tensor, valid_moves_tensor)
```

### MCTS
```python
from mcts import MCTS
mcts = MCTS(network, num_simulations=800, device='cuda')
action, policy = mcts.select_action(game, temperature=1.0)
```

## Performance Notes

- Use `game_fast.py` (Numba) for ~100x speedup on game logic
- Neural network inference is the main bottleneck during MCTS
- GPU strongly recommended for training
- Smaller board (9x9) and fewer MCTS sims for faster iteration

## Conventions

- Board coordinates: (row, col) from top-left (0,0)
- Action index: `row * board_size + col`
- Player: 1 = Black (first), -1 = White
- Winner: 1 = Black wins, -1 = White wins, 0 = Draw/ongoing
