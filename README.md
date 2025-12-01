# AlphaGo Zero for Gomoku (无禁手五子棋)

A PyTorch implementation of the AlphaGo Zero algorithm for Gomoku (Five in a Row) without forbidden moves.

## Overview

This project reproduces the core ideas from the AlphaGo Zero paper:
- **Self-play reinforcement learning** - The AI learns entirely from playing against itself
- **Monte Carlo Tree Search (MCTS)** - Guided by neural network predictions
- **Deep residual network** - Dual heads for policy (move probabilities) and value (win probability)
- **No human knowledge** - Learns from scratch with only game rules

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input (4 channels)                    │
│  [Current player stones, Opponent stones, Last move, Turn]│
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Convolutional Layer (3x3, 128ch)            │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              Residual Blocks x 10                        │
│     ┌─────────────────────────────────┐                 │
│     │  Conv 3x3 → BN → ReLU           │                 │
│     │  Conv 3x3 → BN → (+input) → ReLU│                 │
│     └─────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌──────────────────────┐    ┌──────────────────────┐
│     Policy Head      │    │     Value Head       │
│  Conv 1x1 → FC       │    │  Conv 1x1 → FC → FC  │
│  Output: 225 moves   │    │  Output: [-1, 1]     │
└──────────────────────┘    └──────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/alpha-gomoku.git
cd alpha-gomoku

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy
- tqdm
- Numba (for fast game logic)

## Usage

### Training

```bash
# Basic training
python train.py

# Custom configuration
python train.py \
    --board-size 15 \
    --num-iterations 1000 \
    --games-per-iter 100 \
    --mcts-sims 800 \
    --num-channels 128 \
    --num-res-blocks 10

# Resume from checkpoint
python train.py --resume checkpoints/latest.pt

# Quick test with smaller settings
python train.py --board-size 9 --mcts-sims 100 --num-channels 64 --num-res-blocks 5
```

### Playing Against the AI

```bash
# Play against trained model
python play.py play --model checkpoints/latest.pt

# Let AI play first (you play white)
python play.py play --model checkpoints/latest.pt --ai-first

# Adjust AI strength
python play.py play --model checkpoints/latest.pt --mcts-sims 1600
```

### Evaluating Models

```bash
# Compare two models
python play.py evaluate --model1 checkpoints/v1.pt --model2 checkpoints/v2.pt --num-games 100

# Evaluate against random player
python play.py evaluate --model1 checkpoints/latest.pt --num-games 50
```

## Project Structure

```
alpha-gomoku/
├── game.py          # Gomoku game environment (pure Python)
├── game_fast.py     # Numba-accelerated game environment
├── network.py       # Neural network architecture (ResNet)
├── mcts.py          # Monte Carlo Tree Search
├── self_play.py     # Self-play data generation
├── train.py         # Training pipeline
├── play.py          # Human vs AI interface
├── requirements.txt # Dependencies
└── checkpoints/     # Saved models
```

## Key Components

### Game Environment (`game.py`, `game_fast.py`)
- 15x15 board (configurable)
- No forbidden moves (无禁手)
- 5 in a row wins
- State representation: 4-channel tensor

### Neural Network (`network.py`)
- Input: 4 x 15 x 15 tensor
- Body: 10 residual blocks with 128 channels
- Policy head: probability distribution over 225 moves
- Value head: scalar in [-1, 1] (win probability)

### MCTS (`mcts.py`)
- Selection: PUCT formula (Q + c * P * sqrt(N_parent) / (1 + N))
- Expansion: Neural network evaluation
- Backup: Alternating value propagation
- Dirichlet noise at root for exploration

### Training (`train.py`)
- Self-play game generation
- Replay buffer (500K positions)
- Loss: MSE(value) + CrossEntropy(policy)
- Adam optimizer with weight decay

## Training Tips

1. **Start small**: Use 9x9 board and fewer MCTS simulations for faster iteration
2. **GPU recommended**: Training on CPU is very slow
3. **Monitor progress**: Watch win rate against random player
4. **Data augmentation**: 8-fold symmetry is applied automatically

## Performance Optimization

The `game_fast.py` module uses Numba JIT compilation for critical game logic:
- Clone: ~2.7M ops/sec
- Step: ~265K ops/sec
- Win check: ~4.1M ops/sec

## References

- [Mastering the Game of Go without Human Knowledge](https://www.nature.com/articles/nature24270) (AlphaGo Zero paper)
- [Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815) (AlphaZero paper)

## License

MIT License
