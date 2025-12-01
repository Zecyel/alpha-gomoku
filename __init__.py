"""
AlphaGo Zero for Gomoku (无禁手五子棋)

A PyTorch implementation of the AlphaGo Zero algorithm for Gomoku.

Usage:
    # Train a new model
    python train.py --num-iterations 100 --games-per-iter 25

    # Resume training
    python train.py --resume checkpoints/latest.pt

    # Play against trained model
    python play.py play --model checkpoints/latest.pt

    # Evaluate models
    python play.py evaluate --model1 checkpoints/model_v1.pt --model2 checkpoints/model_v2.pt
"""

from game import GomokuGame
from network import AlphaZeroNetwork
from mcts import MCTS
from self_play import SelfPlay, ReplayBuffer
from train import Trainer

__all__ = [
    'GomokuGame',
    'AlphaZeroNetwork',
    'MCTS',
    'SelfPlay',
    'ReplayBuffer',
    'Trainer',
]
