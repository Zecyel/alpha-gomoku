"""
Monte Carlo Tree Search (MCTS) for AlphaGo Zero style Gomoku.

This module re-exports BatchedMCTS as the default MCTS implementation
for better performance. The batched version evaluates multiple leaf nodes
in a single neural network forward pass.
"""

# Re-export batched implementation as default
from mcts_batched import BatchedMCTS as MCTS, MCTSNode

__all__ = ['MCTS', 'MCTSNode']
