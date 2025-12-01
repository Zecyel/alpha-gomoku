"""
Self-play data generation for AlphaGo Zero style training.
"""
import numpy as np
import torch
from typing import List, Tuple
from dataclasses import dataclass
from collections import deque
import random

try:
    from game_fast import GomokuGame
except ImportError:
    from game import GomokuGame
from mcts import MCTS


@dataclass
class GameRecord:
    """Record of a single position during self-play."""
    state: np.ndarray  # Board state
    policy: np.ndarray  # MCTS visit counts (training target)
    value: float  # Game outcome from this player's perspective


class ReplayBuffer:
    """
    Replay buffer for storing self-play games.
    """

    def __init__(self, max_size: int = 500000):
        self.buffer = deque(maxlen=max_size)

    def push(self, records: List[GameRecord]):
        """Add game records to buffer."""
        for record in records:
            self.buffer.append(record)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch from the buffer.

        Returns:
            states: Shape (batch_size, channels, board_size, board_size)
            policies: Shape (batch_size, board_size * board_size)
            values: Shape (batch_size,)
        """
        batch = random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

        states = np.array([r.state for r in batch])
        policies = np.array([r.policy for r in batch])
        values = np.array([r.value for r in batch])

        return states, policies, values

    def __len__(self):
        return len(self.buffer)


class SelfPlay:
    """
    Self-play game generation for training.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        game: GomokuGame,
        mcts_simulations: int = 800,
        mcts_batch_size: int = 8,
        c_puct: float = 1.5,
        temperature_threshold: int = 30,
        device: str = 'cpu',
    ):
        """
        Args:
            network: Neural network for MCTS
            game: Game instance (for configuration)
            mcts_simulations: Number of MCTS simulations per move
            mcts_batch_size: Batch size for MCTS leaf evaluation
            c_puct: Exploration constant
            temperature_threshold: Move number after which temperature becomes 0
            device: Device for inference
        """
        self.network = network
        self.game = game
        self.mcts = MCTS(
            network=network,
            num_simulations=mcts_simulations,
            batch_size=mcts_batch_size,
            c_puct=c_puct,
            device=device,
        )
        self.temperature_threshold = temperature_threshold
        self.device = device

    def play_game(self, use_augmentation: bool = True) -> List[GameRecord]:
        """
        Play a single self-play game.

        Args:
            use_augmentation: Whether to apply symmetry augmentation

        Returns:
            List of game records for training
        """
        game = GomokuGame(self.game.board_size, self.game.win_length)
        game.reset()

        history = []  # (state, policy, current_player)
        move_count = 0

        while not game.game_over:
            # Temperature: 1 for first N moves, 0 afterwards
            temperature = 1.0 if move_count < self.temperature_threshold else 0.0

            # Get MCTS policy
            action, policy = self.mcts.select_action(
                game,
                temperature=temperature,
                add_noise=True,
            )

            # Record state and policy
            state = game.get_state()
            current_player = game.current_player
            history.append((state.copy(), policy.copy(), current_player))

            # Make move
            game.step(action)
            move_count += 1

        # Assign values based on game outcome
        records = []
        for state, policy, player in history:
            # Value from this player's perspective
            if game.winner == 0:
                value = 0.0  # Draw
            elif game.winner == player:
                value = 1.0  # This player won
            else:
                value = -1.0  # This player lost

            if use_augmentation:
                # Apply all 8 symmetries
                symmetries = game.get_symmetries(state, policy)
                for sym_state, sym_policy in symmetries:
                    records.append(GameRecord(
                        state=sym_state,
                        policy=sym_policy,
                        value=value,
                    ))
            else:
                records.append(GameRecord(
                    state=state,
                    policy=policy,
                    value=value,
                ))

        return records

    def generate_games(self, num_games: int, use_augmentation: bool = True) -> List[GameRecord]:
        """
        Generate multiple self-play games.

        Args:
            num_games: Number of games to play
            use_augmentation: Whether to apply symmetry augmentation

        Returns:
            List of all game records
        """
        all_records = []

        for i in range(num_games):
            records = self.play_game(use_augmentation=use_augmentation)
            all_records.extend(records)

        return all_records


def generate_games_parallel(
    network: torch.nn.Module,
    game: GomokuGame,
    num_games: int,
    num_workers: int = 4,
    mcts_simulations: int = 800,
    device: str = 'cpu',
) -> List[GameRecord]:
    """
    Generate games in parallel using multiple workers.
    Note: For simplicity, this uses sequential generation.
    For true parallelism, consider using multiprocessing with separate network copies.
    """
    self_play = SelfPlay(
        network=network,
        game=game,
        mcts_simulations=mcts_simulations,
        device=device,
    )

    return self_play.generate_games(num_games)


if __name__ == '__main__':
    from network import AlphaZeroNetwork

    # Test self-play
    board_size = 15
    game = GomokuGame(board_size=board_size)
    network = AlphaZeroNetwork(board_size=board_size)

    self_play = SelfPlay(
        network=network,
        game=game,
        mcts_simulations=50,  # Use fewer simulations for testing
    )

    print("Playing a self-play game...")
    records = self_play.play_game(use_augmentation=True)
    print(f"Generated {len(records)} training examples")
    print(f"State shape: {records[0].state.shape}")
    print(f"Policy shape: {records[0].policy.shape}")
    print(f"Value: {records[0].value}")

    # Test replay buffer
    buffer = ReplayBuffer(max_size=10000)
    buffer.push(records)
    print(f"\nReplay buffer size: {len(buffer)}")

    states, policies, values = buffer.sample(32)
    print(f"Sampled batch - States: {states.shape}, Policies: {policies.shape}, Values: {values.shape}")
