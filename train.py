"""
Training loop for AlphaGo Zero style Gomoku.
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from typing import Optional
import json
from datetime import datetime

try:
    from game_fast import GomokuGame
except ImportError:
    from game import GomokuGame
from network import AlphaZeroNetwork, AlphaZeroLoss
from mcts import MCTS
from self_play import SelfPlay, ReplayBuffer
try:
    from parallel_compiled import CompiledParallelMCTS as ParallelMCTS
    USING_COMPILED = True
except ImportError:
    from parallel_self_play import ParallelMCTS
    USING_COMPILED = False


class Trainer:
    """
    AlphaGo Zero training pipeline.
    """

    def __init__(
        self,
        board_size: int = 15,
        num_channels: int = 128,
        num_res_blocks: int = 10,
        mcts_simulations: int = 800,
        mcts_batch_size: int = 16,
        parallel_games: int = 16,
        c_puct: float = 1.5,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        buffer_size: int = 500000,
        checkpoint_dir: str = 'checkpoints',
        device: str = None,
        use_bf16: bool = True,
    ):
        """
        Initialize trainer.

        Args:
            board_size: Size of the Gomoku board
            num_channels: Number of channels in residual network
            num_res_blocks: Number of residual blocks
            mcts_simulations: MCTS simulations per move
            mcts_batch_size: Batch size for MCTS leaf evaluation
            parallel_games: Number of games to play in parallel
            c_puct: Exploration constant for MCTS
            lr: Learning rate
            weight_decay: L2 regularization
            batch_size: Training batch size
            buffer_size: Maximum replay buffer size
            checkpoint_dir: Directory for saving checkpoints
            device: Device for training
            use_bf16: Use bfloat16 mixed precision
        """
        self.board_size = board_size
        self.mcts_simulations = mcts_simulations
        self.mcts_batch_size = mcts_batch_size
        self.parallel_games = parallel_games
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.use_bf16 = use_bf16

        # Setup device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"Using device: {self.device}")

        # Set TensorFloat32 precision for better performance
        if 'cuda' in self.device:
            torch.set_float32_matmul_precision('high')

        # Create game and network
        self.game = GomokuGame(board_size=board_size)
        self.network = AlphaZeroNetwork(
            board_size=board_size,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
        ).to(self.device)

        # Convert to bfloat16 if requested
        if use_bf16 and 'cuda' in self.device:
            print("Using bfloat16 precision")
            self.network = self.network.to(torch.bfloat16)
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        # Compile network for faster inference (PyTorch 2.0+)
        # Note: Skip 'reduce-overhead' mode - it causes issues with mixed inference/training
        # Use 'default' mode for compatibility
        if hasattr(torch, 'compile'):
            print("Compiling network with torch.compile()...")
            self.network = torch.compile(self.network, mode='default')

        # Optimizer and loss
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.criterion = AlphaZeroLoss()

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=buffer_size)

        # Self-play generator (legacy, for single game)
        self.self_play = SelfPlay(
            network=self.network,
            game=self.game,
            mcts_simulations=mcts_simulations,
            mcts_batch_size=mcts_batch_size,
            c_puct=c_puct,
            device=self.device,
        )

        # Parallel self-play generator
        self.parallel_mcts = ParallelMCTS(
            network=self.network,
            num_games=parallel_games,
            num_simulations=mcts_simulations,
            batch_size=mcts_batch_size * parallel_games,
            c_puct=c_puct,
            device=self.device,
        )

        # Training stats
        self.iteration = 0
        self.training_history = []

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

    def generate_self_play_data(self, num_games: int) -> int:
        """
        Generate self-play games and add to replay buffer using parallel self-play.

        Args:
            num_games: Number of games to generate

        Returns:
            Number of new training examples added
        """
        self.network.eval()

        all_records = []
        games_played = 0

        # Play games in parallel batches
        with tqdm(total=num_games, desc="Self-play") as pbar:
            while games_played < num_games:
                # Adjust parallel games for last batch
                batch_games = min(self.parallel_games, num_games - games_played)
                self.parallel_mcts.num_games = batch_games

                records = self.parallel_mcts.play_games(
                    board_size=self.board_size,
                    use_augmentation=True
                )
                all_records.extend(records)
                games_played += batch_games
                pbar.update(batch_games)

        self.replay_buffer.push(all_records)
        return len(all_records)

    def train_step(self, num_batches: int) -> dict:
        """
        Train the network for a number of batches.

        Args:
            num_batches: Number of training batches

        Returns:
            Dictionary of training metrics
        """
        self.network.train()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for _ in range(num_batches):
            # Sample batch from replay buffer
            states, policies, values = self.replay_buffer.sample(self.batch_size)

            # Convert to tensors with correct dtype
            states = torch.tensor(states, dtype=self.dtype, device=self.device)
            target_policies = torch.tensor(policies, dtype=self.dtype, device=self.device)
            target_values = torch.tensor(values, dtype=self.dtype, device=self.device)

            # Forward pass
            policy_logits, pred_values = self.network(states)

            # Compute loss (cast to float32 for stable loss computation)
            loss, policy_loss, value_loss = self.criterion(
                policy_logits.float(), pred_values.float(),
                target_policies.float(), target_values.float()
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        return {
            'loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
        }

    def train(
        self,
        num_iterations: int = 1000,
        games_per_iteration: int = 100,
        batches_per_iteration: int = 100,
        min_buffer_size: int = 10000,
        checkpoint_interval: int = 10,
        eval_interval: int = 50,
    ):
        """
        Main training loop.

        Args:
            num_iterations: Total training iterations
            games_per_iteration: Self-play games per iteration
            batches_per_iteration: Training batches per iteration
            min_buffer_size: Minimum buffer size before training
            checkpoint_interval: Iterations between checkpoints
            eval_interval: Iterations between evaluation games
        """
        print(f"\n{'='*60}")
        print("Starting AlphaGo Zero Training for Gomoku")
        print(f"{'='*60}")
        print(f"Board size: {self.board_size}x{self.board_size}")
        print(f"Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")
        print(f"Device: {self.device}")
        print(f"Parallel games: {self.parallel_games}")
        print(f"MCTS: {'Compiled (Numba)' if USING_COMPILED else 'Standard'}")
        print(f"{'='*60}\n")

        start_iteration = self.iteration
        end_iteration = start_iteration + num_iterations

        for iteration in range(start_iteration, end_iteration):
            self.iteration = iteration
            print(f"\n--- Iteration {iteration + 1}/{end_iteration} ---")

            # Generate self-play data
            num_new_examples = self.generate_self_play_data(games_per_iteration)
            print(f"Generated {num_new_examples} new examples, buffer size: {len(self.replay_buffer)}")

            # Wait for minimum buffer size
            if len(self.replay_buffer) < min_buffer_size:
                print(f"Buffer size ({len(self.replay_buffer)}) < minimum ({min_buffer_size}), skipping training")
                continue

            # Train
            metrics = self.train_step(batches_per_iteration)
            print(f"Loss: {metrics['loss']:.4f} (Policy: {metrics['policy_loss']:.4f}, Value: {metrics['value_loss']:.4f})")

            # Record history
            self.training_history.append({
                'iteration': iteration,
                'buffer_size': len(self.replay_buffer),
                **metrics,
            })

            # Save checkpoint
            if (iteration + 1) % checkpoint_interval == 0:
                self.save_checkpoint()

            # Evaluation
            if (iteration + 1) % eval_interval == 0:
                self.evaluate()

            # Step scheduler
            self.scheduler.step()

        print("\nTraining complete!")
        self.save_checkpoint()

    def evaluate(self, num_games: int = 10):
        """
        Evaluate the current model by playing games against a random opponent.
        """
        self.network.eval()
        mcts = MCTS(self.network, num_simulations=self.mcts_simulations, device=self.device)

        wins = 0
        draws = 0
        losses = 0

        for game_idx in range(num_games):
            game = GomokuGame(self.board_size)
            game.reset()

            # Alternate who plays first
            ai_player = 1 if game_idx % 2 == 0 else -1

            while not game.game_over:
                if game.current_player == ai_player:
                    # AI plays with MCTS
                    action, _ = mcts.select_action(game, temperature=0, add_noise=False)
                else:
                    # Random opponent
                    valid_moves = game.get_valid_moves_list()
                    action = np.random.choice(valid_moves)

                game.step(action)

            if game.winner == ai_player:
                wins += 1
            elif game.winner == 0:
                draws += 1
            else:
                losses += 1

        print(f"Evaluation vs Random: Wins={wins}, Draws={draws}, Losses={losses}")
        return wins, draws, losses

    def save_checkpoint(self, path: Optional[str] = None):
        """Save training checkpoint."""
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.checkpoint_dir, f"checkpoint_iter{self.iteration}_{timestamp}.pt")

        checkpoint = {
            'iteration': self.iteration,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'config': {
                'board_size': self.board_size,
                'mcts_simulations': self.mcts_simulations,
                'c_puct': self.c_puct,
                'batch_size': self.batch_size,
            },
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

        # Also save as 'latest'
        latest_path = os.path.join(self.checkpoint_dir, "latest.pt")
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.iteration = checkpoint['iteration']
        self.training_history = checkpoint['training_history']
        print(f"Loaded checkpoint from {path}, iteration {self.iteration}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train AlphaGo Zero for Gomoku')
    parser.add_argument('--board-size', type=int, default=15, help='Board size')
    parser.add_argument('--num-iterations', type=int, default=100, help='Training iterations')
    parser.add_argument('--games-per-iter', type=int, default=25, help='Self-play games per iteration')
    parser.add_argument('--batches-per-iter', type=int, default=100, help='Training batches per iteration')
    parser.add_argument('--mcts-sims', type=int, default=400, help='MCTS simulations per move')
    parser.add_argument('--mcts-batch', type=int, default=16, help='MCTS batch size for leaf evaluation')
    parser.add_argument('--parallel-games', type=int, default=16, help='Number of games to play in parallel')
    parser.add_argument('--num-channels', type=int, default=128, help='Network channels')
    parser.add_argument('--num-res-blocks', type=int, default=10, help='Residual blocks')
    parser.add_argument('--batch-size', type=int, default=256, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default=None, help='Device (e.g., cuda:0, cuda:1)')
    parser.add_argument('--no-bf16', action='store_true', help='Disable bfloat16 precision')

    args = parser.parse_args()

    trainer = Trainer(
        board_size=args.board_size,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        mcts_simulations=args.mcts_sims,
        mcts_batch_size=args.mcts_batch,
        parallel_games=args.parallel_games,
        lr=args.lr,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        use_bf16=not args.no_bf16,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train(
        num_iterations=args.num_iterations,
        games_per_iteration=args.games_per_iter,
        batches_per_iteration=args.batches_per_iter,
    )
