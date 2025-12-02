"""
Play interface for Gomoku AI.
Allows human vs AI games and model evaluation.
"""
import torch
import numpy as np
from typing import Optional
import argparse

try:
    from game_fast import GomokuGame
except ImportError:
    from game import GomokuGame
from network import AlphaZeroNetwork
from mcts import MCTS


class GomokuPlayer:
    """Base class for Gomoku players."""

    def get_action(self, game: 'GomokuGame') -> int:
        raise NotImplementedError


class HumanPlayer(GomokuPlayer):
    """Human player via command line input."""

    def get_action(self, game: 'GomokuGame') -> int:
        while True:
            try:
                user_input = input("Enter move (row col): ").strip()
                parts = user_input.split()
                if len(parts) != 2:
                    print("Please enter row and column separated by space")
                    continue

                row, col = int(parts[0]), int(parts[1])
                action = game.coord_to_action(row, col)

                if game.is_valid_move(action):
                    return action
                else:
                    print("Invalid move, position already occupied")
            except (ValueError, IndexError):
                print("Invalid input, please enter two numbers")


class AIPlayer(GomokuPlayer):
    """AI player using MCTS."""

    def __init__(
        self,
        network: AlphaZeroNetwork,
        mcts_simulations: int = 800,
        c_puct: float = 1.5,
        device: str = 'cpu',
    ):
        self.network = network
        self.mcts = MCTS(
            network=network,
            num_simulations=mcts_simulations,
            c_puct=c_puct,
            device=device,
        )
        self.device = device

    def get_action(self, game: 'GomokuGame') -> int:
        action, policy = self.mcts.select_action(
            game,
            temperature=0,  # Deterministic play
            add_noise=False,
        )

        # Show top moves
        top_actions = np.argsort(policy)[-3:][::-1]
        print("AI thinking...")
        for a in top_actions:
            r, c = game.action_to_coord(a)
            print(f"  ({r}, {c}): {policy[a]:.2%}")

        return action


class RandomPlayer(GomokuPlayer):
    """Random player for testing."""

    def get_action(self, game: 'GomokuGame') -> int:
        valid_moves = game.get_valid_moves_list()
        return np.random.choice(valid_moves)


def load_model(checkpoint_path: str, device: str = 'cpu') -> AlphaZeroNetwork:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get('config', {})
    board_size = config.get('board_size', 15)

    # Try to infer network architecture from state dict
    state_dict = checkpoint['model_state_dict']
    num_channels = state_dict['input_conv.weight'].shape[0]

    # Count residual blocks
    res_block_keys = [k for k in state_dict.keys() if k.startswith('res_blocks')]
    num_res_blocks = len(set(k.split('.')[1] for k in res_block_keys))

    network = AlphaZeroNetwork(
        board_size=board_size,
        num_channels=num_channels,
        num_res_blocks=num_res_blocks,
    ).to(device)

    network.load_state_dict(state_dict)
    network.eval()

    print(f"Loaded model: {board_size}x{board_size} board, {num_channels} channels, {num_res_blocks} res blocks")
    return network


def play_game(
    player1: GomokuPlayer,
    player2: GomokuPlayer,
    board_size: int = 15,
    show_board: bool = True,
) -> int:
    """
    Play a game between two players.

    Args:
        player1: Black player (first)
        player2: White player (second)
        board_size: Board size
        show_board: Whether to print the board

    Returns:
        Winner: 1 for player1, -1 for player2, 0 for draw
    """
    game = GomokuGame(board_size=board_size)
    game.reset()

    players = {1: player1, -1: player2}
    player_names = {1: "Black (X)", -1: "White (O)"}

    if show_board:
        print("\nStarting new game!")
        print(game)

    move_count = 0
    while not game.game_over:
        current_player = players[game.current_player]

        if show_board:
            print(f"\n{player_names[game.current_player]}'s turn:")

        action = current_player.get_action(game)
        row, col = game.action_to_coord(action)

        if show_board:
            print(f"Move: ({row}, {col})")

        game.step(action)
        move_count += 1

        if show_board:
            print(game)

    if show_board:
        if game.winner == 0:
            print("\nGame over: Draw!")
        else:
            print(f"\nGame over: {player_names[game.winner]} wins in {move_count} moves!")

    return game.winner


def evaluate_models(
    model1_path: Optional[str],
    model2_path: Optional[str],
    num_games: int = 100,
    mcts_simulations: int = 400,
    board_size: int = 15,
    device: str = 'cpu',
):
    """
    Evaluate two models against each other.

    Args:
        model1_path: Path to first model (None for random)
        model2_path: Path to second model (None for random)
        num_games: Number of games to play
        mcts_simulations: MCTS simulations per move
        board_size: Board size
        device: Device for inference
    """
    # Load models
    if model1_path:
        network1 = load_model(model1_path, device)
        player1_name = f"Model1 ({model1_path})"
        player1 = AIPlayer(network1, mcts_simulations, device=device)
    else:
        player1_name = "Random"
        player1 = RandomPlayer()

    if model2_path:
        network2 = load_model(model2_path, device)
        player2_name = f"Model2 ({model2_path})"
        player2 = AIPlayer(network2, mcts_simulations, device=device)
    else:
        player2_name = "Random"
        player2 = RandomPlayer()

    print(f"\nEvaluating: {player1_name} vs {player2_name}")
    print(f"Number of games: {num_games}")
    print(f"MCTS simulations: {mcts_simulations}")

    # Play games
    player1_wins = 0
    player2_wins = 0
    draws = 0

    for game_idx in range(num_games):
        # Alternate colors
        if game_idx % 2 == 0:
            black_player, white_player = player1, player2
            result = play_game(black_player, white_player, board_size, show_board=False)
            if result == 1:
                player1_wins += 1
            elif result == -1:
                player2_wins += 1
            else:
                draws += 1
        else:
            black_player, white_player = player2, player1
            result = play_game(black_player, white_player, board_size, show_board=False)
            if result == 1:
                player2_wins += 1
            elif result == -1:
                player1_wins += 1
            else:
                draws += 1

        if (game_idx + 1) % 10 == 0:
            print(f"Progress: {game_idx + 1}/{num_games} games")
            print(f"  {player1_name}: {player1_wins} | {player2_name}: {player2_wins} | Draws: {draws}")

    # Final results
    print(f"\n{'='*50}")
    print("Final Results:")
    print(f"{'='*50}")
    print(f"{player1_name}: {player1_wins} ({player1_wins/num_games*100:.1f}%)")
    print(f"{player2_name}: {player2_wins} ({player2_wins/num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/num_games*100:.1f}%)")

    return player1_wins, player2_wins, draws


def interactive_game(
    model_path: Optional[str] = None,
    human_first: bool = True,
    mcts_simulations: int = 800,
    board_size: int = 15,
    device: str = 'cpu',
):
    """
    Play an interactive game against the AI.

    Args:
        model_path: Path to model checkpoint (None for random AI)
        human_first: Whether human plays first (black)
        mcts_simulations: MCTS simulations for AI
        board_size: Board size
        device: Device for inference
    """
    # Setup players
    human = HumanPlayer()

    if model_path:
        network = load_model(model_path, device)
        ai = AIPlayer(network, mcts_simulations, device=device)
        ai_name = "AI"
    else:
        ai = RandomPlayer()
        ai_name = "Random AI"

    print(f"\n{'='*50}")
    print(f"Gomoku: Human vs {ai_name}")
    print(f"{'='*50}")
    print(f"Board size: {board_size}x{board_size}")
    print(f"You are: {'Black (X)' if human_first else 'White (O)'}")
    print(f"{'='*50}")

    if human_first:
        result = play_game(human, ai, board_size)
    else:
        result = play_game(ai, human, board_size)

    if human_first:
        if result == 1:
            print("Congratulations! You won!")
        elif result == -1:
            print(f"The {ai_name} wins!")
        else:
            print("It's a draw!")
    else:
        if result == -1:
            print("Congratulations! You won!")
        elif result == 1:
            print(f"The {ai_name} wins!")
        else:
            print("It's a draw!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play Gomoku against AI')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Play command
    play_parser = subparsers.add_parser('play', help='Play against AI')
    play_parser.add_argument('--model', type=str, default=None, help='Model checkpoint path')
    play_parser.add_argument('--ai-first', action='store_true', help='Let AI play first')
    play_parser.add_argument('--mcts-sims', type=int, default=800, help='MCTS simulations')
    play_parser.add_argument('--board-size', type=int, default=15, help='Board size')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    eval_parser.add_argument('--model1', type=str, default=None, help='First model path')
    eval_parser.add_argument('--model2', type=str, default=None, help='Second model path')
    eval_parser.add_argument('--num-games', type=int, default=100, help='Number of games')
    eval_parser.add_argument('--mcts-sims', type=int, default=400, help='MCTS simulations')
    eval_parser.add_argument('--board-size', type=int, default=15, help='Board size')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.command == 'play':
        interactive_game(
            model_path=args.model,
            human_first=not args.ai_first,
            mcts_simulations=args.mcts_sims,
            board_size=args.board_size,
            device=device,
        )
    elif args.command == 'evaluate':
        evaluate_models(
            model1_path=args.model1,
            model2_path=args.model2,
            num_games=args.num_games,
            mcts_simulations=args.mcts_sims,
            board_size=args.board_size,
            device=device,
        )
    else:
        # Default: play against random AI
        interactive_game(device=device)
