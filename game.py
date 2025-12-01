"""
Gomoku Game Environment (无禁手五子棋)
No forbidden moves - standard 15x15 board, 5 in a row wins
"""
import numpy as np
from typing import Optional, Tuple, List


class GomokuGame:
    """
    Gomoku game environment.
    Board representation:
        0 = empty
        1 = black (first player)
        -1 = white (second player)
    """

    def __init__(self, board_size: int = 15, win_length: int = 5):
        self.board_size = board_size
        self.win_length = win_length
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the game to initial state."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1  # Black plays first
        self.last_move: Optional[Tuple[int, int]] = None
        self.game_over = False
        self.winner = 0  # 0 = ongoing/draw, 1 = black wins, -1 = white wins
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """
        Get the current state representation for neural network.
        Returns shape: (4, board_size, board_size)
        - Channel 0: current player's stones
        - Channel 1: opponent's stones
        - Channel 2: last move position (1 at last move, 0 elsewhere)
        - Channel 3: current player indicator (all 1s if black, all 0s if white)
        """
        state = np.zeros((4, self.board_size, self.board_size), dtype=np.float32)

        # Current player's stones
        state[0] = (self.board == self.current_player).astype(np.float32)
        # Opponent's stones
        state[1] = (self.board == -self.current_player).astype(np.float32)
        # Last move
        if self.last_move is not None:
            state[2, self.last_move[0], self.last_move[1]] = 1.0
        # Current player indicator
        if self.current_player == 1:
            state[3] = 1.0

        return state

    def get_valid_moves(self) -> np.ndarray:
        """Get a binary mask of valid moves."""
        return (self.board == 0).astype(np.float32).flatten()

    def get_valid_moves_list(self) -> List[int]:
        """Get list of valid move indices."""
        return np.where(self.board.flatten() == 0)[0].tolist()

    def action_to_coord(self, action: int) -> Tuple[int, int]:
        """Convert action index to board coordinates."""
        return (action // self.board_size, action % self.board_size)

    def coord_to_action(self, row: int, col: int) -> int:
        """Convert board coordinates to action index."""
        return row * self.board_size + col

    def is_valid_move(self, action: int) -> bool:
        """Check if an action is valid."""
        if action < 0 or action >= self.board_size * self.board_size:
            return False
        row, col = self.action_to_coord(action)
        return self.board[row, col] == 0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute a move.

        Args:
            action: Move index (0 to board_size^2 - 1)

        Returns:
            state: New state after the move
            reward: 1 if current player wins, -1 if loses, 0 otherwise
            done: Whether the game is over
        """
        if self.game_over:
            raise ValueError("Game is already over")

        if not self.is_valid_move(action):
            raise ValueError(f"Invalid move: {action}")

        row, col = self.action_to_coord(action)
        self.board[row, col] = self.current_player
        self.last_move = (row, col)

        # Check for win
        if self._check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
            reward = 1.0  # Current player wins
        # Check for draw (board full)
        elif np.all(self.board != 0):
            self.game_over = True
            self.winner = 0
            reward = 0.0
        else:
            reward = 0.0

        # Switch player
        self.current_player = -self.current_player

        return self.get_state(), reward, self.game_over

    def _check_win(self, row: int, col: int) -> bool:
        """Check if the last move at (row, col) results in a win."""
        player = self.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horizontal, vertical, diagonal

        for dr, dc in directions:
            count = 1
            # Count in positive direction
            for i in range(1, self.win_length):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break
            # Count in negative direction
            for i in range(1, self.win_length):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break

            if count >= self.win_length:
                return True

        return False

    def clone(self) -> 'GomokuGame':
        """Create a deep copy of the game state."""
        game = GomokuGame(self.board_size, self.win_length)
        game.board = self.board.copy()
        game.current_player = self.current_player
        game.last_move = self.last_move
        game.game_over = self.game_over
        game.winner = self.winner
        return game

    def get_symmetries(self, state: np.ndarray, policy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get all 8 symmetries (rotations and reflections) of the state and policy.
        This is used for data augmentation during training.
        """
        symmetries = []
        policy_2d = policy.reshape(self.board_size, self.board_size)

        for i in range(4):
            # Rotate
            rotated_state = np.rot90(state, i, axes=(1, 2))
            rotated_policy = np.rot90(policy_2d, i)
            symmetries.append((rotated_state, rotated_policy.flatten()))

            # Reflect and rotate
            flipped_state = np.flip(rotated_state, axis=2)
            flipped_policy = np.flip(rotated_policy, axis=1)
            symmetries.append((flipped_state, flipped_policy.flatten()))

        return symmetries

    def render(self) -> str:
        """Return a string representation of the board."""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        lines = []

        # Column headers
        header = '   ' + ' '.join(f'{i:2d}' for i in range(self.board_size))
        lines.append(header)

        for i in range(self.board_size):
            row_str = f'{i:2d} '
            row_str += ' '.join(f' {symbols[self.board[i, j]]}' for j in range(self.board_size))
            lines.append(row_str)

        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.render()


if __name__ == '__main__':
    # Test the game
    game = GomokuGame()
    print(game)
    print(f"\nValid moves: {len(game.get_valid_moves_list())}")

    # Play a few moves
    moves = [112, 113, 127, 128, 142, 143, 157, 158, 172]  # Diagonal win for black
    for move in moves:
        state, reward, done = game.step(move)
        print(f"\nAfter move {move}:")
        print(game)
        if done:
            print(f"Game over! Winner: {'Black' if game.winner == 1 else 'White' if game.winner == -1 else 'Draw'}")
            break
