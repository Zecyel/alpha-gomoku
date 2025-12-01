"""
Fast Gomoku Game Environment using Numba JIT compilation.
"""
import numpy as np
from numba import njit, int8, int32, boolean
from numba.experimental import jitclass
from typing import Optional, Tuple, List

# Numba-compiled core functions
@njit(cache=True)
def check_win_fast(board: np.ndarray, row: int, col: int, board_size: int, win_length: int) -> bool:
    """Fast win check using Numba."""
    player = board[row, col]
    if player == 0:
        return False

    # Directions: horizontal, vertical, diagonal, anti-diagonal
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))

    for dr, dc in directions:
        count = 1
        # Positive direction
        for i in range(1, win_length):
            r, c = row + dr * i, col + dc * i
            if 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
                count += 1
            else:
                break
        # Negative direction
        for i in range(1, win_length):
            r, c = row - dr * i, col - dc * i
            if 0 <= r < board_size and 0 <= c < board_size and board[r, c] == player:
                count += 1
            else:
                break

        if count >= win_length:
            return True

    return False


@njit(cache=True)
def get_valid_moves_fast(board: np.ndarray) -> np.ndarray:
    """Fast valid moves calculation."""
    return (board.flatten() == 0).astype(np.float32)


@njit(cache=True)
def get_state_fast(board: np.ndarray, current_player: int, last_move_row: int, last_move_col: int, board_size: int) -> np.ndarray:
    """Fast state representation."""
    state = np.zeros((4, board_size, board_size), dtype=np.float32)

    for i in range(board_size):
        for j in range(board_size):
            if board[i, j] == current_player:
                state[0, i, j] = 1.0
            elif board[i, j] == -current_player:
                state[1, i, j] = 1.0

    if last_move_row >= 0 and last_move_col >= 0:
        state[2, last_move_row, last_move_col] = 1.0

    if current_player == 1:
        state[3, :, :] = 1.0

    return state


@njit(cache=True)
def apply_symmetry_fast(state: np.ndarray, policy: np.ndarray, sym_idx: int, board_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Apply one of 8 symmetries (4 rotations x 2 reflections)."""
    policy_2d = policy.reshape(board_size, board_size)

    # Number of 90-degree rotations
    rot = sym_idx // 2
    do_flip = sym_idx % 2 == 1

    # Apply rotation to each channel of state
    new_state = np.empty_like(state)
    new_policy = policy_2d.copy()

    for c in range(state.shape[0]):
        channel = state[c]
        for _ in range(rot):
            # Manual 90-degree rotation: new[i,j] = old[n-1-j, i]
            rotated = np.empty_like(channel)
            n = channel.shape[0]
            for i in range(n):
                for j in range(n):
                    rotated[i, j] = channel[n - 1 - j, i]
            channel = rotated
        new_state[c] = channel

    # Rotate policy
    for _ in range(rot):
        rotated_policy = np.empty_like(new_policy)
        n = new_policy.shape[0]
        for i in range(n):
            for j in range(n):
                rotated_policy[i, j] = new_policy[n - 1 - j, i]
        new_policy = rotated_policy

    # Apply horizontal flip if needed
    if do_flip:
        for c in range(new_state.shape[0]):
            new_state[c] = new_state[c, :, ::-1].copy()
        new_policy = new_policy[:, ::-1].copy()

    return new_state, new_policy.flatten()


class GomokuGameFast:
    """
    Fast Gomoku game using Numba-accelerated functions.
    """

    def __init__(self, board_size: int = 15, win_length: int = 5):
        self.board_size = board_size
        self.win_length = win_length
        self.reset()

    def reset(self) -> np.ndarray:
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.last_move_row = -1
        self.last_move_col = -1
        self.game_over = False
        self.winner = 0
        return self.get_state()

    def get_state(self) -> np.ndarray:
        return get_state_fast(self.board, self.current_player,
                              self.last_move_row, self.last_move_col, self.board_size)

    def get_valid_moves(self) -> np.ndarray:
        return get_valid_moves_fast(self.board)

    def get_valid_moves_list(self) -> List[int]:
        return np.where(self.board.flatten() == 0)[0].tolist()

    def action_to_coord(self, action: int) -> Tuple[int, int]:
        return (action // self.board_size, action % self.board_size)

    def coord_to_action(self, row: int, col: int) -> int:
        return row * self.board_size + col

    def is_valid_move(self, action: int) -> bool:
        if action < 0 or action >= self.board_size * self.board_size:
            return False
        row, col = self.action_to_coord(action)
        return self.board[row, col] == 0

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        if self.game_over:
            raise ValueError("Game is already over")

        row, col = self.action_to_coord(action)
        self.board[row, col] = self.current_player
        self.last_move_row = row
        self.last_move_col = col

        if check_win_fast(self.board, row, col, self.board_size, self.win_length):
            self.game_over = True
            self.winner = self.current_player
            reward = 1.0
        elif np.all(self.board != 0):
            self.game_over = True
            self.winner = 0
            reward = 0.0
        else:
            reward = 0.0

        self.current_player = -self.current_player
        return self.get_state(), reward, self.game_over

    def clone(self) -> 'GomokuGameFast':
        game = GomokuGameFast.__new__(GomokuGameFast)
        game.board_size = self.board_size
        game.win_length = self.win_length
        game.board = self.board.copy()
        game.current_player = self.current_player
        game.last_move_row = self.last_move_row
        game.last_move_col = self.last_move_col
        game.game_over = self.game_over
        game.winner = self.winner
        return game

    def get_symmetries(self, state: np.ndarray, policy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        symmetries = []
        for i in range(8):
            sym_state, sym_policy = apply_symmetry_fast(state, policy, i, self.board_size)
            symmetries.append((sym_state, sym_policy))
        return symmetries

    def render(self) -> str:
        symbols = {0: '.', 1: 'X', -1: 'O'}
        lines = []
        header = '   ' + ' '.join(f'{i:2d}' for i in range(self.board_size))
        lines.append(header)
        for i in range(self.board_size):
            row_str = f'{i:2d} '
            row_str += ' '.join(f' {symbols[self.board[i, j]]}' for j in range(self.board_size))
            lines.append(row_str)
        return '\n'.join(lines)

    def __str__(self) -> str:
        return self.render()


# Alias for compatibility
GomokuGame = GomokuGameFast


if __name__ == '__main__':
    import time

    # Benchmark
    print("Warming up Numba JIT...")
    game = GomokuGameFast()
    for _ in range(10):
        game.reset()
        game.step(112)
        game.clone()
        game.get_state()

    print("\nBenchmarking...")

    # Test clone speed
    game = GomokuGameFast()
    start = time.perf_counter()
    for _ in range(100000):
        game.clone()
    elapsed = time.perf_counter() - start
    print(f"Clone: {100000/elapsed:.0f} ops/sec")

    # Test step speed
    start = time.perf_counter()
    for _ in range(10000):
        game.reset()
        for action in [112, 113, 127, 128, 142]:
            game.step(action)
    elapsed = time.perf_counter() - start
    print(f"Step: {50000/elapsed:.0f} ops/sec")

    # Test get_state speed
    game.reset()
    start = time.perf_counter()
    for _ in range(100000):
        game.get_state()
    elapsed = time.perf_counter() - start
    print(f"Get state: {100000/elapsed:.0f} ops/sec")

    # Test win check
    game.reset()
    game.step(112)
    start = time.perf_counter()
    for _ in range(100000):
        check_win_fast(game.board, 7, 7, 15, 5)
    elapsed = time.perf_counter() - start
    print(f"Win check: {100000/elapsed:.0f} ops/sec")
