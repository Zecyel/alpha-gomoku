"""
Numba-compiled MCTS using flat arrays instead of Python objects.
Much faster tree operations by avoiding Python overhead.
"""
import numpy as np
from numba import njit, prange
from numba.typed import List as NumbaList
import torch
from typing import Tuple, List
import math

try:
    from game_fast import GomokuGame
except ImportError:
    from game import GomokuGame


# Pre-allocate node arrays for MCTS tree
# Each node stores: [visit_count, value_sum, prior, parent_idx, virtual_loss]
NODE_VISIT = 0
NODE_VALUE = 1
NODE_PRIOR = 2
NODE_PARENT = 3
NODE_VLOSS = 4
NODE_FIELDS = 5

# Children stored separately: children[node_idx] = array of (action, child_idx) pairs
# Or use a flat array: children_actions[node_idx, :] and children_indices[node_idx, :]


@njit(cache=True)
def init_nodes(max_nodes: int) -> np.ndarray:
    """Initialize node storage array."""
    nodes = np.zeros((max_nodes, NODE_FIELDS), dtype=np.float32)
    nodes[:, NODE_PARENT] = -1  # -1 means no parent
    return nodes


@njit(cache=True)
def get_q_value(nodes: np.ndarray, idx: int) -> float:
    """Get Q value for a node."""
    visits = nodes[idx, NODE_VISIT] + nodes[idx, NODE_VLOSS]
    if visits == 0:
        return 0.0
    value = nodes[idx, NODE_VALUE] - nodes[idx, NODE_VLOSS]
    return value / visits


@njit(cache=True)
def select_child_ucb(
    nodes: np.ndarray,
    children_actions: np.ndarray,
    children_indices: np.ndarray,
    num_children: np.ndarray,
    node_idx: int,
    c_puct: float
) -> Tuple[int, int]:
    """Select child with highest UCB score. Returns (action, child_idx)."""
    n_children = num_children[node_idx]
    if n_children == 0:
        return -1, -1

    # Calculate total visits of children
    total_visits = 0.0
    for i in range(n_children):
        child_idx = children_indices[node_idx, i]
        total_visits += nodes[child_idx, NODE_VISIT] + nodes[child_idx, NODE_VLOSS]

    sqrt_total = math.sqrt(total_visits + 1e-8)

    best_score = -1e9
    best_action = -1
    best_child = -1

    for i in range(n_children):
        action = children_actions[node_idx, i]
        child_idx = children_indices[node_idx, i]

        q = get_q_value(nodes, child_idx)
        prior = nodes[child_idx, NODE_PRIOR]
        visits = nodes[child_idx, NODE_VISIT] + nodes[child_idx, NODE_VLOSS]

        ucb = q + c_puct * prior * sqrt_total / (1.0 + visits)

        if ucb > best_score:
            best_score = ucb
            best_action = action
            best_child = child_idx

    return best_action, best_child


@njit(cache=True)
def expand_node(
    nodes: np.ndarray,
    children_actions: np.ndarray,
    children_indices: np.ndarray,
    num_children: np.ndarray,
    node_idx: int,
    next_node_idx: int,
    policy: np.ndarray,
    valid_moves: np.ndarray,
    action_size: int
) -> int:
    """Expand a node with children. Returns next available node index."""
    # Mask and normalize policy
    masked_policy = policy * valid_moves
    policy_sum = masked_policy.sum()
    if policy_sum > 0:
        masked_policy = masked_policy / policy_sum
    else:
        masked_policy = valid_moves / valid_moves.sum()

    child_count = 0
    for action in range(action_size):
        if valid_moves[action] > 0:
            # Create child node
            child_idx = next_node_idx
            next_node_idx += 1

            nodes[child_idx, NODE_VISIT] = 0
            nodes[child_idx, NODE_VALUE] = 0
            nodes[child_idx, NODE_PRIOR] = masked_policy[action]
            nodes[child_idx, NODE_PARENT] = node_idx
            nodes[child_idx, NODE_VLOSS] = 0

            # Store in children arrays
            children_actions[node_idx, child_count] = action
            children_indices[node_idx, child_count] = child_idx
            child_count += 1

    num_children[node_idx] = child_count
    return next_node_idx


@njit(cache=True)
def add_virtual_loss(nodes: np.ndarray, node_idx: int):
    """Add virtual loss from node to root."""
    idx = node_idx
    while idx >= 0:
        nodes[idx, NODE_VLOSS] += 1
        idx = int(nodes[idx, NODE_PARENT])


@njit(cache=True)
def remove_virtual_loss(nodes: np.ndarray, node_idx: int):
    """Remove virtual loss from node to root."""
    idx = node_idx
    while idx >= 0:
        nodes[idx, NODE_VLOSS] -= 1
        idx = int(nodes[idx, NODE_PARENT])


@njit(cache=True)
def backup(nodes: np.ndarray, node_idx: int, value: float):
    """Backup value from node to root."""
    idx = node_idx
    current_value = value
    while idx >= 0:
        nodes[idx, NODE_VISIT] += 1
        nodes[idx, NODE_VALUE] += current_value
        current_value = -current_value  # Flip for opponent
        idx = int(nodes[idx, NODE_PARENT])


@njit(cache=True)
def get_visit_counts(
    children_actions: np.ndarray,
    children_indices: np.ndarray,
    num_children: np.ndarray,
    nodes: np.ndarray,
    root_idx: int,
    action_size: int
) -> np.ndarray:
    """Get visit counts for all actions from root."""
    visits = np.zeros(action_size, dtype=np.float32)
    n_children = num_children[root_idx]
    for i in range(n_children):
        action = children_actions[root_idx, i]
        child_idx = children_indices[root_idx, i]
        visits[action] = nodes[child_idx, NODE_VISIT]
    return visits


class CompiledMCTS:
    """
    Fast MCTS using Numba-compiled operations.
    Uses flat arrays instead of Python objects for tree storage.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        num_simulations: int = 400,
        batch_size: int = 32,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        max_nodes: int = 100000,
        max_children: int = 225,
        device: str = 'cpu',
    ):
        self.network = network
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.max_nodes = max_nodes
        self.max_children = max_children
        self.device = device

        # Detect dtype
        try:
            self.dtype = next(network.parameters()).dtype
        except StopIteration:
            self.dtype = torch.float32

    @torch.inference_mode()
    def _batch_evaluate(self, states: np.ndarray, valid_moves: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Batch evaluate states."""
        if len(states) == 0:
            return np.array([]), np.array([])

        batch_states = torch.tensor(states, dtype=self.dtype, device=self.device)
        batch_valid = torch.tensor(valid_moves, dtype=self.dtype, device=self.device)

        self.network.eval()
        policy_logits, values = self.network(batch_states)

        policy_logits = policy_logits.float()
        batch_valid = batch_valid.float()
        policy_logits = policy_logits.masked_fill(batch_valid == 0, -1e9)
        policies = torch.softmax(policy_logits, dim=1)

        return policies.cpu().numpy(), values.float().squeeze(-1).cpu().numpy()

    def search(self, game: GomokuGame, add_noise: bool = True) -> np.ndarray:
        """Run MCTS search and return visit counts."""
        board_size = game.board_size
        action_size = board_size * board_size

        # Initialize tree storage
        nodes = init_nodes(self.max_nodes)
        children_actions = np.zeros((self.max_nodes, self.max_children), dtype=np.int32)
        children_indices = np.zeros((self.max_nodes, self.max_children), dtype=np.int32)
        num_children = np.zeros(self.max_nodes, dtype=np.int32)

        root_idx = 0
        next_node_idx = 1

        # Evaluate root
        state = game.get_state()
        valid_moves = game.get_valid_moves()
        policies, values = self._batch_evaluate(
            state[np.newaxis, ...],
            valid_moves[np.newaxis, ...]
        )
        policy = policies[0]

        # Add Dirichlet noise
        if add_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(policy)).astype(np.float32)
            policy = (1 - self.dirichlet_epsilon) * policy + self.dirichlet_epsilon * noise

        # Expand root
        next_node_idx = expand_node(
            nodes, children_actions, children_indices, num_children,
            root_idx, next_node_idx, policy, valid_moves, action_size
        )

        # Run simulations in batches
        sims_done = 0
        while sims_done < self.num_simulations:
            batch_target = min(self.batch_size, self.num_simulations - sims_done)

            # Collect leaves
            pending_nodes = []
            pending_games = []
            pending_states = []
            pending_valid = []
            terminal_backups = []

            for _ in range(batch_target):
                node_idx = root_idx
                sim_game = game.clone()

                # Selection
                while num_children[node_idx] > 0:
                    action, child_idx = select_child_ucb(
                        nodes, children_actions, children_indices,
                        num_children, node_idx, self.c_puct
                    )
                    sim_game.step(action)
                    node_idx = child_idx

                # Add virtual loss
                add_virtual_loss(nodes, node_idx)

                if sim_game.game_over:
                    value = 0.0 if sim_game.winner == 0 else -1.0
                    terminal_backups.append((node_idx, value))
                else:
                    pending_nodes.append(node_idx)
                    pending_games.append(sim_game)
                    pending_states.append(sim_game.get_state())
                    pending_valid.append(sim_game.get_valid_moves())

            # Batch evaluate pending
            if pending_nodes:
                states_arr = np.stack(pending_states)
                valid_arr = np.stack(pending_valid)
                policies, values = self._batch_evaluate(states_arr, valid_arr)

                for i, node_idx in enumerate(pending_nodes):
                    remove_virtual_loss(nodes, node_idx)
                    next_node_idx = expand_node(
                        nodes, children_actions, children_indices, num_children,
                        node_idx, next_node_idx, policies[i], valid_arr[i], action_size
                    )
                    backup(nodes, node_idx, values[i])

            # Backup terminals
            for node_idx, value in terminal_backups:
                remove_virtual_loss(nodes, node_idx)
                backup(nodes, node_idx, value)

            sims_done += batch_target

        # Return visit counts
        return get_visit_counts(
            children_actions, children_indices, num_children,
            nodes, root_idx, action_size
        )

    def select_action(self, game: GomokuGame, temperature: float = 1.0, add_noise: bool = True) -> Tuple[int, np.ndarray]:
        """Select action using MCTS."""
        visits = self.search(game, add_noise)

        # Normalize
        visits_sum = visits.sum()
        if visits_sum > 0:
            pi = visits / visits_sum
        else:
            pi = visits

        if temperature == 0:
            action = np.argmax(visits)
        else:
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / visits_temp.sum()
            action = np.random.choice(len(probs), p=probs)

        return action, pi


# Alias
MCTS = CompiledMCTS


if __name__ == '__main__':
    import time
    from network import AlphaZeroNetwork

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Warmup JIT
    print("Warming up Numba JIT...")
    nodes = init_nodes(1000)
    children_actions = np.zeros((1000, 225), dtype=np.int32)
    children_indices = np.zeros((1000, 225), dtype=np.int32)
    num_children = np.zeros(1000, dtype=np.int32)
    policy = np.random.rand(225).astype(np.float32)
    valid = np.ones(225, dtype=np.float32)
    expand_node(nodes, children_actions, children_indices, num_children, 0, 1, policy, valid, 225)
    select_child_ucb(nodes, children_actions, children_indices, num_children, 0, 1.5)
    backup(nodes, 1, 0.5)

    print("\nBenchmarking CompiledMCTS...")
    network = AlphaZeroNetwork(board_size=15, num_channels=128, num_res_blocks=10).to(device)
    network.eval()

    game = GomokuGame(board_size=15)

    for batch_size in [8, 16, 32, 64]:
        mcts = CompiledMCTS(network, num_simulations=400, batch_size=batch_size, device=device)

        # Warmup
        mcts.select_action(game, temperature=1.0)

        start = time.perf_counter()
        for _ in range(3):
            game.reset()
            mcts.select_action(game, temperature=1.0)
        elapsed = (time.perf_counter() - start) / 3

        print(f"  batch_size={batch_size:2d}: {elapsed:.2f}s per move ({400/elapsed:.0f} sims/sec)")
