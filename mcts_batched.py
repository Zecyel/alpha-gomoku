"""
Batched Monte Carlo Tree Search with Virtual Loss.
Collects multiple leaf nodes and evaluates them in a single batch for massive speedup.
"""
import math
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from game_fast import GomokuGame
except ImportError:
    from game import GomokuGame


class MCTSNode:
    """A node in the MCTS tree with virtual loss support."""

    __slots__ = ['prior', 'parent', 'children', 'visit_count', 'value_sum', 'virtual_loss']

    def __init__(self, prior: float = 0.0, parent: Optional['MCTSNode'] = None):
        self.prior = prior
        self.parent = parent
        self.children: Dict[int, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_loss = 0  # Virtual loss for parallel MCTS

    @property
    def q_value(self) -> float:
        """Q(s, a) with virtual loss applied."""
        total_visits = self.visit_count + self.virtual_loss
        if total_visits == 0:
            return 0.0
        # Virtual loss counts as losses (-1)
        return (self.value_sum - self.virtual_loss) / total_visits

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def select_child(self, c_puct: float) -> Tuple[int, 'MCTSNode']:
        """Select child with highest UCB score (virtual loss aware)."""
        total_visits = sum(c.visit_count + c.virtual_loss for c in self.children.values())
        sqrt_total = math.sqrt(total_visits + 1e-8)

        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in self.children.items():
            effective_visits = child.visit_count + child.virtual_loss
            ucb = child.q_value + c_puct * child.prior * sqrt_total / (1 + effective_visits)

            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, policy: np.ndarray, valid_moves: np.ndarray):
        """Expand node with children."""
        masked_policy = policy * valid_moves
        policy_sum = masked_policy.sum()
        if policy_sum > 0:
            masked_policy /= policy_sum
        else:
            masked_policy = valid_moves / valid_moves.sum()

        for action in range(len(policy)):
            if valid_moves[action] > 0:
                self.children[action] = MCTSNode(prior=masked_policy[action], parent=self)

    def add_virtual_loss(self):
        """Add virtual loss along path to root."""
        node = self
        while node is not None:
            node.virtual_loss += 1
            node = node.parent

    def remove_virtual_loss(self):
        """Remove virtual loss along path to root."""
        node = self
        while node is not None:
            node.virtual_loss -= 1
            node = node.parent

    def backup(self, value: float):
        """Backpropagate value up the tree."""
        node = self
        current_value = value
        while node is not None:
            node.visit_count += 1
            node.value_sum += current_value
            current_value = -current_value
            node = node.parent


@dataclass
class PendingEval:
    """Pending leaf evaluation."""
    node: MCTSNode
    game: GomokuGame
    state: np.ndarray
    valid_moves: np.ndarray


class BatchedMCTS:
    """
    Batched MCTS with virtual loss for parallel leaf evaluation.

    Instead of evaluating one leaf at a time, we:
    1. Traverse to multiple leaves (using virtual loss to diversify)
    2. Batch all leaf states together
    3. Do ONE neural network forward pass
    4. Expand all leaves and backpropagate
    """

    def __init__(
        self,
        network: torch.nn.Module,
        num_simulations: int = 800,
        batch_size: int = 8,  # Number of leaves to batch together
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        device: str = 'cpu',
    ):
        self.network = network
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.device = device

    @torch.inference_mode()
    def _batch_evaluate(self, states: List[np.ndarray], valid_moves_list: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float]]:
        """Evaluate multiple states in a single batch."""
        if len(states) == 0:
            return [], []

        # Detect network dtype
        try:
            dtype = next(self.network.parameters()).dtype
        except StopIteration:
            dtype = torch.float32

        # Stack states into batch
        batch_states = torch.tensor(np.stack(states), dtype=dtype, device=self.device)
        batch_valid = torch.tensor(np.stack(valid_moves_list), dtype=dtype, device=self.device)

        # Single forward pass for entire batch
        self.network.eval()
        policy_logits, values = self.network(batch_states)

        # Mask invalid moves and softmax (use float32 for numerical stability)
        policy_logits = policy_logits.float()
        batch_valid = batch_valid.float()
        policy_logits = policy_logits.masked_fill(batch_valid == 0, -1e9)
        policies = torch.softmax(policy_logits, dim=1)

        # Convert to numpy
        policies_np = policies.cpu().numpy()
        values_np = values.float().squeeze(-1).cpu().numpy()

        return [policies_np[i] for i in range(len(states))], values_np.tolist()

    def search(self, game: GomokuGame, add_noise: bool = True) -> np.ndarray:
        """Run batched MCTS search."""
        root = MCTSNode()
        board_size = game.board_size
        action_space = board_size * board_size

        # Evaluate and expand root
        state = game.get_state()
        valid_moves = game.get_valid_moves()
        policies, values = self._batch_evaluate([state], [valid_moves])
        policy = policies[0]

        # Add Dirichlet noise at root
        if add_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(policy))
            policy = (1 - self.dirichlet_epsilon) * policy + self.dirichlet_epsilon * noise

        root.expand(policy, valid_moves)

        # Run simulations in batches
        sims_done = 0
        while sims_done < self.num_simulations:
            # Collect batch of leaves
            pending: List[PendingEval] = []
            terminal_backups: List[Tuple[MCTSNode, float]] = []

            batch_target = min(self.batch_size, self.num_simulations - sims_done)

            for _ in range(batch_target):
                node = root
                sim_game = game.clone()

                # Selection: traverse to leaf with virtual loss
                while not node.is_leaf():
                    action, node = node.select_child(self.c_puct)
                    sim_game.step(action)

                # Add virtual loss to discourage same path
                node.add_virtual_loss()

                # Check terminal
                if sim_game.game_over:
                    if sim_game.winner == 0:
                        value = 0.0
                    else:
                        value = -1.0  # Current player lost
                    terminal_backups.append((node, value))
                else:
                    # Queue for batch evaluation
                    pending.append(PendingEval(
                        node=node,
                        game=sim_game,
                        state=sim_game.get_state(),
                        valid_moves=sim_game.get_valid_moves(),
                    ))

            # Batch evaluate all pending leaves
            if pending:
                states = [p.state for p in pending]
                valid_moves_list = [p.valid_moves for p in pending]
                policies, values = self._batch_evaluate(states, valid_moves_list)

                # Expand and backup each leaf
                for i, p in enumerate(pending):
                    p.node.remove_virtual_loss()
                    p.node.expand(policies[i], p.valid_moves)
                    p.node.backup(values[i])

            # Backup terminal nodes
            for node, value in terminal_backups:
                node.remove_virtual_loss()
                node.backup(value)

            sims_done += batch_target

        # Return visit counts
        visits = np.zeros(action_space)
        for action, child in root.children.items():
            visits[action] = child.visit_count

        return visits

    def get_action_probs(self, game: GomokuGame, temperature: float = 1.0, add_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Get action probabilities from MCTS."""
        visits = self.search(game, add_noise=add_noise)

        if temperature == 0:
            best_action = np.argmax(visits)
            action_probs = np.zeros_like(visits)
            action_probs[best_action] = 1.0
        else:
            visits_temp = visits ** (1.0 / temperature)
            total = visits_temp.sum()
            if total > 0:
                action_probs = visits_temp / total
            else:
                action_probs = visits_temp

        visit_counts = visits / visits.sum() if visits.sum() > 0 else visits
        return action_probs, visit_counts

    def select_action(self, game: GomokuGame, temperature: float = 1.0, add_noise: bool = True) -> Tuple[int, np.ndarray]:
        """Select an action using MCTS."""
        action_probs, pi = self.get_action_probs(game, temperature, add_noise)

        if temperature == 0:
            action = np.argmax(action_probs)
        else:
            action = np.random.choice(len(action_probs), p=action_probs)

        return action, pi


# Alias for compatibility
MCTS = BatchedMCTS


if __name__ == '__main__':
    import time
    from network import AlphaZeroNetwork

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    board_size = 15
    game = GomokuGame(board_size=board_size)
    network = AlphaZeroNetwork(board_size=board_size, num_channels=128, num_res_blocks=10).to(device)
    network.eval()

    # Benchmark batched MCTS
    print("\nBenchmarking BatchedMCTS...")
    for batch_size in [1, 4, 8, 16, 32]:
        mcts = BatchedMCTS(network, num_simulations=400, batch_size=batch_size, device=device)

        # Warmup
        mcts.select_action(game, temperature=1.0)

        start = time.perf_counter()
        for _ in range(3):
            game.reset()
            mcts.select_action(game, temperature=1.0)
        elapsed = (time.perf_counter() - start) / 3

        print(f"  batch_size={batch_size:2d}: {elapsed:.2f}s per move ({400/elapsed:.0f} sims/sec)")
