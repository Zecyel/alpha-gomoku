"""
Parallel Self-Play with Compiled MCTS.
Uses Numba-compiled tree operations for maximum speed.
"""
import numpy as np
import torch
from numba import njit
from typing import List, Tuple
from dataclasses import dataclass

try:
    from game_fast import GomokuGame
except ImportError:
    from game import GomokuGame

from mcts_compiled import (
    init_nodes, expand_node, select_child_ucb,
    add_virtual_loss, remove_virtual_loss, backup, get_visit_counts,
    NODE_VISIT, NODE_VALUE, NODE_PRIOR, NODE_PARENT, NODE_VLOSS
)


@dataclass
class GameState:
    """State of a single game."""
    game: GomokuGame
    history: List[Tuple[np.ndarray, np.ndarray, int]]
    move_count: int
    done: bool
    # MCTS tree storage
    nodes: np.ndarray
    children_actions: np.ndarray
    children_indices: np.ndarray
    num_children: np.ndarray
    next_node_idx: int


class CompiledParallelMCTS:
    """
    Parallel MCTS using Numba-compiled operations.
    Runs multiple games with independent MCTS trees, batching NN evaluations.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        num_games: int = 16,
        num_simulations: int = 400,
        batch_size: int = 64,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature_threshold: int = 30,
        max_nodes: int = 50000,
        max_children: int = 225,
        device: str = 'cpu',
    ):
        self.network = network
        self.num_games = num_games
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature_threshold = temperature_threshold
        self.max_nodes = max_nodes
        self.max_children = max_children
        self.device = device

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

    def _run_mcts_for_games(self, game_states: List[GameState], action_size: int) -> List[np.ndarray]:
        """Run MCTS for all active games and return visit counts."""
        active_games = [gs for gs in game_states if not gs.done]
        if not active_games:
            return []

        # Initialize trees for each game
        for gs in active_games:
            gs.nodes = init_nodes(self.max_nodes)
            gs.children_actions = np.zeros((self.max_nodes, self.max_children), dtype=np.int32)
            gs.children_indices = np.zeros((self.max_nodes, self.max_children), dtype=np.int32)
            gs.num_children = np.zeros(self.max_nodes, dtype=np.int32)
            gs.next_node_idx = 1  # 0 is root

        # Evaluate roots
        states = np.stack([gs.game.get_state() for gs in active_games])
        valids = np.stack([gs.game.get_valid_moves() for gs in active_games])
        policies, _ = self._batch_evaluate(states, valids)

        # Expand roots with Dirichlet noise
        for i, gs in enumerate(active_games):
            policy = policies[i].copy()
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(policy)).astype(np.float32)
            policy = (1 - self.dirichlet_epsilon) * policy + self.dirichlet_epsilon * noise

            gs.next_node_idx = expand_node(
                gs.nodes, gs.children_actions, gs.children_indices, gs.num_children,
                0, gs.next_node_idx, policy, valids[i], action_size
            )

        # Run simulations
        sims_done = 0
        while sims_done < self.num_simulations:
            batch_target = min(self.batch_size, self.num_simulations - sims_done)
            sims_per_game = max(1, batch_target // len(active_games))

            pending = []  # (game_idx, node_idx, sim_game, state, valid)
            terminal_backups = []  # (game_idx, node_idx, value)

            for game_idx, gs in enumerate(active_games):
                for _ in range(sims_per_game):
                    node_idx = 0
                    sim_game = gs.game.clone()

                    # Selection
                    while gs.num_children[node_idx] > 0:
                        action, child_idx = select_child_ucb(
                            gs.nodes, gs.children_actions, gs.children_indices,
                            gs.num_children, node_idx, self.c_puct
                        )
                        sim_game.step(action)
                        node_idx = child_idx

                    add_virtual_loss(gs.nodes, node_idx)

                    if sim_game.game_over:
                        value = 0.0 if sim_game.winner == 0 else -1.0
                        terminal_backups.append((game_idx, node_idx, value))
                    else:
                        pending.append((
                            game_idx, node_idx, sim_game,
                            sim_game.get_state(), sim_game.get_valid_moves()
                        ))

            # Batch evaluate
            if pending:
                states = np.stack([p[3] for p in pending])
                valids = np.stack([p[4] for p in pending])
                policies, values = self._batch_evaluate(states, valids)

                for i, (game_idx, node_idx, sim_game, state, valid) in enumerate(pending):
                    gs = active_games[game_idx]
                    remove_virtual_loss(gs.nodes, node_idx)
                    gs.next_node_idx = expand_node(
                        gs.nodes, gs.children_actions, gs.children_indices, gs.num_children,
                        node_idx, gs.next_node_idx, policies[i], valid, action_size
                    )
                    backup(gs.nodes, node_idx, values[i])

            # Backup terminals
            for game_idx, node_idx, value in terminal_backups:
                gs = active_games[game_idx]
                remove_virtual_loss(gs.nodes, node_idx)
                backup(gs.nodes, node_idx, value)

            sims_done += len(pending) + len(terminal_backups)

        # Collect visit counts
        all_visits = []
        for gs in active_games:
            visits = get_visit_counts(
                gs.children_actions, gs.children_indices, gs.num_children,
                gs.nodes, 0, action_size
            )
            all_visits.append(visits)

        return all_visits

    def play_games(self, board_size: int = 15, use_augmentation: bool = True) -> List:
        """Play multiple games in parallel."""
        from self_play import GameRecord

        action_size = board_size * board_size

        # Initialize games
        game_states = []
        for _ in range(self.num_games):
            game = GomokuGame(board_size=board_size)
            game.reset()
            game_states.append(GameState(
                game=game,
                history=[],
                move_count=0,
                done=False,
                nodes=None,
                children_actions=None,
                children_indices=None,
                num_children=None,
                next_node_idx=1,
            ))

        # Play until all done
        while not all(gs.done for gs in game_states):
            active_indices = [i for i, gs in enumerate(game_states) if not gs.done]
            active_games = [game_states[i] for i in active_indices]

            # Run MCTS
            all_visits = self._run_mcts_for_games(active_games, action_size)

            # Make moves
            for gs, visits in zip(active_games, all_visits):
                temperature = 1.0 if gs.move_count < self.temperature_threshold else 0.0

                if temperature == 0:
                    action = np.argmax(visits)
                else:
                    visits_temp = visits ** (1.0 / temperature)
                    probs = visits_temp / (visits_temp.sum() + 1e-8)
                    action = np.random.choice(len(probs), p=probs)

                policy = visits / (visits.sum() + 1e-8)

                # Record
                state = gs.game.get_state()
                player = gs.game.current_player
                gs.history.append((state.copy(), policy.copy(), player))

                # Step
                gs.game.step(action)
                gs.move_count += 1

                if gs.game.game_over:
                    gs.done = True

        # Convert to records
        all_records = []
        for gs in game_states:
            game = gs.game
            for state, policy, player in gs.history:
                if game.winner == 0:
                    value = 0.0
                elif game.winner == player:
                    value = 1.0
                else:
                    value = -1.0

                if use_augmentation:
                    symmetries = game.get_symmetries(state, policy)
                    for sym_state, sym_policy in symmetries:
                        all_records.append(GameRecord(
                            state=sym_state,
                            policy=sym_policy,
                            value=value,
                        ))
                else:
                    all_records.append(GameRecord(
                        state=state,
                        policy=policy,
                        value=value,
                    ))

        return all_records


# Alias
ParallelMCTS = CompiledParallelMCTS


if __name__ == '__main__':
    import time
    from network import AlphaZeroNetwork

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    network = AlphaZeroNetwork(board_size=15, num_channels=128, num_res_blocks=10).to(device)
    network.eval()

    print("\nBenchmarking CompiledParallelMCTS...")
    for num_games in [8, 16, 32]:
        parallel = CompiledParallelMCTS(
            network=network,
            num_games=num_games,
            num_simulations=400,
            batch_size=64,
            device=device,
        )

        # Warmup
        parallel.play_games(board_size=15, use_augmentation=False)

        start = time.perf_counter()
        records = parallel.play_games(board_size=15, use_augmentation=False)
        elapsed = time.perf_counter() - start

        print(f"  num_games={num_games:2d}: {elapsed:.1f}s total, {elapsed/num_games:.2f}s/game, {len(records)} records")
