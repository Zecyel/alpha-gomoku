"""
Parallel Self-Play with Batched Inference.
Runs multiple games simultaneously, batching all NN evaluations together.
"""
import numpy as np
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    from game_fast import GomokuGame
except ImportError:
    from game import GomokuGame


@dataclass
class GameState:
    """State of a single game in parallel self-play."""
    game: 'GomokuGame'
    history: List[Tuple[np.ndarray, np.ndarray, int]]  # (state, policy, player)
    move_count: int
    done: bool


class ParallelMCTS:
    """
    Parallel MCTS that runs multiple games simultaneously.
    All NN evaluations are batched together for maximum GPU utilization.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        num_games: int = 16,
        num_simulations: int = 400,
        batch_size: int = 128,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature_threshold: int = 30,
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
        self.device = device

        # Detect dtype
        try:
            self.dtype = next(network.parameters()).dtype
        except StopIteration:
            self.dtype = torch.float32

    @torch.inference_mode()
    def _batch_evaluate(self, states: List[np.ndarray], valid_moves_list: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float]]:
        """Evaluate multiple states in a single batch."""
        if len(states) == 0:
            return [], []

        batch_states = torch.tensor(np.stack(states), dtype=self.dtype, device=self.device)
        batch_valid = torch.tensor(np.stack(valid_moves_list), dtype=self.dtype, device=self.device)

        self.network.eval()
        policy_logits, values = self.network(batch_states)

        policy_logits = policy_logits.float()
        batch_valid = batch_valid.float()
        policy_logits = policy_logits.masked_fill(batch_valid == 0, -1e9)
        policies = torch.softmax(policy_logits, dim=1)

        policies_np = policies.cpu().numpy()
        values_np = values.float().squeeze(-1).cpu().numpy()

        return [policies_np[i] for i in range(len(states))], values_np.tolist()

    def _run_mcts_batched(self, games: List['GomokuGame'], add_noise: bool = True) -> List[np.ndarray]:
        """Run MCTS for multiple games with batched evaluation."""
        from mcts_batched import MCTSNode

        num_games = len(games)
        board_size = games[0].board_size
        action_space = board_size * board_size

        # Initialize roots
        roots = [MCTSNode() for _ in range(num_games)]

        # Batch evaluate root states
        states = [g.get_state() for g in games]
        valid_moves = [g.get_valid_moves() for g in games]
        policies, values = self._batch_evaluate(states, valid_moves)

        # Expand roots with Dirichlet noise
        for i, (root, policy, vm) in enumerate(zip(roots, policies, valid_moves)):
            if add_noise:
                noise = np.random.dirichlet([self.dirichlet_alpha] * len(policy))
                policy = (1 - self.dirichlet_epsilon) * policy + self.dirichlet_epsilon * noise
            root.expand(policy, vm)

        # Run simulations in batches across all games
        sims_per_game = [0] * num_games

        while min(sims_per_game) < self.num_simulations:
            # Collect leaves from all games that need more simulations
            pending = []  # (game_idx, node, sim_game, state, valid_moves)
            terminal_backups = []  # (node, value)

            for game_idx in range(num_games):
                if sims_per_game[game_idx] >= self.num_simulations:
                    continue

                # How many simulations to run for this game in this batch
                sims_needed = min(
                    self.batch_size // num_games + 1,
                    self.num_simulations - sims_per_game[game_idx]
                )

                for _ in range(sims_needed):
                    node = roots[game_idx]
                    sim_game = games[game_idx].clone()

                    # Selection
                    while not node.is_leaf():
                        action, node = node.select_child(self.c_puct)
                        sim_game.step(action)

                    node.add_virtual_loss()

                    if sim_game.game_over:
                        value = 0.0 if sim_game.winner == 0 else -1.0
                        terminal_backups.append((node, value))
                    else:
                        pending.append((
                            game_idx, node, sim_game,
                            sim_game.get_state(), sim_game.get_valid_moves()
                        ))

                    sims_per_game[game_idx] += 1

            # Batch evaluate all pending leaves
            if pending:
                states = [p[3] for p in pending]
                valid_moves = [p[4] for p in pending]
                policies, values = self._batch_evaluate(states, valid_moves)

                for i, (game_idx, node, sim_game, state, vm) in enumerate(pending):
                    node.remove_virtual_loss()
                    node.expand(policies[i], vm)
                    node.backup(values[i])

            # Backup terminal nodes
            for node, value in terminal_backups:
                node.remove_virtual_loss()
                node.backup(value)

        # Collect visit counts
        all_visits = []
        for root in roots:
            visits = np.zeros(action_space)
            for action, child in root.children.items():
                visits[action] = child.visit_count
            all_visits.append(visits)

        return all_visits

    def play_games(self, board_size: int = 15, use_augmentation: bool = True) -> List:
        """Play multiple games in parallel and return training records."""
        from self_play import GameRecord

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
            ))

        # Play until all games are done
        while not all(gs.done for gs in game_states):
            # Get active games
            active_indices = [i for i, gs in enumerate(game_states) if not gs.done]
            active_games = [game_states[i].game for i in active_indices]

            if not active_games:
                break

            # Run MCTS for all active games
            all_visits = self._run_mcts_batched(active_games, add_noise=True)

            # Make moves
            for idx, game_idx in enumerate(active_indices):
                gs = game_states[game_idx]
                visits = all_visits[idx]

                # Temperature
                temperature = 1.0 if gs.move_count < self.temperature_threshold else 0.0

                # Select action
                if temperature == 0:
                    action = np.argmax(visits)
                else:
                    visits_temp = visits ** (1.0 / temperature)
                    probs = visits_temp / visits_temp.sum()
                    action = np.random.choice(len(probs), p=probs)

                # Normalize visits for policy target
                policy = visits / visits.sum()

                # Record state before move
                state = gs.game.get_state()
                player = gs.game.current_player
                gs.history.append((state.copy(), policy.copy(), player))

                # Make move
                gs.game.step(action)
                gs.move_count += 1

                if gs.game.game_over:
                    gs.done = True

        # Convert to training records
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


if __name__ == '__main__':
    import time
    from network import AlphaZeroNetwork

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    network = AlphaZeroNetwork(board_size=15, num_channels=128, num_res_blocks=10).to(device)
    network.eval()

    # Benchmark parallel self-play
    for num_games in [4, 8, 16, 32]:
        parallel = ParallelMCTS(
            network=network,
            num_games=num_games,
            num_simulations=400,
            batch_size=128,
            device=device,
        )

        start = time.perf_counter()
        records = parallel.play_games(board_size=15)
        elapsed = time.perf_counter() - start

        print(f"num_games={num_games:2d}: {elapsed:.1f}s total, {elapsed/num_games:.1f}s/game, {len(records)} records")
