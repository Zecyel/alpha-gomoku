"""
Interactive Gomoku GUI for playing against the AI.
Uses tkinter for cross-platform compatibility.
"""
import tkinter as tk
from tkinter import messagebox, ttk
import threading
import numpy as np
import torch
from typing import Optional

try:
    from game_fast import GomokuGame
except ImportError:
    from game import GomokuGame
from network import AlphaZeroNetwork
from mcts import MCTS


class GomokuGUI:
    """Interactive Gomoku GUI."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        board_size: int = 15,
        cell_size: int = 40,
        mcts_sims: int = 800,
        mcts_batch: int = 16,
    ):
        self.board_size = board_size
        self.cell_size = cell_size
        self.mcts_sims = mcts_sims
        self.mcts_batch = mcts_batch
        self.model_path = model_path

        # Game state
        self.game = GomokuGame(board_size=board_size)
        self.game.reset()
        self.human_player = 1  # Human plays black by default
        self.ai_thinking = False

        # Setup device and load model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.network = None
        self.mcts = None
        if model_path:
            self.load_model(model_path)

        # Create GUI
        self.root = tk.Tk()
        self.root.title("Gomoku - AlphaZero")
        self.root.resizable(False, False)

        self._create_widgets()
        self._draw_board()

    def load_model(self, path: str):
        """Load a trained model."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            config = checkpoint.get('config', {})
            state_dict = checkpoint['model_state_dict']

            # Infer network architecture
            num_channels = state_dict['input_conv.weight'].shape[0]
            res_block_keys = [k for k in state_dict.keys() if k.startswith('res_blocks')]
            num_res_blocks = len(set(k.split('.')[1] for k in res_block_keys))

            self.network = AlphaZeroNetwork(
                board_size=self.board_size,
                num_channels=num_channels,
                num_res_blocks=num_res_blocks,
            ).to(self.device)

            # Handle compiled model state dict
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('_orig_mod.', '')
                new_state_dict[new_key] = v
            self.network.load_state_dict(new_state_dict)
            self.network.eval()

            # Check if bf16
            if next(self.network.parameters()).dtype == torch.bfloat16:
                pass  # Already bf16
            elif 'cuda' in self.device:
                self.network = self.network.to(torch.bfloat16)

            self.mcts = MCTS(
                self.network,
                num_simulations=self.mcts_sims,
                batch_size=self.mcts_batch,
                device=self.device,
            )
            print(f"Loaded model from {path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.network = None
            self.mcts = None

    def _create_widgets(self):
        """Create GUI widgets."""
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)

        # Canvas for board
        canvas_size = self.cell_size * (self.board_size + 1)
        self.canvas = tk.Canvas(
            main_frame,
            width=canvas_size,
            height=canvas_size,
            bg='#DEB887',  # Burlywood color
        )
        self.canvas.pack()
        self.canvas.bind('<Button-1>', self._on_click)

        # Control frame
        control_frame = tk.Frame(main_frame)
        control_frame.pack(pady=10)

        # New game button
        self.new_game_btn = tk.Button(
            control_frame,
            text="New Game",
            command=self._new_game,
            width=12,
        )
        self.new_game_btn.pack(side=tk.LEFT, padx=5)

        # Switch sides button
        self.switch_btn = tk.Button(
            control_frame,
            text="Switch Sides",
            command=self._switch_sides,
            width=12,
        )
        self.switch_btn.pack(side=tk.LEFT, padx=5)

        # Load model button
        self.load_btn = tk.Button(
            control_frame,
            text="Load Model",
            command=self._load_model_dialog,
            width=12,
        )
        self.load_btn.pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_var = tk.StringVar()
        self.status_var.set("Your turn (Black)")
        self.status_label = tk.Label(
            main_frame,
            textvariable=self.status_var,
            font=('Arial', 12),
        )
        self.status_label.pack(pady=5)

        # Info label
        info_text = f"Model: {self.model_path or 'None'} | Device: {self.device}"
        self.info_label = tk.Label(main_frame, text=info_text, font=('Arial', 9), fg='gray')
        self.info_label.pack()

    def _draw_board(self):
        """Draw the game board."""
        self.canvas.delete('all')

        margin = self.cell_size
        board_pixel_size = self.cell_size * (self.board_size - 1)

        # Draw grid lines
        for i in range(self.board_size):
            # Vertical lines
            x = margin + i * self.cell_size
            self.canvas.create_line(x, margin, x, margin + board_pixel_size, fill='black')
            # Horizontal lines
            y = margin + i * self.cell_size
            self.canvas.create_line(margin, y, margin + board_pixel_size, y, fill='black')

        # Draw star points (for 15x15 board)
        if self.board_size == 15:
            star_points = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7),
                          (3, 7), (11, 7), (7, 3), (7, 11)]
            for r, c in star_points:
                x = margin + c * self.cell_size
                y = margin + r * self.cell_size
                self.canvas.create_oval(x-3, y-3, x+3, y+3, fill='black')

        # Draw coordinates
        for i in range(self.board_size):
            # Column labels (A-O)
            x = margin + i * self.cell_size
            self.canvas.create_text(x, margin - 15, text=chr(ord('A') + i), font=('Arial', 9))
            # Row labels (1-15)
            y = margin + i * self.cell_size
            self.canvas.create_text(margin - 15, y, text=str(i + 1), font=('Arial', 9))

        # Draw stones
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.game.board[row, col] != 0:
                    self._draw_stone(row, col, self.game.board[row, col])

        # Highlight last move
        if self.game.last_move_row >= 0:
            self._highlight_last_move(self.game.last_move_row, self.game.last_move_col)

    def _draw_stone(self, row: int, col: int, player: int):
        """Draw a stone on the board."""
        margin = self.cell_size
        x = margin + col * self.cell_size
        y = margin + row * self.cell_size
        r = self.cell_size // 2 - 2

        color = 'black' if player == 1 else 'white'
        outline = 'black'

        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=color, outline=outline, width=1)

    def _highlight_last_move(self, row: int, col: int):
        """Highlight the last move."""
        margin = self.cell_size
        x = margin + col * self.cell_size
        y = margin + row * self.cell_size
        r = 5

        color = 'white' if self.game.board[row, col] == 1 else 'black'
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill=color)

    def _on_click(self, event):
        """Handle mouse click on the board."""
        if self.ai_thinking or self.game.game_over:
            return

        if self.game.current_player != self.human_player:
            return

        margin = self.cell_size
        col = round((event.x - margin) / self.cell_size)
        row = round((event.y - margin) / self.cell_size)

        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            action = self.game.coord_to_action(row, col)
            if self.game.is_valid_move(action):
                self._make_move(action)

    def _make_move(self, action: int):
        """Make a move and update the display."""
        self.game.step(action)
        self._draw_board()

        if self.game.game_over:
            self._show_game_over()
        elif self.game.current_player != self.human_player:
            self._ai_move()

    def _ai_move(self):
        """Let the AI make a move."""
        if self.network is None or self.mcts is None:
            # Random move if no model
            valid_moves = self.game.get_valid_moves_list()
            action = np.random.choice(valid_moves)
            self._make_move(action)
            return

        self.ai_thinking = True
        self.status_var.set("AI is thinking...")
        self.root.update()

        def ai_thread():
            action, _ = self.mcts.select_action(
                self.game,
                temperature=0,
                add_noise=False,
            )
            self.root.after(0, lambda: self._ai_move_done(action))

        threading.Thread(target=ai_thread, daemon=True).start()

    def _ai_move_done(self, action: int):
        """Called when AI finishes thinking."""
        self.ai_thinking = False
        self._make_move(action)
        if not self.game.game_over:
            player_str = "Black" if self.human_player == 1 else "White"
            self.status_var.set(f"Your turn ({player_str})")

    def _show_game_over(self):
        """Show game over message."""
        if self.game.winner == 0:
            msg = "It's a draw!"
        elif self.game.winner == self.human_player:
            msg = "You win! 🎉"
        else:
            msg = "AI wins!"

        self.status_var.set(msg)
        messagebox.showinfo("Game Over", msg)

    def _new_game(self):
        """Start a new game."""
        self.game.reset()
        self._draw_board()
        player_str = "Black" if self.human_player == 1 else "White"
        self.status_var.set(f"Your turn ({player_str})")

        # If AI plays first
        if self.human_player != 1:
            self._ai_move()

    def _switch_sides(self):
        """Switch sides and start a new game."""
        self.human_player = -self.human_player
        self._new_game()

    def _load_model_dialog(self):
        """Open file dialog to load a model."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select Model",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")],
            initialdir="checkpoints",
        )
        if path:
            self.load_model(path)
            self.info_label.config(text=f"Model: {path} | Device: {self.device}")
            self._new_game()

    def run(self):
        """Start the GUI main loop."""
        # If AI plays first
        if self.human_player != 1 and self.network is not None:
            self.root.after(100, self._ai_move)
        self.root.mainloop()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Gomoku GUI')
    parser.add_argument('--model', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--board-size', type=int, default=15, help='Board size')
    parser.add_argument('--mcts-sims', type=int, default=800, help='MCTS simulations')
    parser.add_argument('--mcts-batch', type=int, default=32, help='MCTS batch size')
    parser.add_argument('--cell-size', type=int, default=40, help='Cell size in pixels')
    args = parser.parse_args()

    gui = GomokuGUI(
        model_path=args.model,
        board_size=args.board_size,
        cell_size=args.cell_size,
        mcts_sims=args.mcts_sims,
        mcts_batch=args.mcts_batch,
    )
    gui.run()


if __name__ == '__main__':
    main()
