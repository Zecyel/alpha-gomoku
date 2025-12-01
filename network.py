"""
Neural Network Architecture for AlphaGo Zero style Gomoku.
ResNet with separate policy and value heads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class PolicyHead(nn.Module):
    """Policy head outputs move probabilities."""

    def __init__(self, channels: int, board_size: int):
        super().__init__()
        self.board_size = board_size
        self.conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out  # Raw logits, apply softmax during inference


class ValueHead(nn.Module):
    """Value head outputs position evaluation."""

    def __init__(self, channels: int, board_size: int, hidden_size: int = 256):
        super().__init__()
        self.board_size = board_size
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(board_size * board_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn(self.conv(x)))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.tanh(self.fc2(out))
        return out


class AlphaZeroNetwork(nn.Module):
    """
    AlphaGo Zero style neural network.

    Architecture:
    - Input convolution
    - Stack of residual blocks
    - Policy head (move probabilities)
    - Value head (position evaluation)
    """

    def __init__(
        self,
        board_size: int = 15,
        input_channels: int = 4,
        num_channels: int = 128,
        num_res_blocks: int = 10,
    ):
        super().__init__()
        self.board_size = board_size
        self.input_channels = input_channels
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks

        # Input convolution
        self.input_conv = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(num_channels)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # Policy and value heads
        self.policy_head = PolicyHead(num_channels, board_size)
        self.value_head = ValueHead(num_channels, board_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_channels, board_size, board_size)

        Returns:
            policy_logits: Shape (batch_size, board_size * board_size)
            value: Shape (batch_size, 1)
        """
        # Input convolution
        out = F.relu(self.input_bn(self.input_conv(x)))

        # Residual blocks
        for block in self.res_blocks:
            out = block(out)

        # Heads
        policy_logits = self.policy_head(out)
        value = self.value_head(out)

        return policy_logits, value

    def predict(self, state: torch.Tensor, valid_moves: torch.Tensor = None) -> Tuple[torch.Tensor, float]:
        """
        Get policy probabilities and value for a single state.

        Args:
            state: Input tensor of shape (input_channels, board_size, board_size)
            valid_moves: Optional mask of valid moves

        Returns:
            policy: Probability distribution over moves
            value: Position evaluation
        """
        self.eval()
        with torch.inference_mode():
            if state.dim() == 3:
                state = state.unsqueeze(0)

            policy_logits, value = self.forward(state)

            # Mask invalid moves
            if valid_moves is not None:
                if valid_moves.dim() == 1:
                    valid_moves = valid_moves.unsqueeze(0)
                # Set invalid moves to very negative value
                policy_logits = policy_logits.masked_fill(valid_moves == 0, -1e9)

            policy = F.softmax(policy_logits, dim=1)

        return policy.squeeze(0), value.squeeze().item()


class AlphaZeroLoss(nn.Module):
    """
    Combined loss function for AlphaGo Zero.
    Loss = MSE(value, target_value) + CrossEntropy(policy, target_policy)
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        policy_logits: torch.Tensor,
        value: torch.Tensor,
        target_policy: torch.Tensor,
        target_value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the combined loss.

        Args:
            policy_logits: Predicted policy logits
            value: Predicted value
            target_policy: Target policy (MCTS visit counts)
            target_value: Target value (game outcome)

        Returns:
            total_loss, policy_loss, value_loss
        """
        # Value loss: MSE
        value_loss = F.mse_loss(value.squeeze(), target_value)

        # Policy loss: Cross entropy with soft targets
        # Using log_softmax for numerical stability
        log_policy = F.log_softmax(policy_logits, dim=1)
        policy_loss = -torch.mean(torch.sum(target_policy * log_policy, dim=1))

        total_loss = value_loss + policy_loss

        return total_loss, policy_loss, value_loss


if __name__ == '__main__':
    # Test the network
    board_size = 15
    batch_size = 8

    model = AlphaZeroNetwork(board_size=board_size)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    x = torch.randn(batch_size, 4, board_size, board_size)
    policy_logits, value = model(x)
    print(f"Policy logits shape: {policy_logits.shape}")
    print(f"Value shape: {value.shape}")

    # Test predict
    single_state = torch.randn(4, board_size, board_size)
    valid_moves = torch.ones(board_size * board_size)
    valid_moves[0] = 0  # Mark first move as invalid
    policy, val = model.predict(single_state, valid_moves)
    print(f"Policy shape: {policy.shape}, sum: {policy.sum().item():.4f}")
    print(f"Value: {val:.4f}")

    # Test loss
    criterion = AlphaZeroLoss()
    target_policy = F.softmax(torch.randn(batch_size, board_size * board_size), dim=1)
    target_value = torch.randn(batch_size)
    total_loss, p_loss, v_loss = criterion(policy_logits, value, target_policy, target_value)
    print(f"Total loss: {total_loss.item():.4f}, Policy: {p_loss.item():.4f}, Value: {v_loss.item():.4f}")
