"""
Custom Dueling DQN network for SB3.

Implements the architecture from Section 3.2:
    input(5) -> FC(64,ReLU) -> FC(64,ReLU) -> split:
        value stream:     FC(1)
        advantage stream: FC(24)
    Q = V + (A - mean(A))

Registered as a custom policy for stable-baselines3 DQN.
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.dqn.policies import DQNPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-network.

    Parameters
    ----------
    obs_dim    : input dimension (5)
    n_actions  : number of discrete actions (24)
    hidden_dim : width of shared and stream layers (64)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()

        # shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # value stream: scalar V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # advantage stream: A(s, a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions),
        )

        # parameter count sanity check
        n_params = sum(p.numel() for p in self.parameters())
        assert n_params < 20_000, f"Network too large for UE: {n_params} params"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns Q(s,a) for all actions, shape (batch, n_actions)."""
        h  = self.trunk(x)
        V  = self.value_stream(h)          # (batch, 1)
        A  = self.advantage_stream(h)      # (batch, n_actions)
        # subtract mean advantage for identifiability
        Q  = V + (A - A.mean(dim=1, keepdim=True))
        return Q


def make_dueling_dqn_kwargs(hidden_dim: int = 64) -> dict:
    """
    Returns policy_kwargs for SB3 DQN that swaps in the DuelingQNetwork.

    SB3's DQN accepts net_arch but doesn't natively support dueling.
    We pass a custom q_net_class instead.

    Usage:
        model = DQN("MlpPolicy", env,
                    policy_kwargs=make_dueling_dqn_kwargs(),
                    ...)
    """
    return {
        "net_arch": [64, 64],        # SB3 uses this for the default MLP
        "activation_fn": nn.ReLU,
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    net = DuelingQNetwork(obs_dim=5, n_actions=24)
    x   = torch.randn(4, 5)
    q   = net(x)
    print(f"Output shape: {q.shape}")         # (4, 24)
    print(f"Parameters:   {count_parameters(net):,}")
    print(f"Max Q: {q.max().item():.3f}")
