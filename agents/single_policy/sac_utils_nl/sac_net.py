import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym

from torch.distributions import Categorical

from misc.network import mlp, layer_init


class Actor(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 action_space: gym.spaces.Box,
                 net_arch: list[int] = [256, 256]):
        """Tanh-squashed Gaussian actor rescaled to the bounds of the Box action space."""
        super().__init__()
        self.latent_pi = mlp(obs_dim, -1, net_arch)
        self.mean = nn.Linear(net_arch[-1], action_dim)
        self.log_std_linear = nn.Linear(net_arch[-1], action_dim)
        self.register_buffer('action_scale',
                             torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer('action_bias',
                             torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))
        self.apply(layer_init)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the Gaussian mean and the log-std, squashed into [-5, 2]."""
        h = self.latent_pi(obs)
        log_std = -5 + 3.5 * (torch.tanh(self.log_std_linear(h)) + 1)
        return self.mean(h), log_std

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the deterministic action: the squashed and rescaled mean."""
        mean, _ = self.forward(obs)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a reparameterized action, its log-probability (with tanh/scale correction) and the mean action."""
        mean, log_std = self.forward(obs)
        normal = torch.distributions.Normal(mean, log_std.exp())
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        return action, log_prob.sum(dim=1), torch.tanh(mean) * self.action_scale + self.action_bias


class SoftQNetwork(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 output_dim: int,
                 net_arch: list[int] = [256, 256]):
        """Q-network for continuous actions with one output per objective."""
        super().__init__()
        self.net = mlp(obs_dim + action_dim, output_dim, net_arch, activation_fn=nn.ReLU)
        self.apply(layer_init)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return the Q-values of the observation-action pairs."""
        return self.net(torch.cat((obs, action), dim=1))


class ActorDiscrete(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, net_arch: list[int] = [256, 256]):
        """Categorical actor with one logit per discrete action."""
        super().__init__()
        self.latent_pi = mlp(obs_dim, -1, net_arch)
        self.logits = nn.Linear(net_arch[-1], action_dim)
        self.apply(layer_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the action logits of shape [B, A]."""
        return self.logits(self.latent_pi(obs))

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a sampled action, its log-probability and the probabilities of all actions."""
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        action_log_prob = F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(1)).squeeze(1)
        return action, action_log_prob, dist.probs


class SoftQNetworkDiscrete(nn.Module):
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 output_dim: int,
                 net_arch: list[int] = [256, 256]):
        """Q-network for discrete actions with one output per action and objective."""
        super().__init__()
        self.action_dim = action_dim
        self.output_dim = output_dim
        self.net = mlp(obs_dim, action_dim * output_dim, net_arch, activation_fn=nn.ReLU)
        self.apply(layer_init)

    def forward_all(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the Q-values of all actions with shape [B, A, output_dim]."""
        return self.net(obs).view(-1, self.action_dim, self.output_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return the Q-values of the given action indices."""
        action_idx = action.long().view(-1, 1, 1).expand(-1, 1, self.output_dim)
        return self.forward_all(obs).gather(1, action_idx).squeeze(1)