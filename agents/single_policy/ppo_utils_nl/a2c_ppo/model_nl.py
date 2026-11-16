import torch
import torch.nn as nn

import gymnasium as gym

from torch.distributions import Normal, Categorical

from misc.network import layer_init, mlp


class ActorNL(nn.Module):
    def __init__(
            self,
            num_obs: int,
            num_outputs: int,
            net_arch: list[int],
            layernorm: bool
    ):
        """Gaussian actor with a tanh-bounded mean and a learned, state-independent log-std."""
        super().__init__()
        self.net = mlp(
            input_dim=num_obs,
            output_dim=num_outputs,
            net_arch=net_arch,
            activation_fn=nn.Tanh,
            drop_rate=0.0,
            layer_norm=layernorm,
        )
        self.logstd = nn.Parameter(torch.zeros(1, num_outputs))

        # Orthogonal init: gain 1.0 for hidden layers, 0.01 for the output layer
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for layer in linears[:-1]:
            layer_init(layer, method='orthogonal', weight_gain=1.0, bias_const=0.0)
        layer_init(linears[-1], method='orthogonal', weight_gain=0.01, bias_const=0.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the action mean in [-1, 1] and the clamped log-std."""
        action_mean = torch.tanh(self.net(obs))
        action_logstd = torch.clamp(self.logstd, -5, 2).expand_as(action_mean)
        return action_mean, action_logstd


class CriticNL(nn.Module):
    def __init__(
            self,
            num_obs: int,
            num_outputs: int,
            net_arch: list[int],
            layernorm: bool
    ):
        """Vector-valued critic with one output per objective."""
        super().__init__()
        self.net = mlp(
            input_dim=num_obs,
            output_dim=num_outputs,
            net_arch=net_arch,
            activation_fn=nn.Tanh,
            drop_rate=0.0,
            layer_norm=layernorm,
        )

        # Orthogonal init with gain 1.0 for every layer
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                layer_init(layer, method='orthogonal', weight_gain=1.0, bias_const=0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the value estimate for every objective."""
        return self.net(obs)


class MOMLPBaseNL(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            net_arch: list[int],
            layernorm: bool,
            reward_dim: int
    ):
        """Separate Gaussian actor and multi-objective critic for continuous actions."""
        super().__init__()
        self.actor = ActorNL(obs_dim, action_dim, net_arch, layernorm)
        self.critic = CriticNL(obs_dim, reward_dim, net_arch, layernorm)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return critic value, action mean and action log-std."""
        action_mean, action_logstd = self.actor(obs)
        return self.critic(obs), action_mean, action_logstd


class PolicyNL(nn.Module):
    def __init__(
            self,
            obs_shape: tuple[int, ...],
            action_space: gym.spaces.Box,
            net_arch: list[int] = [64, 64],
            layernorm: bool = True,
            reward_dim: int = 1,
    ):
        """Multi-objective actor-critic policy for continuous (Box) action spaces."""
        super().__init__()
        self.base = MOMLPBaseNL(
            obs_dim=obs_shape[0],
            action_dim=action_space.shape[0],
            net_arch=net_arch,
            layernorm=layernorm,
            reward_dim=reward_dim
        )

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or take the mean) action and return value, action and log-probability."""
        value, mu, logstd = self.base(obs)
        dist = Normal(mu, torch.exp(logstd))
        action = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(-1, keepdim=True)
        return value, action, logp

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the critic value for every objective."""
        value, _, _ = self.base(obs)
        return value

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
    torch.Tensor]:
        """Return value, log-probability of the given actions and mean policy entropy."""
        value, mu, logstd = self.base(obs)
        dist = Normal(mu, torch.exp(logstd))
        logp = dist.log_prob(action).sum(-1, keepdim=True)
        entropy = dist.entropy().mean()
        return value, logp, entropy


class ActorDiscrete(nn.Module):
    def __init__(
            self,
            num_obs: int,
            num_outputs: int,
            net_arch: list[int],
            layernorm: bool
    ):
        """Categorical actor that outputs one logit per discrete action."""
        super().__init__()
        self.net = mlp(
            input_dim=num_obs,
            output_dim=num_outputs,
            net_arch=net_arch,
            activation_fn=nn.Tanh,
            drop_rate=0.0,
            layer_norm=layernorm,
        )

        # Orthogonal init: gain 1.0 for hidden layers, 0.01 for the output layer
        linears = [m for m in self.net if isinstance(m, nn.Linear)]
        for layer in linears[:-1]:
            layer_init(layer, method='orthogonal', weight_gain=1.0, bias_const=0.0)
        layer_init(linears[-1], method='orthogonal', weight_gain=0.01, bias_const=0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the action logits of shape [B, A]."""
        return self.net(obs)


class MOMLPBaseNLDiscrete(nn.Module):
    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            net_arch: list[int],
            layernorm: bool,
            reward_dim: int
    ):
        """Separate categorical actor and multi-objective critic for discrete actions."""
        super().__init__()
        self.actor = ActorDiscrete(obs_dim, action_dim, net_arch, layernorm)
        self.critic = CriticNL(obs_dim, reward_dim, net_arch, layernorm)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return critic value and action logits."""
        return self.critic(obs), self.actor(obs)


class PolicyNLDiscrete(nn.Module):
    def __init__(
            self,
            obs_shape: tuple[int, ...],
            action_space: gym.spaces.Discrete,
            net_arch: list[int] = [64, 64],
            layernorm: bool = True,
            reward_dim: int = 1,
    ):
        """Multi-objective actor-critic policy for discrete action spaces."""
        super().__init__()
        self.base = MOMLPBaseNLDiscrete(
            obs_dim=obs_shape[0],
            action_dim=action_space.n,
            net_arch=net_arch,
            layernorm=layernorm,
            reward_dim=reward_dim
        )

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (or take the mode) action and return value, action [N, 1] and log-probability [N, 1]."""
        value, logits = self.base(obs)
        dist = Categorical(logits=logits)
        action = dist.mode if deterministic else dist.sample()
        logp = dist.log_prob(action).unsqueeze(-1)
        return value, action.unsqueeze(-1), logp

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the critic value for every objective."""
        value, _ = self.base(obs)
        return value

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor,
    torch.Tensor]:
        """Return value, log-probability of the given [N, 1] actions and mean policy entropy."""
        value, logits = self.base(obs)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
        entropy = dist.entropy().mean()
        return value, logp, entropy
