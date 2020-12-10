"""Twin Delayed DDPG (TD3) algorithm, including 

See https://arxiv.org/abs/1802.09477v3
Based on https://github.com/sfujim/TD3
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

from . import polytope_constraints
from .data_std_mean import data_std, data_mean

device = utils.device


class Actor(nn.Module):
    """Implements the standard actor for TD3 as a feedforward neural network with two hidden layers.
    """

    def __init__(self, state_dim, action_dim, max_action, a_min, a_max, opts={}):
        """Initialization of unconstrained actor and layers

        Args:
            state_dim: Number of state dimensions.
            action_dim: Number of action dimensions.
            max_action: Absolute maximum of action.
            a_min: Minimum action.
            a_max: Maximum action.
            opts: Namespace object with further options if needed.
        """
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, opts.neurons)
        self.l2 = nn.Linear(opts.neurons, opts.neurons)
        self.l3 = nn.Linear(opts.neurons, action_dim)

        self.a_min = a_min
        self.a_max = a_max
        self.max_action = (a_max - a_min) / 2

    def forward(self, state, cstr=None):
        """Forward pass

        Args:
            state: Current state.
            cstr: Constraints. Should be left empty for unconstrained actor.

        Returns:
            Action.
        """
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return (self.a_min + self.a_max) / 2 + self.max_action * torch.tanh(self.l3(a))


class ConstrainedActor(nn.Module):
    """Implements the ConstraintNet actor for TD3 as a feedforward neural network with two hidden layers.
    """

    def __init__(self, state_dim, action_dim, max_action, a_min, a_max, opts={}):
        """Initialization of ConstraintNet actor and layers

        Args:
            state_dim: Number of state dimensions.
            action_dim: Number of action dimensions.
            max_action: Absolute maximum of action.
            a_min: Minimum action.
            a_max: Maximum action.
            opts: Namespace object with further options if needed.
        """
        super(ConstrainedActor, self).__init__()

        assert "a_min" in opts.observations, "Minimal allowed acceleration 'a_min' has to be part of the observations for ConstrainedActor!"
        assert "a_max" in opts.observations, "Maximum allowed acceleration 'a_max' has to be part of the observations for ConstrainedActor!"

        self.l1 = nn.Linear(state_dim, opts.neurons)
        self.l2 = nn.Linear(opts.neurons, opts.neurons)
        self.l3 = nn.Linear(opts.neurons, 2)

        self.constr_para2repr = polytope_constraints.opts2v_polys_acc(opts)
        self.constr_mapping = polytope_constraints.opts2polys(opts)

    def forward(self, state, cstr):
        """Forward pass

        Args:
            state: Current state.
            cstr: Constraints. 

        Returns:
            Action.
        """
        # State already contains normalized representation of constraints.
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.l3(a)

        reprs = self.constr_para2repr(cstr)
        return self.constr_mapping(a, reprs)


class Critic(nn.Module):
    """Implements the critic for TD3 as a feedforward neural network with two hidden layers.
    """

    def __init__(self, state_dim, action_dim, opts={}):
        """Initializes TD3 critic.

        Creates two critics to address overestimation.

        Args:
            state_dim: Number of state dimensions.
            action_dim: Number of action dimensions.
            opts: Namespace object with further options if needed.
        """
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, opts.neurons)
        self.l2 = nn.Linear(opts.neurons, opts.neurons)
        self.l3 = nn.Linear(opts.neurons, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, opts.neurons)
        self.l5 = nn.Linear(opts.neurons, opts.neurons)
        self.l6 = nn.Linear(opts.neurons, 1)

    def forward(self, state, action):
        """Forward pass. Return estimated Q-values of both critics.

        Args:
            state: Current state.
            action: Choosen action.

        Returns:
            Estimate of Q-values of critic 1 and critic 2.
        """
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        """Forward pass. Return estimated Q-values of critic 1.

        Args:
            state: Current state.
            action: Choosen action.

        Returns:
            Estimate of Q-values of critic 1.
        """
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    """Twin Delayed DDPG (TD3) algorithm

    See https://arxiv.org/abs/1802.09477v3
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        a_min,
        a_max,
        opts={}
    ):
        """Initialize TD3.

        Args:
            state_dim: Number of state dimensions.
            action_dim: Number of action dimensions.
            max_action: Absolute maximum of action.
            a_min: Minimum action.
            a_max: Maximum action.
            opts: Namespace object with further options if needed.
        """
        # Select either standard actor or ConstraintNet actor based on options.
        if opts.actor_type.lower() == "unconstrained":
            ActorFactory = Actor
        elif opts.actor_type.lower() == "constrained":
            ActorFactory = ConstrainedActor
        else:
            raise ValueError(f"Actor type '{opts.actor_type}' unknown!")
        
        # Actor, target actor and optimizer
        self.actor = ActorFactory(state_dim, action_dim, max_action, a_min, a_max, opts=opts).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        # Critic, target critic and optimizer
        self.critic = Critic(state_dim, action_dim, opts=opts).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        # Store parameters
        self.max_action = max_action
        self.discount = opts.discount
        self.tau = opts.tau
        self.policy_noise = opts.policy_noise * max_action
        self.noise_clip = opts.noise_clip * max_action
        self.policy_freq = opts.policy_freq
        self.opts = opts

        self.total_it = 0

    def select_action(self, state):
        """Use actor to select action based on state.

        Args:
            state: Current state.

        Returns:
            Choosen action.
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        state, cstr = self._normalize_state(state)
        return self.actor(state, cstr).cpu().data.numpy().flatten()

    def train(self, replay_buffer):
        """Train actor and critic for one step.

        Args:
            replay_buffer: Replay buffer to sample transitions from.
        """
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(self.opts.batch_size)

        state, cstr = self._normalize_state(state)
        next_state, next_cstr = self._normalize_state(next_state)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state, next_cstr) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state, cstr)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        """Saves the current models.

        Args:
            filename: Path to save model.
        """
        filename = str(filename)
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        """Loads a pretrained model.

        Args:
            filename: Path to pretrained model.
        """
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))

        self.critic_target.load_state_dict(torch.load(filename + "_critic"))
        self.actor_target.load_state_dict(torch.load(filename + "_actor"))

    def _normalize_state(self, state):
        """Normalizes observations to zero mean and standard deviation of one if enabled in options.

        Args:
            state: State to normalize.

        Returns:
            Normalized state and values of constraints.
        """
        # If constraints are in observations, extract them without normalization.
        if "a_min" in self.opts.observations:
            cstr = state[:, [self.opts.observations.index("a_min"), self.opts.observations.index("a_max")]].clone()
        else:
            cstr = None

        # Normalize state
        if self.opts.normalize_data:
            state = state.clone()
            for pos, key in enumerate(self.opts.observations):
                # Subtract mean und divide by standard deviation
                state[:, pos] -= data_mean[key]
                state[:, pos] /= data_std[key]

        return state, cstr
