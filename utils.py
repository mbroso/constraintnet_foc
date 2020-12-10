"""Utils. Containing several helper function and classes.
"""
import numpy as np
import torch
import signal
import matplotlib
import matplotlib.pyplot as plt
import gym

from AccFocEnv.plotter import Plotter

# Allows to change training device.
device = torch.device("cpu")


def load_env(opts, dont_render=False):
    """Load and initialize environment.

    Args:
        dont_render: Disable rendering.

    Returns:
        Tuple containing:
            env: Environment
            state_dim: Number of state dimensions
            action_dim: Number of action dimensions
            max_action: Absolute maximum value of action
            plotter: Plotter object if plotting is enabled
    """
    if opts.render and not dont_render:
        plotter = Plotter(opts)

        env = gym.make(opts.env, opts=opts, plotter=plotter)
    else:
        plotter = None

        env = gym.make(opts.env, opts=opts)

    # Set seeds
    env.seed(opts.seed)
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(max(abs(opts.vehicle_a_min), abs(opts.vehicle_a_max)))

    return env, state_dim, action_dim, max_action, plotter


class ReplayBuffer(object):
    """Replay buffer storing transitions of the environment for experience replay
    """

    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        """Initialize object

        Args:
            state_dim: Number of observation dimensions.
            action_dim: Number of action dimensions.
            max_size: Maximum size of replay buffer.
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        """Adds a transition to the buffer.

        Args:
            state: Current state.
            action: Choosen action.
            next_state: Next state.
            reward: Observed reward.
            done: Done flag, indicates terminal action.
        """
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """Samples a mini batch of the replay buffer

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple containing tensors of state, action, next_state, reward and not_done.
        """
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device)
        )


class GracefulKiller:
    """Allows graceful exit of running training
    """

    def __init__(self):
        """Catch SIGINT and SIGTERM signal

        The kill_now variable can be read to check if the user pressed Ctrl-C.
        """
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True
