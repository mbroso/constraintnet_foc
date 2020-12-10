"""This module implements a simulated follow object control environment following OpenAI Gym interface.
"""

import math
import numpy as np
import gym
from gym import error, spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from pathlib import Path

from . import traffic_scenarios
from . import vehicle_longitudinal_model
from . import reward_functions
from . import acceleration_constraints


class AccFocEnv(gym.Env):
    """Custom environment for follow object control that follows OpenAI gym interface"""
    metadata = {'render.modes': ['episode']}

    def __init__(self, opts, plotter=None):
        """Initialize environment

        Args:
            opts: Namespace object with options.
            plotter: Plotter object to enable plotting of each episode.

        """
        self.opts = opts

        # Environment parameters
        self.dt = opts.env_dt
        self.phys_dt = opts.sim_dt

        # store plotter for plotting of episodes
        self.plotter = plotter

        # Configure timing
        self.pyhs_steps_subsample = round(self.dt / self.phys_dt)
        assert self.dt >= self.phys_dt and self.pyhs_steps_subsample == self.dt / self.phys_dt, \
            "AccFocEnv: Intervals for train and pyhsics simulation don't match! env_dt has to be a multiple of sim_dt"
        self._max_episode_steps = round(opts.env_stop_time / self.dt)

        # Define action space. Define observation space.
        self.action_space = spaces.Box(low=opts.vehicle_a_min, high=opts.vehicle_a_max, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(self.opts.observations),), dtype=np.float32)

        # Create ego car object. Initial position and velocity will be set by traffic scenario.
        self.ego_car = vehicle_longitudinal_model.my_vehicle_model(
            opts=opts, dt=self.phys_dt
        )

        # Environment including lead car from choosen traffic scenario.
        self.environment = traffic_scenarios.my_scenario(opts=opts, dt=self.phys_dt, ego_car=self.ego_car)

        # Reward function specified by options.
        self.reward_function = reward_functions.my_reward_function(opts=opts)

        # Load specified costraints.
        self.constraints = acceleration_constraints.AccelerationConstraints(self.opts)

    def seed(self, seed=None):
        """Seeds the whole environment.

        Args:
            seed: Random seed.

        Returns:
            Random seed.
        """
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.environment.seed(seed)

        return [seed]

    def reset(self):
        """Resets environment, traffic scenario and variables

        Returns:
            Initial state.
        """
        # Reset internal values and environment
        self.environment.reset()
        # self.ego_car.reset()  => Resetting ego_car is handled by traffic_scenario
        self.t = 0.0
        self.steps = 0
        self.last_a_dem = 0
        self.last_a_ego = 0
        self.last_Hw = -1
        self.last_a_min = -0.1
        self.last_a_max = 0.1

        # Create buffer if it doesn't exist yet. In subsequent resets do nothing, values in buffer will be overwritten.
        if not hasattr(self, "data_store"):
            self.data_store = {}

        return self.step([0])[0]  # Return only state

    def step(self, action):
        """Simulates one environment step

        Args:
            action: List of chosen action.

        Returns:
            OpenAI Gym compatible return: Dict containing (observations, reward, done, debug_infos)
        """

        # Get desired acceleration and check system boundaries.
        a_dem = action[0]
        assert self.opts.vehicle_a_min <= a_dem <= self.opts.vehicle_a_max, f"Action {a_dem} m/s² not part of action space!"
        # Clip a_dem according to constraints when specified in opts
        if self.opts.clip_a_dem == True:
            a_dem = np.clip(a_dem, self.last_a_min, self.last_a_max)

        # Simulate next timesteps of environment and ego_car.
        for i in range(self.pyhs_steps_subsample):
            a_tar, v_tar, x_tar, scenario_done = self.environment.step(self.t + self.phys_dt * i)
            a_ego, v_ego, x_ego = self.ego_car.step(a_dem)

        # Calucate correction velocity to increase distance in Stop&Go scenario.
        v_correction = 0
        if v_ego < self.opts.stop_n_go_velocity:
            v_correction = self.opts.stop_n_go_distance / self.opts.desired_headway * (self.opts.stop_n_go_velocity - v_ego) / self.opts.stop_n_go_velocity

        # Calulate and clip headway and its derivation.
        Hw = (x_tar - x_ego) / max(0.001, v_ego + v_correction)
        dHw = (Hw - self.last_Hw) / self.dt
        if self.last_Hw == -1:
            dHw = 0  # Prevent inital value from being to big
        self.last_Hw = Hw
        Hw = max(0, min(10.01, Hw))
        dHw = max(-0.75, min(0.75, dHw))

        # Calculate safe distance. Increase distance for Stop&Go scenario.
        safe_distance = self.opts.desired_headway * abs(v_ego)
        if v_ego < self.opts.stop_n_go_velocity:
            safe_distance += self.opts.stop_n_go_distance * (1 - max(0, v_ego) / self.opts.stop_n_go_velocity)

        # All variables in this dict can be used as observation, in the reward function or can be plotted.
        state = {
            # Time and raw commanded acceleration by agent.
            't': self.t,
            'a_dem': a_dem,
            # Ego vehicle.
            'a_ego': a_ego,
            'v_ego': v_ego,
            'x_ego': x_ego,
            'j_ego': (a_ego - self.last_a_ego) / self.dt,
            # Target vehicle.
            'a_tar': a_tar,
            'v_tar': v_tar,
            'x_tar': x_tar,
            # Relative values.
            'a_rel': a_tar - a_ego,
            'v_rel': v_tar - v_ego,
            'x_rel': x_tar - x_ego,
            # Control setpoints.
            'd_safe': safe_distance,
            'd_err': safe_distance - (x_tar - x_ego),
            'Hw': Hw,
            'dHw': dHw,
            'v_err': v_tar - v_ego,
            # misc
            'last_a_dem': self.last_a_dem,
            'last_a_ego': self.last_a_ego,
        }

        # Calculation upper and lower constraint for acceleration and add to state.
        state["a_min"], state["a_max"] = self.constraints.calculate(state)

        # end episode of ego car crashed in the lead car or car goes backwards fast
        # done signal
        #   done = 0: not done, episode can continue
        #   done = 1: done, because simulated time ended
        #   done = 2: done, because agent ended in terminal step (e.g. crash)
        done = 1 if scenario_done or (self.steps >= self._max_episode_steps - 1) else 0
        done = 2 if (x_tar - x_ego) < -50 or v_ego < -5 else done
        state["done"] = done

        # Calculate reward and add to state.
        reward = self.reward_function(state, self.opts)
        state["reward"] = reward

        # Store state values in buffer for later plotting.
        if self.steps < self._max_episode_steps:
            # Store all state variables in data_store.
            for k, v in state.items():
                if k not in self.data_store:
                    self.data_store[k] = np.zeros(self._max_episode_steps)
                self.data_store[k][self.steps] = v

        # Add choosen action to previous timestep in state dict.
        if self.steps >= 1:
            self.data_store["a_dem"][self.steps - 1] = a_dem

        # Extract observations from state dict.
        obs = [state[key] for key in self.opts.observations]

        # Increment counter and time. Store last values.
        self.steps += 1
        self.t += self.dt
        self.last_a_dem = a_dem
        self.last_a_ego = a_ego
        self.last_a_min = state["a_min"]
        self.last_a_max = state["a_max"]

        # OpenAI Gym compatible return: (observations, reward, done, debug_infos)
        return np.array(obs, dtype=np.float32), reward, done, {}

    def render(self, mode='human', close=False):
        """Live rendering not supported. See render_episode()"""
        pass

    def render_episode(self, prefix=""):
        """Render a complete episode at its end using the plotter in a seperate thread.
        """
        if self.plotter is None:
            return

        self.plotter.plot([self.data_store, self.steps, prefix])

    def calc_metrics(self):
        """Calculate metrics at the end of an episode.

        Returns:
            Dict with keys:
                safety: Metric for safety. Higher values are better. A value of 0 indicates a crash.
                discomfort: Metric measuring discomfort. Lower values are better.
                tracking_error: Metric measuring tracking error. Lower values are better.
        """
        safety = min(1, np.min(self.data_store["Hw"][0:self.steps]) / self.opts.desired_headway)

        discomfort = np.mean(self.data_store["a_ego"][0:self.steps]**2) + 0.5 * np.mean(self.data_store["j_ego"][0:self.steps]**2)

        tracking_error = np.mean((self.data_store["Hw"][0:self.steps] - self.opts.desired_headway)**2)
        tracking_error = min(9, tracking_error)

        return {"safety": safety, "discomfort": discomfort, "tracking_error": tracking_error}
