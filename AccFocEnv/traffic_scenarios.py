"""This module implements differenct traffic scenarios.

A traffic scenario consists of the trajectory of the target-vehicle.
"""
import numpy as np
import math
from . import vehicle_longitudinal_model
from gym.utils import seeding
import h5py


def my_scenario(*args, **kwargs):
    """Function to select traffic scenario specified in options.

    Args:
        *args: Anything required by traffic scenario.
        **kwargs: Anything required by traffic scenario. Needs to include opts namespace object.

    Returns:
        Traffic scenario object.
    """
    ts = kwargs["opts"].traffic_scenario
    if ts == "SimpleSine":
        return SimpleSine(*args, **kwargs)
    elif ts == "RampsAndHold":
        return RampsAndHold(*args, **kwargs)
    else:
        raise ValueError("Unknown traffic scenario!")


# after creating a traffic scenario object, it has to be seeded and then reset!!
class BaseScenario:
    """Base class to show interface of traffic scenario.
    """
    def __init__(self, opts, dt, ego_car):
        """Initialization

        Args:
            opts: Namespace object with options.
            dt: Discrete step time in seconds.
            ego_car: ego_car object from vehicle_longituidinal_model.py.
        """
        self.ego_car = ego_car  # ego car in follow object control
        self.opts = opts
        self.dt = dt

        # Initial position and velocities.
        self.x_ego_init = 10   # m
        self.v_ego_init = 20   # m/s
        self.x_target_init = 50  # m
        self.v_target_init = 30  # m/s

        # Create target vehicle object.
        self.target_vehicle = vehicle_longitudinal_model.SimpleVehicle(
            dt=dt,
            T=opts.traffic_scenario_vehicle_time_constant,
            v_init=self.v_target_init,
            x_init=self.x_target_init,
            allow_v_negative=False
        )

    def seed(self, seed=None):
        """Seeds the traffic scenario.

        Args:
            seed: Seed to use.
        """
        self.rnd, seed = seeding.np_random(seed)

    def step(self, t):
        """Calculate next values for target vehicle.

        Args:
            t: Current time.

        Returns:
            Tuple consisting of (acceleration, velocity, position, done signal) of target vehicle.
        """
        # return a, v, x, done

        raise NotImplementedError

    def reset(self):
        """Reset initial positions and velocities for lead and ego car.
        """
        raise NotImplementedError


class RampsAndHold(BaseScenario):
    """Traffic scenario for target vehicle including acceleration and deceleration and
    constant velocity driving. The parameters of acceleration and deceleration are 
    defined by a ramp, which is randomly rescaled.
    """
    def __init__(self, opts, dt, ego_car):
        """Initialization

        Args:
            opts: Namespace object with options.
            dt: Discrete step time in seconds.
            ego_car: ego_car object from vehicle_longituidinal_model.py.
        """
        super().__init__(opts, dt, ego_car)

        self.chunk_time = 10
        self.gradient_limit = vehicle_longitudinal_model.GradientLimiter(dt=dt, gradient_limit=4)

        # Parameters of ramp. The ramp consists of 36 segments with different acceleration values.
        # Each segment is chunk_time seconds long.
        self.seg_end = [3, 5, 8, 11, 13, 16, 17, 19, 24, 25, 26, 28, 30, 36]
        self.seg_v_init = [9, 19.4, 19.4, 42, 42, 9, 9, 19.4, 20, 30, 30, 19.4, 50, 50]
        self.seg_a_rel = [3.46666667, 0.0, 7.53333333, 0.0, -16.5, 0.0, 10.4, 0.3, 2.0, 0.0, -10.6, 15.3, 0.0, -6.83333333]
        self.num_chunks = 36

    def step(self, t):
        self.t += self.dt
        if self.t >= self.chunk_time * self.num_chunks:
            self.t = 0
            if self.rnd.random_sample() < self.opts.RampsAndHold_random[3]:
                self.randomize_period()

        # Random Cut-In & Cut-Out of target-vehicle.
        v_ego = self.ego_car.get_v()
        v_tar = self.target_vehicle.get_v()
        if self.rnd.uniform(0, 1) < self.opts.cut_in_cut_out_random[0] and v_ego > self.opts.stop_n_go_velocity:
            new_Hw = self.rnd.uniform(self.opts.cut_in_cut_out_random[1], self.opts.cut_in_cut_out_random[2])
            old_Hw = (self.target_vehicle.get_s() - self.ego_car.get_s()) / max(0.01, v_ego)

            if v_ego - v_tar > 5:
                new_Hw = max(old_Hw, new_Hw)
                v_tar = max(self.target_vehicle.get_v(), self.target_vehicle.get_v() + self.rnd.uniform(-self.opts.cut_in_cut_out_random[3] * 0.75, self.opts.cut_in_cut_out_random[3]))
            elif v_ego - v_tar > 2.5:
                new_Hw = max(old_Hw * 0.75, new_Hw)
                v_tar = max(self.target_vehicle.get_v() - 2.5, self.target_vehicle.get_v() + self.rnd.uniform(-self.opts.cut_in_cut_out_random[3] * 0.75, self.opts.cut_in_cut_out_random[3]))
            else:
                v_tar = max(0, self.target_vehicle.get_v() + self.rnd.uniform(-self.opts.cut_in_cut_out_random[3] * 0.75, self.opts.cut_in_cut_out_random[3]))
            v_tar = max(v_tar, self.opts.stop_n_go_velocity)
            self.target_vehicle.set_v(v_tar)
            dist = new_Hw * v_ego
            self.target_vehicle.set_s(self.ego_car.get_s() + dist)

        # Calculate current segment.
        seg = math.floor(self.t / self.chunk_time)
        seg += 1
        pos = -1
        for i in range(len(self.seg_end)):
            pos = i
            if seg <= self.seg_end[i]:
                break

        a = self.seg_a_rel[pos] / self.chunk_time * self.a_factor

        a, v, s = self.target_vehicle.step(self.gradient_limit(a))

        return a, v, s, False

    def reset(self):
        """Reset initial positions and velocities for lead and ego car.
        """
        self.randomize_period()

        self.a_factor = self.rnd.uniform(0.5, 1)

        # Start at beginning of this segment
        start_seg = self.rnd.randint(0, len(self.seg_end))

        # Offset that is added to velocity profile
        self.v_offset = self.rnd.uniform(-8, 10)

        v = max(0, self.seg_v_init[start_seg] + self.v_offset)

        # Ideal initial conditions: v_ego==v_lead and ideal time gap
        if self.opts.traffic_scenario_start_ideal:
            self.ego_car.set_v_init(v)
            self.ego_car.set_x_init(0)

            self.target_vehicle.set_v_init(v)
            self.target_vehicle.set_x_init(v * self.opts.desired_headway + self.opts.stop_n_go_distance)
        else:
            v_ego_init = max(1, v + self.rnd.uniform(-20, 6))
            self.ego_car.set_v_init(v_ego_init)
            self.ego_car.set_x_init(0)

            self.target_vehicle.set_v_init(v)
            self.target_vehicle.set_x_init(v_ego_init * self.rnd.uniform(1, 10) + 3)
        self.ego_car.reset()
        self.target_vehicle.reset()

        self.last_seg = start_seg - 1
        self.t = self.seg_end[start_seg] * self.chunk_time

    def randomize_period(self):
        """Randomizes the period of the traffic scenario according to opts.
        """
        if self.opts.RampsAndHold_random[0]:
            self.chunk_time = self.rnd.uniform(self.opts.RampsAndHold_random[1], self.opts.RampsAndHold_random[2])
