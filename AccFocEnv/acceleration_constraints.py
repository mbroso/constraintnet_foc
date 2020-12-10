"""This module implements calculation of acceleration constraints.
"""

import math
import numpy as np


class AccelerationConstraints:
    """Calculates lower and upper constraint for commanded acceleration by RL-FOC.

    Different types of constraints can be specified by parameters.
    """

    def __init__(self, opts):
        """Initializes constraints by supplied options.

        Args:
            opts: Namespace object with options.
        """
        self.opts = opts
        self.dt = self.opts.env_dt

        self.constraint_lower = opts.constraints_lower.replace(" ", "").split("+")
        self.constraint_upper = opts.constraints_upper.replace(" ", "").split("+")

        assert opts.cstr_no_backwards_jerk > 0, "cstr_no_backwards_jerk has to be a positive number!"
        assert opts.cstr_no_crash_tg_min > 0, "cstr_no_crash_tg_min has to be a positve nubmer!"
        assert opts.cstr_no_crash_a_min < 0, "cstr_no_crash_a_min has to be a negative number!"
        assert opts.cstr_no_crash_jerk > 0, "cstr_no_crash_jerk has to be a positive number!"

    def calculate(self, state):
        """Calcuate constraints for current state.

        Args:
            state: State dict containing all signals.

        Returns:
            Tuple containing minimal and maximum acceleration (a_min, a_max).
        """
        a_min = self.lower(state)
        a_max = self.upper(state)

        # a_max may not be lower than a_min
        if a_max < a_min:
            a_max = a_min

        return (a_min, a_max)

    def lower(self, state):
        """Calculates lower constraint.

        Args:
            state: state of AccFocEnv containing all relevant measurements.

        Returns:
            lower bound for acceleration in m/s^2
        """
        a = self.opts.vehicle_a_min
        if "iso" in self.constraint_lower:
            a = max(a, self._iso_15622_acceleration_lower_limit(state["v_ego"]))
        if "no_backwards" in self.constraint_lower:
            # Approximate lagging vehicle behaviour with a dead time and assume a_ego to be constant during this time.
            T_deadtime = 0.5
            v = max(0, state["v_ego"] + min(0, state["a_ego"]) * T_deadtime)
            a = max(a, -math.sqrt(2 * v * self.opts.cstr_no_backwards_jerk))

        # minimal acceleration of car is -4.5 m/s^2
        a = min(a, 0)
        a = max(a, self.opts.vehicle_a_min)

        return a

    def upper(self, state):
        """Calculates upper constraint.

        Args:
            state: state of AccFocEnv containing all relevant measurements.

        Returns:
            upper bound for acceleration in m/s^2

        """
        a = self.opts.vehicle_a_max
        if "no_crash" in self.constraint_upper:
            a = self._no_crash_acceleration_upper_limit(state, compensate_lag=True, stop_n_go=True)
            a = max(self.opts.cstr_no_crash_a_min, a)

        # limit maximal acceleration to system limit of 3 m/s^2
        a = max(a, self.opts.vehicle_a_min)
        a = min(a, self.opts.vehicle_a_max)

        return a

    def _iso_15622_acceleration_lower_limit(self, v):
        """Lowest allowd acceleration (highes deceleration) according ISO 15622

        Args:
            v: Velocity of vehicle in m/s.

        Returns:
            Minimum allowd acceleration in m/s^2

        """
        if v < 5:
            a_iso = -5.0
        elif 5 <= v <= 20:
            a_iso = -5.0 + (v - 5.0) / 10.0
        else:
            a_iso = -3.5

        return a_iso

    def _no_crash_acceleration_upper_limit(self, state, compensate_lag=True, stop_n_go=True):
        """Calculate upper bound to satisfy safety criteria.

        Args:
            state: Current state.
            compensate_lag: Compensate lagging behaviour of ego vehicle.
            stop_n_go: Consider Stop&Go scenarios and increase required distance by stop&go distance.

        Returns:
            Maximum allowed acceleration to prevent undershooting of minimal time gap.
        """
        # Extract needed values from state.
        v_ego = state["v_ego"]
        a_ego = state["a_ego"]
        v_tar = state["v_tar"]
        a_tar = state["a_tar"]
        x_rel = state["x_rel"]
        v_rel = state["v_rel"]

        # Reduce relative distance by desired standstill distance.
        if stop_n_go:
            x_rel -= self.opts.stop_n_go_distance

        # If the time gap has fallen below the minimum value (e.g. due to a cut-in vehicle) apply maximum braking.
        if x_rel < self.opts.cstr_no_crash_tg_min * v_ego:
            return self.opts.vehicle_a_min

        # Compensate lagging dynamics of ego vehicle.
        if compensate_lag:
            T_deadtime = 0.5
            v_ego += a_ego * T_deadtime
            v_tar = v_tar + a_tar * T_deadtime
            v_rel = v_tar - v_ego
            x_rel = x_rel + (state["v_tar"] - state["v_ego"]) * T_deadtime + 0.5 * (a_tar - a_ego) * T_deadtime**2

        # Extract constraint parameter
        tg_min = self.opts.cstr_no_crash_tg_min
        a_min = self.opts.cstr_no_crash_a_min
        j_ego = -self.opts.cstr_no_crash_jerk

        # calculate relative acceleration at the critical point
        a_rel_k = a_tar - a_min
        if a_rel_k == 0.0:
            a_rel_k = 0.00001

        # define coefficents of the control equation
        coeff = np.zeros(5)
        coeff[0] = -0.25 * j_ego**2
        coeff[1] = -1/3 * j_ego * a_rel_k
        coeff[2] = j_ego * (a_tar * tg_min - v_rel)
        coeff[3] = 0
        coeff[4] = x_rel * a_rel_k * 2 - tg_min * (v_ego * a_rel_k - v_rel * a_min) * 2 - v_rel**2

        try:
            sol_t_krit = np.roots(coeff)
        except:
            print(coeff)
            print(a_rel_k)
            print(a_min)
            print(v_rel)
            print(tg_min)

        # Get valid solution (real and >0)
        real_sol_t_krit = sol_t_krit[np.isreal(sol_t_krit)]
        if len(real_sol_t_krit) > 0:
            t_krit = float(np.real(np.max(real_sol_t_krit)))
        else:
            return self.opts.cstr_no_crash_a_min

        # Calculate upper constraint
        constr_high_1 = min(self.opts.vehicle_a_max, a_min - j_ego * t_krit)

        return constr_high_1
