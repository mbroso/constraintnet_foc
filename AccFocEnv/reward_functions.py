"""This module implements different reward functions.
"""

import numpy as np
import math


def my_reward_function(opts):
    """Choose reward function specified by opts.

    Args:
        opts: Namespace object with options.
    """
    reward_function = opts.reward_function

    if reward_function == "CACC Paper":
        return reward_CACC_paper
    else:
        raise ValueError("Unknown reward function!")


def reward_CACC_paper(state, opts):
    """Reward function for safety and function based on https://ieeexplore.ieee.org/document/5876320
    It is extended by a cost for commanded acceleration and change in commanded acceleration to increase comfort.

    Args:
        opts: Namespace object with options.

    Returns:
        Reward signal.
    """
    Hw, dHw = state["Hw"], state["dHw"]
    desired_Hw = opts.desired_headway
    assert 0 <= Hw <= 10.1, f"ERROR: Headway value of {Hw} is invalid in reward function!"

    if Hw < desired_Hw * 0.25:
        reward = -100
    elif desired_Hw * 0.25 <= Hw < desired_Hw * 0.875:
        reward = -5 if dHw < 0 else 1
    elif desired_Hw * 0.875 <= Hw < desired_Hw * 0.95:
        reward = 5
    elif desired_Hw * 0.95 <= Hw < desired_Hw * 1.05:
        reward = 50
    elif desired_Hw * 1.05 <= Hw < desired_Hw * 1.125:
        reward = 5
    elif desired_Hw * 1.125 <= Hw < 10:
        reward = 1 if dHw < 0 else -1
    elif 10 <= Hw:
        reward = 0.5 if dHw < 0 else -100

    # Cost for acceleration and jerk to favour comfortable driving.
    reward -= opts.reward_control_weight * state["a_dem"]**2
    # Note: We do not divide by env_dt, consequently the time factor needs to be
    # considered in reward_control_jerk_weight
    reward -= opts.reward_control_jerk_weight * (state["a_dem"] - state["last_a_dem"])**2

    return reward
