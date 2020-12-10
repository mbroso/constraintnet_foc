from . import TD3


def load_policy(opts, env):
    kwargs = {
        "state_dim": env.observation_space.shape[0],
        "action_dim": env.action_space.shape[0],
        "max_action": float(max(abs(opts.vehicle_a_min), abs(opts.vehicle_a_max))),
        "a_min": opts.vehicle_a_min,
        "a_max": opts.vehicle_a_max,
        "opts": opts
    }

    # Initialize policy
    if opts.policy == "TD3":
        policy = TD3.TD3(**kwargs)
    else:
        raise ValueError("Unknown policy!")

    return policy
