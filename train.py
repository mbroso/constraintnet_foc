import os
import utils
import datetime
from options.opt_manager import OptManager
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import gym
import time
import datetime
from pathlib import Path
import shutil

import algorithms

import AccFocEnv
from tqdm import tqdm, trange


def eval_policy(opts, policy, plotter=None, train_steps=0):
    """Evaluate policy for several episodes with a fixed seed and return average reward.

    Args:
        opts: Namespace object with options.
        policy: Policy to evaluatio
        plotter: Plotter object if plotting of episodes is enabled.
        train_steps: Number of training steps so far.

    Returns:
        Array of (total reward, average reward, episode steps, safety metric) per evaluation episode.
    """
    # Create eval env and seed it
    eval_env = gym.make(opts.env, opts=opts, plotter=plotter)
    eval_env.seed(opts.seed + np.random.randint(0, 100000))

    reward = 0.
    results = []
    for eval_episode in range(opts.evaluation_episodes):
        episode_steps = 0
        episode_reward = 0
        state, done = eval_env.reset(), False
        while not done and eval_env.steps < eval_env._max_episode_steps - 1:
            action = policy.select_action(state)
            state, r, done, _ = eval_env.step(action)
            reward += r
            episode_reward += r
            episode_steps += 1
        metrics = eval_env.calc_metrics()
        results.append([episode_reward, episode_reward / episode_steps, episode_steps, metrics["safety"]])
    results = np.array(results)

    # Render last episode if render is enabled
    if opts.render:
        eval_env.render_episode(f"{(train_steps+1):07d}_eval")

    avg_return = reward / opts.evaluation_episodes

    tqdm.write(f"=====> Evaluation over {opts.evaluation_episodes} episodes. Average return: {avg_return:.3f}")
    return results

if __name__ == '__main__':
    # Load options and create neccessary folder
    opt_manager = OptManager(
        config_default=Path() / "options/config.yaml",
        opt_def_default=Path() / "options/opt_def.yaml",
        default_flow_style=False
    )
    opts = opt_manager.parse()
    experiment_path = Path("./experiments") / Path(f"{opts.experiment_id:04}")
    if experiment_path.exists():
        print(f"\nERROR: {experiment_path} already exists! Either delete this folder or change the experiment ID!")
        exit()

    # Add a_min and a_max to observations and acceleration plot if a constrained actor was specified.
    if opts.actor_type.lower().find("constrained") == 0:
        if "a_max" not in opts.observations:
            opts.observations.insert(0, "a_max")
            print("INFO: Added a_max to observations because constrained actor was specified!")
        if "a_min" not in opts.observations:
            opts.observations.insert(0, "a_min")
            print("INFO: Added a_max to observations because constrained actor was specified!")
        if opts.plot_variables[0].find("a_") == 0:   # First subplot is plot of accelerations, plot constraints as well
            if "a_max" not in opts.plot_variables[0].split("+"):
                opts.plot_variables[0] += "+a_max"
                print("INFO: Added a_max to acceleration plot because constrained actor was specified")
            if "a_min" not in opts.plot_variables[0].split("+"):
                opts.plot_variables[0] += "+a_min"
                print("INFO: Added a_min to acceleration plot because constrained actor was specified")

    # Save config and options definition.
    experiment_path.mkdir()
    opt_manager.obj2yaml(opts, dest=experiment_path / Path("config.yaml"))
    opt_manager.opt_def2yaml(experiment_path / Path("opt_def.yaml"))

    opts.path_results = experiment_path / Path("results")
    opts.path_results.mkdir()
    opts.path_models = experiment_path / Path("models")
    opts.path_models.mkdir()

    # If rendering of episodes is enabled, create folders for images.
    if opts.render:
        opts.path_img = experiment_path / Path("img")
        opts.path_img.mkdir()
        opts.path_img_train = experiment_path / Path("img/train")
        opts.path_img_train.mkdir()


    file_name = f"{opts.policy}_{opts.env}_{opts.seed}"
    print("---------------------------------------")
    print(f"Policy: {opts.policy}, Env: {opts.env}, Seed: {opts.seed}")
    print("---------------------------------------")

    torch.set_num_threads(opts.num_threads)
    env, state_dim, action_dim, max_action, plotter = utils.load_env(opts)
    policy = algorithms.load_policy(opts, env)

    if opts.load_model != "":
        policy_file = file_name if opts.load_model == "default" else opts.load_model
        policy.load(f"./{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(opts.replaybuffer_size))

    # Evaluate untrained policy
    evaluations = [eval_policy(opts, policy, plotter=plotter, train_steps=0)]

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    last_episode_timesteps = 0
    episode_num = 0
    steps_since_last_save = 0

    killer = utils.GracefulKiller()

    for t in trange(int(opts.max_timesteps), desc=f"Exp {opts.experiment_id}"):
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < opts.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(state)
                + np.random.normal(0, max_action * opts.expl_noise, size=action_dim)
            ).clip(opts.vehicle_a_min, opts.vehicle_a_max)

        # clip action when using constrained reinforcement learning (clip random actions at beginning and later action with noise to be within constraints)
        if "a_min" in opts.observations:
            action = action.clip(state[opts.observations.index("a_min")], state[opts.observations.index("a_max")])

        # Perform action
        # done signal
        #   done = 0: not done, episode can continue
        #   done = 1: done, because simulated time ended
        #   done = 2: done, because agent ended in terminal step (e.g. crash)
        next_state, reward, done, _ = env.step(action)

        done_bool = 1 if done >= 2 else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if replay_buffer.size >= opts.start_timesteps:
            policy.train(replay_buffer)

        if done:
            if opts.render:
                if t - steps_since_last_save >= opts.render_min_interval or last_episode_timesteps >= opts.render_min_interval:
                    env.render_episode(f"train/{t:07d}_train")
                    steps_since_last_save = t

            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            tqdm.write(f"Total T: {t+1:7} Episode Num: {episode_num+1:5} Episode T: {episode_timesteps:5} Reward: {episode_reward:12.2f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            last_episode_timesteps = episode_timesteps
            episode_timesteps = 0
            episode_num += 1

        # Evaluate episode
        if (t + 1) % opts.eval_freq == 0:
            evaluations.append(eval_policy(opts, policy, plotter=plotter, train_steps=t))
            np.save(opts.path_results / Path(f"{file_name}"), evaluations)
            if opts.save_model:
                policy.save(opts.path_models / Path(f"{file_name}_{t+1}"))

        # Allow gracefull stopping by pressing Ctrl+C
        if killer.kill_now:
            tqdm.write("Training was interrupted by user!")
            break
