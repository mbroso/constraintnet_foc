"""Evaluation of trained agents.
"""
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import glob
from pathlib import Path
import gym
from tqdm import tqdm
import torch
import os
import importlib
import copy
import collections

from options.opt_manager import OptManager
import utils
import algorithms
import AccFocEnv


class Results:
    """Results class allows loading reward during training and trained agent by experiment id.
    """
    def __init__(self, experiment_id):
        """Initialize

        Args:
            experiment_id: Experiment to load.
        """
        self.experiment_id = experiment_id
        self.folder = f'./experiments/{experiment_id:04}/'

        self.opts = self.get_opts()

    def get_opts(self):
        """Load options of experiment.

        Returns:
            Namespace object containing options.
        """
        opt_manager = OptManager(
            config_default=Path(self.folder) / "config.yaml",
            opt_def_default=Path(self.folder) / "opt_def.yaml",
            default_flow_style=False
        )
        opts = opt_manager.parse(ignore_remaining=True)
        return opts

    def load_results(self):
        """Load numpy file containing reward and safety during training.
        """
        data = np.load(glob.glob(self.folder + 'results/*.npy')[0])

        self.t = np.array(range(data.shape[0])) * self.opts.eval_freq

        self.reward = data[:, :, 0]
        self.reward_per_step = data[:, :, 1]
        self.steps = data[:, :, 2]
        self.safety = data[:, :, 3]

    def load_eval_env(self):
        """Loads an environment in which the trained agent can be evaluated.
        """
        env, state_dim, action_dim, max_action, plotter = utils.load_env(self.opts)
        policy = algorithms.load_policy(self.opts, env)

        eval_env = gym.make(self.opts.env, opts=self.opts, plotter=plotter)
        eval_env.seed(31415926)

        self.eval_env = eval_env
        self.policy = policy

    def load_policy(self, training_steps):
        """Loads the pretrained policy.

        Args:
            training_steps: Number of training steps of pretrained policy to load.
        """
        if not hasattr(self, "policy"):
            opts_cpy = copy.deepcopy(self.opts)
            opts_cpy.render = False
            env_, state_dim, action_dim, max_action, plotter = utils.load_env(opts_cpy)
            self.policy = algorithms.load_policy(opts_cpy, env_)

        self.policy.load(f"experiments/{self.opts.experiment_id:04}/models/{self.opts.policy}_{self.opts.env}_{self.opts.seed}_{training_steps}")

    def evaluate(self, training_steps, evaluation_episodes=30, render=False):
        """Evaluate the trained agent.

        Args:
            training_steps: Number of training steps of pretrained policy to load.
            evaluation_episodes: Number of episodes for evaluation.
            render: Whether episodes are rendered or not.
        
        Returns:
            Dict containing keys:
                safety: Average safety metric.
                crash_rate: Crash rate during evaluation episodes.
                tracking_error: Average tracking error.
                discomfort: Average discomfort value.
                avg_reward: Average reward during evaluation episodes.
                avg_reward_per_step: Average reward per evaluation step.
        """
        render_old, self.opts.render = self.opts.render, render

        if not hasattr(self, "eval_env"):
            self.load_eval_env()

        eval_env = self.eval_env
        self.load_policy(training_steps)

        steps = 0
        reward = 0.
        results = []
        us = []
        safety, discomfort, tracking_error = [], [], []
        for eval_episode in range(evaluation_episodes):
            timesteps = 0
            episod_reward = 0
            state, done = eval_env.reset(), False
            while not done and eval_env.steps < eval_env._max_episode_steps - 1:
                action = self.policy.select_action(state)
                state, r, done, _ = eval_env.step(action)
                reward += r
                episod_reward += r
                timesteps += 1
                steps += 1
            results.append([episod_reward, episod_reward / timesteps, timesteps])

            metric = self.eval_env.calc_metrics()
            safety.append(metric["safety"])
            discomfort.append(metric["discomfort"])
            tracking_error.append(metric["tracking_error"])

        eval_env.render_episode()

        results = {}
        results["safety"] = np.array(safety).mean()
        results["crash_rate"] = (np.array(safety) < 0.01).sum() / len(safety)
        results["tracking_error"] = np.array(tracking_error).mean()
        results["discomfort"] = np.array(discomfort).mean()
        results["avg_reward"] = reward / evaluation_episodes
        results["avg_reward_per_step"] = reward / steps

        return results


def calc_metrics(args):
    """Evaluate agents and calc metrics.

    Args:
        args:
    """
    metrics = {
        "crash_rate": np.zeros(len(args.exp_ids)),
        "safety": np.zeros(len(args.exp_ids)),
        "discomfort": np.zeros(len(args.exp_ids)),
        "tracking_error": np.zeros(len(args.exp_ids)),
    }

    for i, exp_id in enumerate(args.exp_ids):
        result = Results(experiment_id=exp_id)
        m = result.evaluate(args.training_steps, evaluation_episodes=args.episodes)
        
        metrics["crash_rate"][i] = m["crash_rate"]
        metrics["safety"][i] = m["safety"]
        metrics["discomfort"][i] = m["discomfort"]
        metrics["tracking_error"][i] = m["tracking_error"]

    
    print(f"Evaluation of experiments {', '.join(map(str, args.exp_ids))} for {args.episodes} episode each:")
    print(f"   Crash rate: {metrics['crash_rate'].mean()*100:5.1f} +- {metrics['crash_rate'].std()*100:5.1f}%")
    print(f"   Safety: {metrics['safety'].mean():5.2f} +- {metrics['safety'].std():5.2f}")
    print(f"   Discomfort: {metrics['discomfort'].mean():5.2f} +- {metrics['discomfort'].std():5.2f}")
    print(f"   Tracking error: {metrics['tracking_error'].mean():5.1f} +- {metrics['tracking_error'].std():5.1f}")


def get_mean_crash_rate(exp_ids):
    """Returns the mean crash rate during training for the specified experiment ids.

    Args:
        exp_ids: Experiment ids.

    Returns:
        Tuple containing:
            t: Training timesteps.
            crash_rate: Average crash rate.
    """
    safety_list = None
    t = None
    for i, exp_id in enumerate(exp_ids):
        res = Results(experiment_id=exp_id)
        res.load_results()
        
        if safety_list is None:
            num_eval = res.reward_per_step.shape[1]
            tau_dem = res.opts.desired_headway
            safety_list = np.zeros((res.reward_per_step.shape[0], len(exp_ids)*num_eval))
            t = res.t
        
        safety_list[:, i*num_eval:(i+1)*num_eval] = res.safety
        
    crash_rate = (safety_list * tau_dem <= 0.0).mean(axis=1)
    return t, crash_rate


def get_mean_reward(exp_ids):
    """Returns the mean reward and standard deviation during training for the specified experiment ids.

    Args:
        exp_ids: Experiment ids.

    Returns:
        Tuple containing:
            t: Training timesteps
            mean: Mean reward
            lower: Mean reward - standard deviation
            upper: Mean reward + standard deviation
    """
    reward_list = None
    t = None
    for i, exp_id in enumerate(exp_ids):
        res = Results(exp_id)
        res.load_results()
        
        if reward_list is None:
            reward_list = np.zeros((res.reward_per_step.shape[0], len(exp_ids)))
            t = res.t
        
        reward_list[:, i] = res.reward.mean(axis=1)
        
    mean = reward_list.mean(axis=1)  
    lower = mean - reward_list.std(axis=1)
    upper = mean + reward_list.std(axis=1)
    
    return t, mean, lower, upper


def enable_latex_style():
    """Enable Latex font in matplotlib.
    """
    #matplotlib.rcParams["font.family"] = ["Latin Modern Roman"]
    #matplotlib.rc("font", **{"family":"serif","serif":["Latin Modern Roman"]})
    matplotlib.rc("text", usetex=True)
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams.update({'font.size': 22})
    matplotlib.rc('axes', titlesize=22)


if __name__ == "__main__":
    fontsize = 25
    fontsize_legend = 20
    parser = argparse.ArgumentParser(description='Evaluation of trained FOC models.')

    parser.add_argument("--method", help="Can be 'metrics', 'crash_rate' or 'reward'.")

    # Metrics
    parser.add_argument("--exp_ids", type=int, nargs="+", help="Experiment ids to calculate metrics for.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes.")
    parser.add_argument("--training_steps", type=int, default=1_000_000, help="Number of training steps of pretrained model.")

    # Comparison of training reward and crash rate.
    parser.add_argument("--exp_ids_unconstrained", type=int, nargs="+", help="Experiment ids of unconstrained FOC.")
    parser.add_argument("--exp_ids_unconstrained_clipped", type=int, nargs="+", help="Experiment ids of clipped unconstrained FOC.")
    parser.add_argument("--exp_ids_ConstraintNet", type=int, nargs="+", help="Experiment ids of ConstraintNet FOC.")

    args = parser.parse_args()
    
    if args.method == "metrics":
        calc_metrics(args)
    elif args.method == "crash_rate":
        enable_latex_style()

        t, crash_rate_unconstrained = get_mean_crash_rate(args.exp_ids_unconstrained)
        t, crash_rate_unconstrained_clipped = get_mean_crash_rate(args.exp_ids_unconstrained_clipped)
        _, crash_rate_ConstraintNet = get_mean_crash_rate(args.exp_ids_ConstraintNet)

        plt.figure(figsize=(6.5,3.5))
        plt.plot(t/1e5, crash_rate_unconstrained*100, c="k")
        plt.plot(t/1e5, crash_rate_unconstrained_clipped*100, c="red", ls=":")
        plt.plot(t/1e5, crash_rate_ConstraintNet*100., c="b", ls="--")
        plt.xlabel("Time steps ($1\mathrm{e}5$)", fontsize=fontsize)
        plt.ylabel("Crash rate (\%)", fontsize=fontsize)
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
        plt.ylim(-3, 68)
        plt.xlim(0, 10)
        plt.legend(["Unconstrained", "Clipped unconstrained", "\\textit{ConstraintNet}"],
                fontsize=fontsize_legend, frameon=False)
        plt.savefig("foc_crash_rate_comparison.png", bbox_inches="tight")
    elif args.method == "reward":
        enable_latex_style()

        plt.figure(figsize=(6.5,3.73))
        
        t, mean, lower, upper = get_mean_reward(args.exp_ids_unconstrained)
        id_max = np.argmax(mean)
        t_max = t[id_max]
        print('Unconstrained: Maximum score for time step: ', t_max)
        plt.plot(t/1e5, mean/1e5, c="k", label="Unconstrained")
        plt.axvline(t_max/1e5, 0.55, 0.85, c='k')
        plt.axvline(t_max/1e5, 0, 0.1, c='k')
        plt.fill_between(t/1e5, lower/1e5, upper/1e5, alpha=0.15, color="k")
        
        t, mean, lower, upper = get_mean_reward(args.exp_ids_unconstrained_clipped)
        id_max = np.argmax(mean)
        t_max = t[id_max]
        print('Clipped unconstrained: Maximum score for time step: ', t_max)
        plt.plot(t/1e5, mean/1e5, c="red", ls=':', label="Clipped unconstrained")
        plt.axvline(t_max/1e5, 0, 0.45, c='red', ls=':')
        #plt.fill_between(t/1e5, lower/1e5, upper/1e5, alpha=0.15, color="red")
        
        t, mean, lower, upper = get_mean_reward(args.exp_ids_ConstraintNet)
        id_max = np.argmax(mean)
        t_max = t[id_max]
        print('ConstraintNet: Maximum score for time step: ', t_max)
        plt.plot(t/1e5, mean/1e5, c="b", ls="--", label="\\textit{ConstraintNet}")
        plt.axvline(t_max/1e5, 0.55, 0.9, c='b', ls="--")
        plt.axvline(t_max/1e5, 0, 0.1, c='b', ls="--")
        plt.fill_between(t/1e5, lower/1e5, upper/1e5, alpha=0.15, color="b")
        plt.xlabel("Time steps ($1\mathrm{e}5$)", fontsize=fontsize)
        plt.ylabel("Average return ($1\mathrm{e}5$)", fontsize=fontsize)
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(4))
        plt.ylim(-0.9, 1.4)
        plt.xlim(0, 10)
        plt.legend(frameon=True, edgecolor='white', facecolor='white', framealpha=0.4, fontsize=fontsize_legend, bbox_to_anchor=(0.3,0.62))
        plt.savefig("foc_reward_comparison.png", bbox_inches="tight")
    else:
        raise ValueError("Unsupported value for method!")
