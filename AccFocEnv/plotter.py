"""Plotting of episode in second process. Allows interacting with the plot while training is running.
"""
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class ProcessPlotter(object):
    """Started in a seperate process to plot episode data.
    """
    def __init__(self, opts):
        """Init

        Args:
            opts: Namespace object containing options.
        """
        self.opts = opts
        self.fixed = False

    def terminate(self):
        """Terminate plotting thread.
        """
        plt.close()

    def key_press(self, event):
        """If # is pressing, plotting is paused to allow for interaction with the plot.
        Is called by keypress event.
        """
        if event.key == "#":
            self.fixed = not self.fixed

            if self.fixed:
                self.fig.canvas.set_window_title(f"Experiment {self.opts.experiment_id} - TRAINING STOPPED. Press # to continue.")
            else:
                self.fig.canvas.set_window_title(f"Experiment {self.opts.experiment_id}")

    def call_back(self):
        """Callback for plotting.
        """
        while not self.fixed and self.pipe.poll():
            raw_data = self.pipe.recv()
            if raw_data is None:
                self.terminate()
                return False
            else:
                data, steps, prefix = raw_data
                t = data["t"][0:steps]
                # plot varibles according to options
                for i, plot_variable in enumerate(self.opts.plot_variables):
                    self.axs[i].cla()
                    variables = plot_variable.replace(" ", "").split("+")
                    for variable in variables:
                        if variable in ["a_min", "a_max"]:
                            self.axs[i].plot(t, data[variable][0:steps], color="#0A55B0", ls="--", lw=1.2)
                        else:
                            self.axs[i].plot(t, data[variable][0:steps], label=variable)
                    self.axs[i].legend(loc="upper left", bbox_to_anchor=(1.01, 1.01), borderaxespad=0, frameon=True)
                    if "Hw" in variables:  # plot boundaries for ideal headway
                        self.axs[i].axhline(self.opts.desired_headway * 0.95, color='g')
                        self.axs[i].axhline(self.opts.desired_headway * 1.05, color='g')

                for i, title in enumerate(self.opts.plot_titles):
                    self.axs[i].set_title(title)

                if self.opts.plot_grid:
                    for i, _ in enumerate(self.opts.plot_variables):
                        self.axs[i].grid()

                # also save plot to disc
                if prefix:
                    self.fig.canvas.set_window_title(f"Experiment {self.opts.experiment_id} - {prefix}")

                    episode_reward = np.sum(data["reward"])
                    p = self.opts.path_img / Path(f"{prefix}_{steps:06.0f}_{episode_reward:011.0f}.png")
                    self.fig.savefig(str(p))

                plt.pause(0.01)
                self.fig.canvas.draw()
        return True

    def __call__(self, pipe):
        """Start plotting in different thread.
        """
        print("Starting plotting in different thread ...")

        self.pipe = pipe

        self.fig, self.axs = plt.subplots(len(self.opts.plot_variables), 1, sharex="all", num="timeseries_plots")
        if matplotlib.get_backend() == "Qt5Agg":
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        plt.pause(0.01)
        self.fig.subplots_adjust(left=0.04, right=0.91, top=0.96, bottom=0.04)
        self.fig.canvas.set_window_title(f"Experiment {self.opts.experiment_id}")

        timer = self.fig.canvas.new_timer(interval=1000)
        timer.add_callback(self.call_back)
        timer.start()

        self.fig.canvas.mpl_connect('key_press_event', self.key_press)

        plt.show()


class Plotter(object):
    """Plotter class, that receives a plotting request and sends it to another thread.
    """

    def __init__(self, opts):
        """Init

        Args:
            opts: Namespace object containing opts.
        """
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = ProcessPlotter(opts)
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True
        )
        self.plot_process.start()

    def plot(self, data, finished=False):
        """Receives plotting request
        
        Args:
            data: State dict with data of last episode
            finished: If set to True, the plotting thread is stopped.
        """
        send = self.plot_pipe.send
        if finished:
            send(None)
        else:
            send(data)
