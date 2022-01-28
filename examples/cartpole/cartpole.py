"""
Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import seaborn as sns
import ast  # Because argparse bool doesn't work
sns.set_theme(style="whitegrid")


def load_data(file: str):
    df = pd.read_csv(file)
    return df


def plot_cartpole_trajectory(datafile: str, animate: bool, trace: bool,
                             video_name: str, dpi: int, fps: int,
                             clean_axes: bool):
    df = load_data(file=datafile)

    t = df["t"]
    p = df["p"]
    a = df["a"]
    v = df["v"]
    w = df["w"]
    u = df["f"]

    # Tip point
    l = 0.75
    xt = p + l * np.sin(a)
    yt = l * np.cos(a)

    h = [0, 0, 0, 0, 0, 0]

    # Turn on interactive mode and show non blocking before defining the
    # figure and axes.
    if animate:
        plt.ion()
        plt.show(block=True)

    writer = FFMpegWriter(fps=fps,
                          metadata=dict(title='Cartpole control problem',
                                        artist='Parker Owan'),
                          bitrate=5000,
                          extra_args=['-vcodec', 'libx264'])

    fig, ax = plt.subplots(3, 1, figsize=(1280 / dpi, 720 / dpi), dpi=dpi)
    sns.despine(fig, left=True, bottom=True)

    # show pendulum 2D
    h[0], = ax[0].plot([p[0], xt[0]], [0, yt[0]], '-k', lw=3)
    ax[0].set_title("Cartpole MPC", fontsize=18)
    ax[0].axis('equal')
    ax[0].axis('off')
    ax[0].set_ylim([-1, 1])

    # show traces
    h[1], = ax[1].plot(t, p, ".", ms=2, label="p")
    h[2], = ax[1].plot(t, a, ".", ms=2, label="a")
    h[3], = ax[1].plot(t, v, ".", ms=2, label="v")
    h[4], = ax[1].plot(t, w, ".", ms=2, label="w")
    ax[1].legend()
    ax[1].set_ylabel("States")

    h[5], = ax[2].plot(t, u, ".k", ms=2, label="u")
    ax[2].legend()
    ax[2].set_ylabel("Control")
    ax[2].set_xlabel("Time (s)")

    if clean_axes:
        ax[1].axis('off')
        ax[2].axis('off')

    # Draw the figure.  The pause is needed because the GUI events happen while
    # the main code is sleeping, including drawing.
    if animate:
        plt.draw()
        plt.pause(0.001)

    with writer.saving(fig, video_name, dpi=dpi):
        for i in range(len(t)):
            if trace:
                c = float(i) / len(t)
                ax[0].plot([p[i], xt[i]], [0, yt[i]],
                           '-',
                           color=(0, 0, c),
                           lw=1)

            if animate:
                h[0].set_data([p[i], xt[i]], [0, yt[i]])
                h[1].set_data(t[:i], p[:i])
                h[2].set_data(t[:i], a[:i])
                h[3].set_data(t[:i], v[:i])
                h[4].set_data(t[:i], w[:i])
                h[5].set_data(t[:i], u[:i])
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)

                writer.grab_frame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process and plot cartpole experiment data")
    parser.add_argument('--datafile',
                        action="store",
                        dest="datafile",
                        default="/libsia/data/cartpole.csv",
                        type=str,
                        help="File path of the experiment data csv")
    parser.add_argument('--animate',
                        action="store",
                        dest="animate",
                        default=True,
                        type=ast.literal_eval,
                        help="Animate the plot")
    parser.add_argument('--trace',
                        action="store",
                        dest="trace",
                        default=False,
                        type=ast.literal_eval,
                        help="Show the pendulum trace")
    parser.add_argument('--video_name',
                        action="store",
                        default="cartpole-animated.mp4",
                        type=str,
                        help="Name of the rendered video file")
    parser.add_argument('--dpi',
                        action="store",
                        default=150,
                        type=int,
                        help="Resolution of the images to render")
    parser.add_argument('--fps',
                        action="store",
                        default=30,
                        type=int,
                        help="Frames per second render")
    parser.add_argument('--clean_axes',
                        action="store",
                        default=False,
                        type=ast.literal_eval,
                        help="Remove axes when plotting data")
    args = parser.parse_args()

    plot_cartpole_trajectory(datafile=args.datafile,
                             animate=args.animate,
                             trace=args.trace,
                             video_name=args.video_name,
                             dpi=args.dpi,
                             fps=args.fps,
                             clean_axes=args.clean_axes)
    plt.show()
