"""
Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import seaborn as sns
sns.set_theme(style="whitegrid")


def load_data(file: str):
    df = pd.read_csv(file)
    return df


def plot_navigator_trajectory(datafile: str, animate: bool, video_name: str,
                              dpi: int, fps: int, clean_axes: bool):
    df = load_data(file=datafile)

    t = df["t"]
    xm = df[["xm0", "xm1"]].values
    xc = df[["xc0", "xc1"]].values
    vm = df[["vm0", "vm1"]].values
    vc = df[["vc0", "vc1"]].values
    u = df[["f0", "f1"]].values

    h = [0, 0, 0, 0]

    # Turn on interactive mode and show non blocking before defining the
    # figure and axes.
    if animate:
        plt.ion()
        plt.show(block=True)

    writer = FFMpegWriter(fps=fps,
                          metadata=dict(title='Celestial navigation problem',
                                        artist='Parker Owan'),
                          bitrate=5000,
                          extra_args=['-vcodec', 'libx264'])

    fig = plt.figure(figsize=(1280 / dpi, 720 / dpi), dpi=dpi)
    fig.patch.set_facecolor('black')
    ax = fig.add_subplot(111)
    ax.patch.set_facecolor('black')
    ax.axis('equal')
    ax.plot(0, 0, '.b', ms=35)
    h[0], = ax.plot(xm[-1, 0], xm[-1, 1], '.w', ms=10)
    h[1], = ax.plot(xm[0:, 0], xm[0:, 1], '-w', lw=2)
    h[2], = ax.plot(xc[-1, 0], xc[-1, 1], '.y', ms=4)
    h[3], = ax.plot(xc[0:, 0], xc[0:, 1], '-y', lw=1)

    if clean_axes:
        ax.axis('off')

    # Draw the figure.  The pause is needed because the GUI events happen while
    # the main code is sleeping, including drawing.
    if animate:
        plt.draw()
        plt.pause(0.001)

    with writer.saving(fig, video_name, dpi=dpi):
        for i in range(0, len(t), 5):
            if animate:
                h[0].set_data(xm[i, 0], xm[i, 1])
                h[1].set_data(xm[:i, 0], xm[:i, 1])
                h[2].set_data(xc[i, 0], xc[i, 1])
                h[3].set_data(xc[:i, 0], xc[:i, 1])
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.001)

                writer.grab_frame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process and plot navigator experiment data")
    parser.add_argument('--datafile',
                        action="store",
                        dest="datafile",
                        default="/libsia/data/navigator.csv",
                        type=str,
                        help="File path of the experiment data csv")
    parser.add_argument('--animate',
                        action="store",
                        dest="animate",
                        default=True,
                        type=bool,
                        help="Animate the plot")
    parser.add_argument('--video_name',
                        action="store",
                        default="navigator-animated.mp4",
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
                        type=bool,
                        help="Remove axes when plotting data")
    args = parser.parse_args()

    plot_navigator_trajectory(args.datafile,
                              animate=args.animate,
                              video_name=args.video_name,
                              dpi=args.dpi,
                              fps=args.fps,
                              clean_axes=args.clean_axes)
    plt.show()
