"""
Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause
"""

# Import the libSIA python bindings and numpy
import pysia as sia
import numpy as np
import argparse

# Import plotting helpers
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation, rc
from mpl_toolkits.mplot3d import Axes3D

# Colormap for particles
cmap = cm.plasma


def create_dynamics(q: float, dt: float) -> sia.NonlinearGaussianDynamicsCT:
    """Creates the system dynamics model"""
    # Lorenz attractor chaotic parameters
    rho = 28
    sig = 10
    bet = 8 / 3

    # Lorenz attractor dynamics equation.  For NonlinearGaussian systems, we
    # pass a lambda function to the constructor
    f = lambda x, u: np.array([
        sig * (x[1] - x[0]),
        x[0] * (rho - x[2]) - x[1],
        x[0] * x[1] - bet * x[2],
    ])

    # Suppose that noise is added to all 3 channels
    Q = q * np.identity(3)

    # Create the system
    return sia.NonlinearGaussianDynamicsCT(f, Q, dt)


def create_measurement(r: float, dt: float) -> sia.NonlinearGaussianMeasurementCT:
    """Creates the system measurement model"""
    # Suppose we measure a linear combination of states in the measurement
    # equation.  For NonlinearGaussian systems, we pass a lambda function to
    # the constructor
    h = lambda x: np.array([
        x[0] - x[1],
        x[2],
    ])
    R = r * np.identity(2)

    # Create the system
    return sia.NonlinearGaussianMeasurementCT(h, R, dt)


def create_estimators(dynamics: sia.NonlinearGaussianDynamicsCT,
                      measurement: sia.NonlinearGaussianMeasurementCT,
                      num_particles: int,
                      resample_threshold: float,
                      roughening_factor: float,
                      buffer_size: int):
    """Creates the estimators"""
    # Initialize a gaussian belief
    gaussian = sia.Gaussian(mean=np.array([0, 0, 20]),
                            covariance=1e3 * np.identity(3))

    # Initialize the extended kalman filter
    ekf = sia.EKF(dynamics=dynamics, measurement=measurement, state=gaussian)

    # Initialize a particle belief
    particles = sia.Particles.uniform(lower=np.array([-30, -30, -10]),
                                      upper=np.array([30, 30, 50]),
                                      num_particles=num_particles,
                                      weighted_stats=True)

    # Initialize the particle filter
    pf = sia.PF(dynamics=dynamics,
                measurement=measurement,
                particles=particles,
                resample_threshold=resample_threshold,
                roughening_factor=roughening_factor)

    # Initial true state
    x = np.array([-10, 5, 20])
    state = sia.Gaussian(3)
    state.setMean(x)

    # Initialize the runner
    runner = sia.Runner({"ekf": ekf, "pf": pf}, buffer_size=buffer_size)

    return (runner, state, ekf, pf)


def create_animate_3d_sim(dynamics, measurement, runner, state, ekf, pf, num_steps, dpi):
    """Creates an animation function for the 3d sim plot"""
    # Set up the figure, the axis, and the plot element we want to animate
    particles = pf.getBelief()
    xp = particles.values()
    wp = particles.weights()
    x = state.mean()

    fig = plt.figure(figsize=(1280 / dpi, 720 / dpi), dpi=dpi)
    fig.patch.set_facecolor('black')
    ax0 = fig.add_subplot(111, projection='3d')
    ax0.patch.set_facecolor('black')
    ax0.set_xlim([-30, 30])
    ax0.set_ylim([-30, 30])
    ax0.set_zlim([0, 50])
    cs = 100 * wp / max(wp)
    scatter = ax0.scatter(xp[0, :],
                          xp[1, :],
                          xp[2, :],
                          color=cmap(cs.astype(int)),
                          marker='.',
                          s=25,
                          alpha=1,
                          linewidths=None)
    line, = ax0.plot(x[0], x[1], x[2], '-w')
    point, = ax0.plot(x[0], x[1], x[2], '.w', ms=25)

    plt.tight_layout()
    plt.axis('off')

    # Render the animation
    return animation.FuncAnimation(fig,
                                   step_animate_3d_sim,
                                   fargs=(dynamics, measurement, runner, state, ekf, pf,
                                          scatter, line, point, num_steps),
                                   frames=num_steps,
                                   interval=20,
                                   blit=False)


def step_animate_3d_sim(i, dynamics, measurement, runner, state, ekf, pf, scatter, line,
                        point, num_steps):
    """Animation function for 3d sim. This is called sequentially."""
    recorder = runner.recorder()
    particles = pf.getBelief()

    if i > 0:
        # There is not forcing term to the system so we just assign zeros
        u = np.zeros(3)
        x = state.mean()

        # This steps the system state, takes a measurement and steps each estimator
        state.setMean(runner.stepAndEstimate(dynamics, measurement, x, u))

        # Update state trajectory plot
        line.set_data(recorder.getStates()[:2, :])
        line.set_3d_properties(recorder.getStates()[2, :])

        # Update the state point
        point.set_data(recorder.getStates()[:2, -1])
        point.set_3d_properties(recorder.getStates()[2, -1])

    # Update particle plot
    xp = particles.values()
    wp = particles.weights()
    scatter._offsets3d = (xp[0, :], xp[1, :], xp[2, :])
    cs = 100 * wp / max(wp)
    scatter._facecolor3d = cmap(cs.astype(int))
    scatter._edgecolor3d = cmap(cs.astype(int))


def plot_estimates(dynamics, measurement, runner, state, ekf, pf, num_steps, dpi):
    """Plots estimates of the particle filter and ekf"""
    # Set up the figure, the axis, and the plot element we want to animate
    recorder = runner.recorder()
    state = recorder.getStates()
    pf_mean = recorder.getEstimateMeans("pf")
    pf_mode = recorder.getEstimateModes("pf")
    pf_var = recorder.getEstimateVariances("pf")
    ekf_mean = recorder.getEstimateMeans("ekf")
    ekf_var = recorder.getEstimateVariances("ekf")

    fig = plt.figure(figsize=(1280 / dpi, 720 / dpi), dpi=dpi)
    t = np.linspace(1, num_steps, num_steps)
    ax = [0] * 3
    for i in range(3):
        ax[i] = fig.add_subplot(3, 1, (i + 1))
        plt.sca(ax[i])
        ax[i].fill_between(t,
                           pf_mean[i, :] - 3 * np.sqrt(pf_var[i, :]),
                           pf_mean[i, :] + 3 * np.sqrt(pf_var[i, :]),
                           alpha=0.2,
                           label="PF")
        ax[i].plot(t, pf_mode[i, :], lw=1)
        ax[i].fill_between(t,
                           ekf_mean[i, :] - 3 * np.sqrt(ekf_var[i, :]),
                           ekf_mean[i, :] + 3 * np.sqrt(ekf_var[i, :]),
                           alpha=0.2,
                           label="EKF")
        ax[i].plot(t, ekf_mean[i, :], lw=1)
        ax[i].plot(t, state[i, :], "-k", label="Truth")
        if i == 2:
            ax[i].legend(frameon=False, loc='lower center', ncol=3)
        plt.ylabel("State " + str(i))
        plt.axis("on")
        plt.box(on=None)
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(True)
        ax[i].set_yticks([])


def main(num_steps: int, dt: float, process_noise: float, measurement_noise: float,
         num_particles: int, resample_threshold: float,
         roughening_factor: float, video_name: str, dpi: int,
         show_plots: bool):
    """"Run estimators on a Lorenz attractor estimation problem"""

    # Create the system
    dynamics = create_dynamics(process_noise, dt)
    measurement = create_measurement(measurement_noise, dt)

    # Setup the estimators
    runner, state, ekf, pf = create_estimators(dynamics, measurement, num_particles,
                                               resample_threshold,
                                               roughening_factor, num_steps)

    # Create the animation function
    anim = create_animate_3d_sim(dynamics, measurement, runner, state, ekf, pf, num_steps,
                                 dpi)

    # Render and save the animation
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30,
                    metadata=dict(title='Particle filter Lorenz attractor',
                                  artist='Parker Owan'),
                    bitrate=5000,
                    extra_args=['-vcodec', 'libx264'])

    anim.save(video_name, writer=writer, dpi=dpi)

    # Plot the estimates
    plot_estimates(dynamics, measurement, runner, state, ekf, pf, num_steps, dpi)

    # Show the animation
    if show_plots:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run estimators on a Lorenz attractor estimation problem")
    parser.add_argument('--num_steps',
                        action="store",
                        dest="num_steps",
                        default=300,
                        type=int,
                        help="Number of time steps to animate")
    parser.add_argument('--dt',
                        action="store",
                        dest="dt",
                        default=0.01,
                        type=float,
                        help="Time step (s)")
    parser.add_argument('--process_noise',
                        action="store",
                        dest="process_noise",
                        default=1E-1,
                        type=float,
                        help="Process noise variance")
    parser.add_argument('--measurement_noise',
                        action="store",
                        dest="measurement_noise",
                        default=1E1,
                        type=float,
                        help="Process noise variance")
    parser.add_argument('--num_particles',
                        action="store",
                        dest="num_particles",
                        default=1000,
                        type=int,
                        help="Number of particles to initialize")
    parser.add_argument('--resample_threshold',
                        action="store",
                        dest="resample_threshold",
                        default=0.4,
                        type=float,
                        help="Threshold [0, 1] to resample particles")
    parser.add_argument('--roughening_factor',
                        action="store",
                        dest="roughening_factor",
                        default=2E-3,
                        type=float,
                        help="Magnitude of roughening [0, \infty]")
    parser.add_argument('--video_name',
                        action="store",
                        dest="video_name",
                        default="lorenz-animated.mp4",
                        type=str,
                        help="Name of the rendered video file")
    parser.add_argument('--dpi',
                        action="store",
                        dest="dpi",
                        default=150,
                        type=int,
                        help="Resolution of the images to render")
    parser.add_argument('--show_plots',
                        action="store",
                        dest="show_plots",
                        default=True,
                        type=bool,
                        help="Show and animate plots")
    args = parser.parse_args()

    main(num_steps=args.num_steps,
         dt=args.dt,
         process_noise=args.process_noise,
         measurement_noise=args.measurement_noise,
         num_particles=args.num_particles,
         resample_threshold=args.resample_threshold,
         roughening_factor=args.roughening_factor,
         video_name=args.video_name,
         dpi=args.dpi,
         show_plots=args.show_plots)
