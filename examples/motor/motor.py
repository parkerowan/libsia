"""
Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause
"""

# Import the libSIA python bindings and numpy
import pysia as sia
import numpy as np
import argparse

# Import plotting helpers
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")


def create_system(voltage_noise: float = 1e2,
                  current_noise: float = 1e0,
                  dt: float = 1e-4) -> sia.LinearGaussianDynamicsCT:
    """Creates the system model"""
    Lm = 1e-3  # Motor inductance (H)
    Rm = 1.2  # Motor resistance (ohm)
    Kb = 2.0  # Motor back EMF (V-s/rad)
    Jm = 1e-3  # Motor inertia (kg-m^2)
    Bm = 1e-4  # Motor viscous damping (N-m-s/rad)
    Kt = 2.0  # Motor torque constant (N-m/A)
    A = np.array([[-Rm / Lm, -Kb / Lm, 0], [Kt / Jm, -Bm / Jm, 0], [0, 1, 0]])
    B = np.array([[1 / Lm], [0], [0]])
    Q = np.diag([voltage_noise, current_noise, 1e-8])
    return sia.LinearGaussianDynamicsCT(A, B, Q, dt)


def create_cost(current_cost: float = 1e3,
                velocity_cost: float = 1e2,
                position_cost: float = 1e8) -> sia.QuadraticCost:
    """Create the cost function"""
    Qlqr = np.diag([current_cost, velocity_cost, position_cost])
    Rlqr = np.diag([1])
    xd = np.zeros(3)
    return sia.QuadraticCost(Qlqr, Qlqr, Rlqr, xd)


def create_controller(dynamics: sia.LinearGaussianDynamicsCT,
                      cost: sia.QuadraticCost,
                      horizon: int = 31) -> sia.LQR:
    """Create the controller"""
    return sia.LQR(dynamics, cost, horizon)


def init_state() -> np.array:
    """Return the initial state"""
    return np.array([1, 1, 0])


def plot_results(t: np.array, xd: np.array, x: np.array, u: np.array):
    """Plots the resulting data"""
    f, ax = plt.subplots(nrows=4, ncols=1, figsize=(15, 10))
    sns.despine(f, left=True, bottom=True)
    ax[0].plot(t, xd[0, :], "--k", ms=5, label="Desired")
    ax[0].plot(t, x[0, :], "-b", ms=5, label="Controlled")
    ax[0].set_ylabel("Current (A)")
    ax[0].set_title("LQR Tracking for Linear DC Motor")
    ax[0].legend()

    ax[1].plot(t, xd[1, :], "--k", ms=5, label="Desired")
    ax[1].plot(t, x[1, :], "-b", ms=5, label="Controlled")
    ax[1].set_ylabel("Velocity (rad/s)")

    ax[2].plot(t, xd[2, :], "--k", ms=5, label="Desired")
    ax[2].plot(t, x[2, :], "-b", ms=5, label="Controlled")
    ax[2].set_ylabel("Position (rad)")

    ax[3].plot(t, u[0, :], "-k", ms=5)
    ax[3].set_ylabel("Voltage (V)")
    ax[3].set_xlabel("Time (s)")
    plt.show()


def main(num_steps: int, dt: float, voltage_noise: float, current_noise: float,
         current_cost: float, velocity_cost: float, position_cost: float,
         horizon: int, show_plots: bool):
    """Run an MPC on a motor position tracking problem"""

    # Create the system and cost function
    dynamics = create_system(voltage_noise, current_noise, dt)
    cost = create_cost(current_cost, velocity_cost, position_cost)

    # Create the controller
    mpc = create_controller(dynamics, cost, horizon)

    # Simulate
    n = num_steps
    t = np.arange(0, n * dt, dt)
    u = np.zeros((1, n))
    x = np.zeros((3, n))
    x[:, 0] = init_state()

    # Generate a position sin wave to track, zero in velocity and current
    fd = 50  # Hz
    Ad = 0.01  # rad/s
    yd = Ad * np.sin(2 * np.pi * fd * t)
    xd = np.array([[0, 0, 1]]).T @ np.reshape(yd, (1, n))

    # Simulate
    state = sia.Gaussian(3)
    for k in range(n - 1):
        # Set the trajectory to track for the cost, append the last state when the horizon runs out of runway
        Xd = xd[:, k:horizon + k].T.tolist()
        [Xd.append(Xd[-1]) for i in range(horizon - (n - k))]
        cost.setTrajectory(Xd)

        # Compute LQR
        state.setMean(x[:, k])
        u[:, k] = mpc.policy(state)

        # Simulate dynamics forward
        x[:, k + 1] = dynamics.dynamics(x[:, k], u[:, k]).sample()

    # Plot the results
    if show_plots:
        plot_results(t, xd, x, u)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a particle filter for the Lorenz attractor")
    parser.add_argument('--num_steps',
                        action="store",
                        dest="num_steps",
                        default=1000,
                        type=int,
                        help="Number of steps to simulate")
    parser.add_argument('--dt',
                        action="store",
                        dest="dt",
                        default=1E-4,
                        type=float,
                        help="Simulation time step")
    parser.add_argument('--voltage_noise',
                        action="store",
                        dest="voltage_noise",
                        default=1E2,
                        type=float,
                        help="Voltage noise variance")
    parser.add_argument('--current_noise',
                        action="store",
                        dest="current_noise",
                        default=1E0,
                        type=float,
                        help="Electrical current noise variance")
    parser.add_argument('--current_cost',
                        action="store",
                        dest="current_cost",
                        default=1E3,
                        type=float,
                        help="Electrical current cost")
    parser.add_argument('--velocity_cost',
                        action="store",
                        dest="velocity_cost",
                        default=1E2,
                        type=float,
                        help="Angular velocity cost")
    parser.add_argument('--position_cost',
                        action="store",
                        dest="position_cost",
                        default=1E8,
                        type=float,
                        help="Angular position cost")
    parser.add_argument('--horizon',
                        action="store",
                        dest="horizon",
                        default=31,
                        type=int,
                        help="Controller horizon")
    parser.add_argument('--show_plots',
                        action="store",
                        dest="show_plots",
                        default=True,
                        type=bool,
                        help="Show plots")
    args = parser.parse_args()

    main(num_steps=args.num_steps,
         dt=args.dt,
         voltage_noise=args.voltage_noise,
         current_noise=args.current_noise,
         current_cost=args.current_cost,
         velocity_cost=args.velocity_cost,
         position_cost=args.position_cost,
         horizon=args.horizon,
         show_plots=args.show_plots)
