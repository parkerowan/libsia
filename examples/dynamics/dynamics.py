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

# For explanations of these dynamical systems, 
# see: http://www.ccs.fau.edu/~fuchs/pub/Huys_nonlin.pdf


def create_spiral(process_noise: float = 1e-4,
                  dt: float = 1e-1) -> sia.LinearGaussianDynamicsCT:
    A = np.array([[0, -1], [1, -1]])
    B = np.array([[0], [1]])
    Q = process_noise * np.eye(2)
    return sia.LinearGaussianDynamicsCT(A, B, Q, dt)


def create_saddle_point(process_noise: float = 1e-4,
                        dt: float = 1e-1) -> sia.LinearGaussianDynamicsCT:
    A = np.array([[0, 1], [1, 0]])
    B = np.array([[0], [1]])
    Q = process_noise * np.eye(2)
    return sia.LinearGaussianDynamicsCT(A, B, Q, dt)


def create_degenerate_node(process_noise: float = 1e-4,
                           dt: float = 1e-1) -> sia.LinearGaussianDynamicsCT:
    A = np.array([[-1, 1], [0, -1]])
    B = np.array([[0], [1]])
    Q = process_noise * np.eye(2)
    return sia.LinearGaussianDynamicsCT(A, B, Q, dt)


def create_homoclinic_orbit(process_noise: float = 1e-4,
                            dt: float = 1e-1) -> sia.NonlinearGaussianDynamicsCT:
    f = lambda x, u: np.array([
        x[1] * (1 - x[1]),
        x[0] + u[0]
    ])
    Q = process_noise * np.eye(2)
    return sia.NonlinearGaussianDynamicsCT(f, Q, dt)


def create_heteroclinic_orbit(process_noise: float = 1e-4,
                              dt: float = 1e-1) -> sia.NonlinearGaussianDynamicsCT:
    f = lambda x, u: np.array([
        x[1] - x[1]**3,
        -x[0] - x[1]**2 + u[0]
    ])
    Q = process_noise * np.eye(2)
    return sia.NonlinearGaussianDynamicsCT(f, Q, dt)


def create_van_der_pol(process_noise: float = 1e-4,
                       dt: float = 1e-1) -> sia.NonlinearGaussianDynamicsCT:
    mu = 2.0
    f = lambda x, u: np.array([
        x[1],
        mu * (1 - x[0]**2) * x[1] - x[0] + u[0]
    ])
    Q = process_noise * np.eye(2)
    return sia.NonlinearGaussianDynamicsCT(f, Q, dt)
