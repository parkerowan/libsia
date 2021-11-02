/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/distribution.h"

#include <Eigen/Dense>

namespace sia {

/// Helper to evaluate the logProb of a distribution for several samples (rows
/// are number of dimensions, cols are number of samples)
const Eigen::VectorXd logProb(const Distribution& distribution,
                              const Eigen::MatrixXd& x);

/// Helper to evaluate the logProb of a distribution for several samples from a
/// 1D distribution (each element of x is a sample)
const Eigen::VectorXd logProb1d(const Distribution& distribution,
                                const Eigen::VectorXd& x);

/// Helper to evaluate the logProb of a distribution for several samples from a
/// 2D distribution (each element of x and y is a sample)
const Eigen::VectorXd logProb2d(const Distribution& distribution,
                                const Eigen::VectorXd& x,
                                const Eigen::VectorXd& y);

}  // namespace sia
