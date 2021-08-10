/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "tests/helpers.h"

#include <gtest/gtest.h>
#include <sia/sia.h>

TEST(Controllers, QuadraticCost) {
  Eigen::MatrixXd Q(3, 3), Qf(3, 3), R(2, 2);
  Eigen::VectorXd xd(3);
  Qf << 100, 0, 0, 0, 50, 0, 0, 0, 10;
  Q << 10, 0, 0, 0, 5, 0, 0, 0, 1;
  R << 3, 0, 0, 1;
  xd << 1, 2, 3;
  sia::QuadraticCost cost_a(Qf, Q, R);

  Eigen::VectorXd x(3), u(2);
  x << 3, 6, 9;
  u << -1, -2;

  // Evaluation and terms
  EXPECT_DOUBLE_EQ(cost_a.c(x, u, 0), x.dot(Q * x) / 2 + u.dot(R * u) / 2);
  EXPECT_DOUBLE_EQ(cost_a.cf(x), x.dot(Qf * x) / 2);
  EXPECT_TRUE(cost_a.Qf().isApprox(Qf));
  EXPECT_TRUE(cost_a.Q().isApprox(Q));
  EXPECT_TRUE(cost_a.R().isApprox(R));

  // Derivatives
  EXPECT_TRUE(cost_a.cx(x, u, 0).isApprox(Q * x));
  EXPECT_TRUE(cost_a.cu(x, u, 0).isApprox(R * u));
  EXPECT_TRUE(cost_a.cxx(x, u, 0).isApprox(Q));
  EXPECT_TRUE(cost_a.cux(x, u, 0).isApprox(Eigen::MatrixXd::Zero(3, 2)));
  EXPECT_TRUE(cost_a.cuu(x, u, 0).isApprox(R));
  EXPECT_TRUE(cost_a.cfx(x).isApprox(Qf * x));
  EXPECT_TRUE(cost_a.cfxx(x).isApprox(Qf));

  // Single target
  sia::QuadraticCost cost_b(Qf, Q, R, xd);
  EXPECT_DOUBLE_EQ(cost_b.c(xd, Eigen::VectorXd::Zero(2), 0), 0);

  // Trajectories
  std::vector<Eigen::VectorXd> X, U;
  X.emplace_back(x);
  X.emplace_back(Eigen::VectorXd::Zero(3));
  U.emplace_back(Eigen::VectorXd::Zero(2));
  U.emplace_back(Eigen::VectorXd::Zero(2));

  // Trajectory tracking
  sia::QuadraticCost cost_c(Qf, Q, R, X);
  EXPECT_DOUBLE_EQ(cost_c.eval(X, U), 0);
  EXPECT_DOUBLE_EQ(cost_c.c(X.at(0), U.at(0), 0), 0);
  EXPECT_DOUBLE_EQ(cost_c.c(X.at(1), U.at(1), 1), 0);
  EXPECT_DOUBLE_EQ(cost_c.cf(X.at(1)), 0);
  EXPECT_TRUE(cost_c.xd(0).isApprox(x));
  EXPECT_TRUE(cost_c.xd(1).isApprox(Eigen::VectorXd::Zero(3)));
}

TEST(Controllers, FunctionalCost) {
  Eigen::MatrixXd Q(3, 3), Qf(3, 3), R(2, 2);
  Eigen::VectorXd xd(3);
  Qf << 100, 0, 0, 0, 50, 0, 0, 0, 10;
  Q << 10, 0, 0, 0, 5, 0, 0, 0, 1;
  R << 3, 0, 0, 1;
  xd << 1, 2, 3;

  sia::TerminalCostFunction terminal_cost = [&](const Eigen::VectorXd& x) {
    return x.dot(Qf * x) / 2;
  };

  sia::RunningCostFunction running_cost = [&](const Eigen::VectorXd& x,
                                              const Eigen::VectorXd& u) {
    return x.dot(Q * x) / 2 + u.dot(R * u) / 2;
  };

  sia::FunctionalCost cost(terminal_cost, running_cost);

  Eigen::VectorXd x(3), u(2);
  x << 3, 6, 9;
  u << -1, -2;

  // Evaluation and terms
  Eigen::MatrixXd E;
  EXPECT_DOUBLE_EQ(cost.c(x, u, 0), running_cost(x, u));
  EXPECT_DOUBLE_EQ(cost.cf(x), terminal_cost(x));

  E = cost.cx(x, u, 0) - Q * x;
  EXPECT_NEAR(E.norm(), 0.0, 2e-8);

  E = cost.cu(x, u, 0) - R * u;
  EXPECT_NEAR(E.norm(), 0.0, 2e-8);

  E = cost.cxx(x, u, 0) - Q;
  EXPECT_NEAR(E.norm(), 0.0, 2e-2);

  E = cost.cux(x, u, 0) - Eigen::MatrixXd::Zero(3, 2);
  EXPECT_NEAR(E.norm(), 0.0, 2e-2);

  E = cost.cuu(x, u, 0) - R;
  EXPECT_NEAR(E.norm(), 0.0, 2e-2);

  E = cost.cfx(x) - Qf * x;
  EXPECT_NEAR(E.norm(), 0.0, 2e-7);

  E = cost.cfxx(x) - Qf;
  EXPECT_NEAR(E.norm(), 0.0, 2e-1);
}

TEST(Controllers, LQR) {
  sia::LinearGaussianDynamics dynamics = createIntegratorDynamics();
  sia::QuadraticCost cost = createTestCost();
  std::size_t horizon = 2;
  sia::LQR mpc(dynamics, cost, horizon);

  // Simulate a step forward and check the cost is at a local minima using a
  // stencil around the optimal control
  sia::Gaussian state(1);
  Eigen::VectorXd eps(1), x(1);
  eps << 1e-2;
  x << 10;

  // Compute the policy for a single step
  state.setMean(x);
  Eigen::VectorXd u = mpc.policy(state);
  auto xm = dynamics.f(state.mean(), u);
  auto xp = dynamics.f(state.mean(), u + eps);
  auto xn = dynamics.f(state.mean(), u - eps);

  // Compute the cost for each point in the stencil
  double cm = cost.eval(std::vector<Eigen::VectorXd>{xm},
                        std::vector<Eigen::VectorXd>{u});
  double cp = cost.eval(std::vector<Eigen::VectorXd>{xp},
                        std::vector<Eigen::VectorXd>{u});
  double cn = cost.eval(std::vector<Eigen::VectorXd>{xn},
                        std::vector<Eigen::VectorXd>{u});

  // Expect cmm to be the minimum
  EXPECT_LT(cm, cp);
  EXPECT_LT(cm, cn);
}

TEST(Controllers, iLQR) {
  sia::LinearGaussianDynamics dynamics = createIntegratorDynamics();
  sia::QuadraticCost cost = createTestCost();
  std::size_t max_iter = 10;
  std::size_t max_backsteps = 10;
  Eigen::VectorXd zero = Eigen::VectorXd::Zero(1);
  std::vector<Eigen::VectorXd> u0{zero, zero};
  sia::iLQR mpc(dynamics, cost, u0, max_iter, max_backsteps);

  // Simulate a step forward and check the cost is at a local minima using a
  // stencil around the optimal control
  sia::Gaussian state(1);
  Eigen::VectorXd eps(1), x(1);
  eps << 1e-2;
  x << 10;

  // Compute the policy for a single step
  state.setMean(x);
  Eigen::VectorXd u = mpc.policy(state);
  auto xm = dynamics.f(state.mean(), u);
  auto xp = dynamics.f(state.mean(), u + eps);
  auto xn = dynamics.f(state.mean(), u - eps);

  // Compute the cost for each point in the stencil
  double cm = cost.eval(std::vector<Eigen::VectorXd>{xm},
                        std::vector<Eigen::VectorXd>{u});
  double cp = cost.eval(std::vector<Eigen::VectorXd>{xp},
                        std::vector<Eigen::VectorXd>{u});
  double cn = cost.eval(std::vector<Eigen::VectorXd>{xn},
                        std::vector<Eigen::VectorXd>{u});

  // Expect cmm to be the minimum
  EXPECT_LT(cm, cp);
  EXPECT_LT(cm, cn);
}

TEST(Controllers, MPPI) {
  sia::LinearGaussianDynamics dynamics = createIntegratorDynamics();
  sia::QuadraticCost cost = createTestCost();
  std::size_t num_samples = 100;
  Eigen::VectorXd zero = Eigen::VectorXd::Zero(1);
  std::vector<Eigen::VectorXd> u0{zero, zero};
  Eigen::MatrixXd sigma(1, 1);
  sigma << 1;
  sia::MPPI mpc(dynamics, cost, u0, num_samples, sigma);

  // Simulate a step forward and check the cost is at a local minima using a
  // stencil around the optimal control
  sia::Gaussian state(1);
  Eigen::VectorXd eps(1), x(1);

  // Here we use sigma as the step for the stencil around the control because it
  // is the covariance of the perturbation used to generate the control
  eps = sigma;
  x << 10;

  // Compute the policy for a single step
  state.setMean(x);
  Eigen::VectorXd u = mpc.policy(state);
  auto xm = dynamics.f(state.mean(), u);
  auto xp = dynamics.f(state.mean(), u + eps);
  auto xn = dynamics.f(state.mean(), u - eps);

  // Compute the cost for each point in the stencil
  double cm = cost.eval(std::vector<Eigen::VectorXd>{xm},
                        std::vector<Eigen::VectorXd>{u});
  double cp = cost.eval(std::vector<Eigen::VectorXd>{xp},
                        std::vector<Eigen::VectorXd>{u});
  double cn = cost.eval(std::vector<Eigen::VectorXd>{xn},
                        std::vector<Eigen::VectorXd>{u});

  // Expect cmm to be the minimum
  EXPECT_LT(cm, cp);
  EXPECT_LT(cm, cn);
}