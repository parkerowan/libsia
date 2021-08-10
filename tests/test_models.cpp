/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include <gtest/gtest.h>
#include <sia/sia.h>

TEST(Models, LinearGaussianDynamics) {
  Eigen::Matrix<double, 1, 1> F, G, Q;
  F << -0.1;
  G << 0.1;
  Q << 0.2;

  sia::LinearGaussianDynamics a(F, G, Q);
  EXPECT_TRUE(a.F().isApprox(F));
  EXPECT_TRUE(a.G().isApprox(G));
  EXPECT_TRUE(a.Q().isApprox(Q));

  F << -0.2;
  G << 0.2;
  Q << 0.5;
  a.setF(F);
  a.setG(G);
  a.setQ(Q);
  EXPECT_TRUE(a.F().isApprox(F));
  EXPECT_TRUE(a.G().isApprox(G));
  EXPECT_TRUE(a.Q().isApprox(Q));

  Eigen::Matrix<double, 1, 1> x, u;
  x << 1;
  u << 10;
  EXPECT_TRUE(a.f(x, u).isApprox(F * x + G * u));
  EXPECT_TRUE(a.F(x, u).isApprox(F));
  EXPECT_TRUE(a.G(x, u).isApprox(G));
  EXPECT_TRUE(a.Q(x, u).isApprox(Q));

  sia::Gaussian state = a.dynamics(x, u);
  EXPECT_TRUE(state.mean().isApprox(F * x + G * u));
  EXPECT_TRUE(state.covariance().isApprox(Q));
}

TEST(Models, LinearGaussianMeasurement) {
  Eigen::Matrix<double, 1, 1> H, R;
  H << 2;
  R << 0.1;

  sia::LinearGaussianMeasurement a(H, R);
  EXPECT_TRUE(a.H().isApprox(H));
  EXPECT_TRUE(a.R().isApprox(R));

  H << 1;
  R << 0.2;
  a.setH(H);
  a.setR(R);
  EXPECT_TRUE(a.H().isApprox(H));
  EXPECT_TRUE(a.R().isApprox(R));

  Eigen::Matrix<double, 1, 1> x;
  x << 1;
  EXPECT_TRUE(a.h(x).isApprox(H * x));
  EXPECT_TRUE(a.H(x).isApprox(H));
  EXPECT_TRUE(a.R(x).isApprox(R));

  sia::Gaussian observation = a.measurement(x);
  EXPECT_TRUE(observation.mean().isApprox(H * x));
  EXPECT_TRUE(observation.covariance().isApprox(R));
}

TEST(Models, LinearGaussianDynamicsCT) {
  Eigen::Matrix<double, 1, 1> A, B, Qpsd;
  A << -0.1;
  B << 0.1;
  Qpsd << 0.5;

  double dt = 0.1;
  sia::LinearGaussianDynamicsCT a(A, B, Qpsd, dt);
  EXPECT_TRUE(a.A().isApprox(A));
  EXPECT_TRUE(a.B().isApprox(B));
  EXPECT_TRUE(a.Q().isApprox(Qpsd * dt));
  EXPECT_EQ(a.getType(), sia::LinearGaussianDynamicsCT::BACKWARD_EULER);
  EXPECT_DOUBLE_EQ(a.getTimeStep(), dt);

  A << -0.2;
  B << 0.2;
  Qpsd << 2;
  dt = 0.01;
  a.setA(A);
  a.setB(B);
  a.setQpsd(Qpsd);
  a.setType(sia::LinearGaussianDynamicsCT::FORWARD_EULER);
  a.setTimeStep(dt);
  EXPECT_TRUE(a.A().isApprox(A));
  EXPECT_TRUE(a.B().isApprox(B));
  EXPECT_TRUE(a.Q().isApprox(Qpsd * dt));
  EXPECT_EQ(a.getType(), sia::LinearGaussianDynamicsCT::FORWARD_EULER);
  EXPECT_DOUBLE_EQ(a.getTimeStep(), dt);

  Eigen::Matrix<double, 1, 1> x, u;
  x << 1;
  u << 10;
  EXPECT_TRUE(a.f(x, u).isApprox(a.F() * x + a.G() * u));
  EXPECT_TRUE(a.F(x, u).isApprox(a.F()));
  EXPECT_TRUE(a.G(x, u).isApprox(a.G()));
  EXPECT_TRUE(a.Q(x, u).isApprox(a.Q()));

  sia::Gaussian state = a.dynamics(x, u);
  EXPECT_TRUE(state.mean().isApprox(a.F() * x + a.G() * u));
  EXPECT_TRUE(state.covariance().isApprox(a.Q()));
}

TEST(Models, LinearGaussianMeasurementCT) {
  Eigen::Matrix<double, 1, 1> H, Rpsd;
  H << 2;
  Rpsd << 0.1;

  double dt = 0.1;
  sia::LinearGaussianMeasurementCT a(H, Rpsd, dt);
  EXPECT_TRUE(a.H().isApprox(H));
  EXPECT_TRUE(a.R().isApprox(Rpsd / dt));
  EXPECT_DOUBLE_EQ(a.getTimeStep(), dt);

  H << 1;
  Rpsd << 0.2;
  dt = 0.01;
  a.setH(H);
  a.setRpsd(Rpsd);
  a.setTimeStep(dt);
  EXPECT_TRUE(a.H().isApprox(H));
  EXPECT_TRUE(a.R().isApprox(Rpsd / dt));
  EXPECT_DOUBLE_EQ(a.getTimeStep(), dt);

  Eigen::Matrix<double, 1, 1> x;
  x << 1;
  EXPECT_TRUE(a.h(x).isApprox(H * x));
  EXPECT_TRUE(a.H(x).isApprox(H));
  EXPECT_TRUE(a.R(x).isApprox(a.R()));

  sia::Gaussian observation = a.measurement(x);
  EXPECT_TRUE(observation.mean().isApprox(H * x));
  EXPECT_TRUE(observation.covariance().isApprox(a.R()));
}

TEST(Models, NonlinearGaussianDynamics) {
  Eigen::Matrix<double, 1, 1> F, G, Q;
  F << -0.1;
  G << 0.1;
  Q << 0.2;

  // Try out the linear case
  auto dynamics = [F, G](const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
    return F * x + G * u;
  };

  sia::NonlinearGaussianDynamics a(dynamics, Q);
  EXPECT_TRUE(a.Q().isApprox(Q));

  Q << 0.5;
  a.setQ(Q);
  EXPECT_TRUE(a.Q().isApprox(Q));

  Eigen::Matrix<double, 1, 1> x, u;
  x << 1;
  u << 10;
  EXPECT_TRUE(a.f(x, u).isApprox(dynamics(x, u)));
  EXPECT_NEAR(a.F(x, u)(0, 0), F(0, 0), 1e-6);
  EXPECT_NEAR(a.G(x, u)(0, 0), G(0, 0), 1e-6);
  EXPECT_TRUE(a.Q(x, u).isApprox(Q));

  sia::Gaussian state = a.dynamics(x, u);
  EXPECT_TRUE(state.mean().isApprox(dynamics(x, u)));
  EXPECT_TRUE(state.covariance().isApprox(Q));
}

TEST(Models, NonlinearGaussianMeasurement) {
  Eigen::Matrix<double, 1, 1> H, R;
  H << 2;
  R << 0.1;

  // Try out the linear case
  auto measurement = [H](const Eigen::VectorXd& x) { return H * x; };

  sia::NonlinearGaussianMeasurement a(measurement, R);
  EXPECT_TRUE(a.R().isApprox(R));

  R << 0.2;
  a.setR(R);
  EXPECT_TRUE(a.R().isApprox(R));

  Eigen::Matrix<double, 1, 1> x;
  x << 1;
  EXPECT_TRUE(a.h(x).isApprox(measurement(x)));
  EXPECT_NEAR(a.H(x)(0, 0), H(0, 0), 1e-6);
  EXPECT_TRUE(a.R(x).isApprox(R));

  sia::Gaussian observation = a.measurement(x);
  EXPECT_TRUE(observation.mean().isApprox(measurement(x)));
  EXPECT_TRUE(observation.covariance().isApprox(R));
}

TEST(Models, NonlinearGaussianDynamicsCT) {
  Eigen::Matrix<double, 1, 1> A, B, Qpsd;
  A << -0.1;
  B << 0.1;
  Qpsd << 0.5;

  // Try out the linear case
  auto dynamics = [A, B](const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
    return A * x + B * u;
  };

  double dt = 0.1;
  sia::NonlinearGaussianDynamicsCT a(dynamics, Qpsd, dt);
  EXPECT_TRUE(a.Q().isApprox(Qpsd * dt));
  EXPECT_DOUBLE_EQ(a.getTimeStep(), dt);

  Qpsd << 2;
  dt = 0.01;
  a.setQpsd(Qpsd);
  a.setTimeStep(dt);
  EXPECT_TRUE(a.Q().isApprox(Qpsd * dt));
  EXPECT_DOUBLE_EQ(a.getTimeStep(), dt);

  Eigen::Matrix<double, 1, 1> x, u;
  x << 1;
  u << 10;

  // Exact discretization for 1st order system
  double F = exp(A(0, 0) * dt);
  double G = (F - 1) / A(0, 0) * B(0, 0);
  EXPECT_NEAR(a.f(x, u)(0), F * x(0) + G * u(0), 1e-6);
  EXPECT_NEAR(a.F(x, u)(0, 0), F, 1e-6);
  EXPECT_NEAR(a.G(x, u)(0, 0), G, 1e-6);
  EXPECT_TRUE(a.Q(x, u).isApprox(a.Q()));

  sia::Gaussian state = a.dynamics(x, u);
  EXPECT_NEAR(state.mean()(0), F * x(0) + G * u(0), 1e-6);
  EXPECT_TRUE(state.covariance().isApprox(a.Q()));
}

TEST(Models, NonlinearGaussianMeasurementCT) {
  Eigen::Matrix<double, 1, 1> H, Rpsd;
  H << 2;
  Rpsd << 0.1;

  // Try out the linear case
  auto measurement = [H](const Eigen::VectorXd& x) { return H * x; };

  double dt = 0.1;
  sia::NonlinearGaussianMeasurementCT a(measurement, Rpsd, dt);
  EXPECT_TRUE(a.R().isApprox(Rpsd / dt));
  EXPECT_DOUBLE_EQ(a.getTimeStep(), dt);

  Rpsd << 0.2;
  dt = 0.01;
  a.setRpsd(Rpsd);
  a.setTimeStep(dt);
  EXPECT_TRUE(a.R().isApprox(Rpsd / dt));
  EXPECT_DOUBLE_EQ(a.getTimeStep(), dt);

  Eigen::Matrix<double, 1, 1> x;
  x << 1;
  EXPECT_TRUE(a.h(x).isApprox(measurement(x)));
  EXPECT_NEAR(a.H(x)(0, 0), H(0, 0), 1e-6);
  EXPECT_TRUE(a.R(x).isApprox(a.R()));

  sia::Gaussian observation = a.measurement(x);
  EXPECT_TRUE(observation.mean().isApprox(measurement(x)));
  EXPECT_TRUE(observation.covariance().isApprox(a.R()));
}

TEST(Models, Simulate) {
  Eigen::Matrix<double, 1, 1> F, G, H, Q, R;
  F << -0.1;
  G << 0.1;
  H << 2;
  Q << 1;
  R << 0.1;

  sia::LinearGaussianDynamics dynamics(F, G, Q);
  sia::LinearGaussianMeasurement measurement(H, R);

  Eigen::VectorXd x(1);
  x << 0;
  std::vector<Eigen::VectorXd> states;
  states.emplace_back(x);
  states.emplace_back(x);

  Eigen::MatrixXd u(1, 3);
  u << 1, -2, 1;
  std::vector<Eigen::MatrixXd> controls;
  controls.emplace_back(u);
  controls.emplace_back(u);

  sia::Trajectory traj = sia::simulate(dynamics, measurement, x, u, false);
  for (std::size_t k = 0; k < static_cast<std::size_t>(u.cols()); ++k) {
    x = F * x + G * u.col(k);
    Eigen::VectorXd y = H * x;
    ASSERT_TRUE(traj.states.col(k).isApprox(x));
    ASSERT_TRUE(traj.controls.col(k).isApprox(u.col(k)));
    ASSERT_TRUE(traj.measurements.col(k).isApprox(y));
  }

  sia::Trajectories trajs =
      sia::simulate(dynamics, measurement, states, u, false);
  EXPECT_EQ(trajs.size(), 2);
  for (std::size_t i = 0; i < trajs.data().size(); ++i) {
    x << states.at(i);
    for (std::size_t k = 0; k < static_cast<std::size_t>(u.cols()); ++k) {
      x = F * x + G * u.col(k);
      Eigen::VectorXd y = H * x;
      ASSERT_TRUE(trajs.states(k).col(i).isApprox(x));
      ASSERT_TRUE(trajs.controls(k).col(i).isApprox(u.col(k)));
      ASSERT_TRUE(trajs.measurements(k).col(i).isApprox(y));
    }
  }

  trajs = sia::simulate(dynamics, measurement, states, controls, false);
  EXPECT_EQ(trajs.size(), 2);
  for (std::size_t i = 0; i < trajs.data().size(); ++i) {
    x << states.at(i);
    for (std::size_t k = 0; k < static_cast<std::size_t>(u.cols()); ++k) {
      x = F * x + G * controls.at(i).col(k);
      Eigen::VectorXd y = H * x;
      ASSERT_TRUE(trajs.states(k).col(i).isApprox(x));
      ASSERT_TRUE(trajs.controls(k).col(i).isApprox(u.col(k)));
      ASSERT_TRUE(trajs.measurements(k).col(i).isApprox(y));
    }
  }
}
