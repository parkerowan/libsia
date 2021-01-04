/// Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include <gtest/gtest.h>
#include <sia/sia.h>

TEST(Models, linearGaussian) {
  Eigen::Matrix<double, 1, 1> F, G, C, H, Q, R;
  F << -0.1;
  G << 0.1;
  C << 0.2;
  H << 2;
  Q << 1;
  R << 0.1;

  sia::LinearGaussian a(F, G, C, H, Q, R);
  EXPECT_TRUE(a.F().isApprox(F));
  EXPECT_TRUE(a.G().isApprox(G));
  EXPECT_TRUE(a.C().isApprox(C));
  EXPECT_TRUE(a.H().isApprox(H));
  EXPECT_TRUE(a.Q().isApprox(Q));
  EXPECT_TRUE(a.R().isApprox(R));

  F << -0.2;
  G << 0.2;
  C << 0.5;
  H << 1;
  Q << 2;
  R << 0.2;
  a.setF(F);
  a.setG(G);
  a.setC(C);
  a.setH(H);
  a.setQ(Q);
  a.setR(R);
  EXPECT_TRUE(a.F().isApprox(F));
  EXPECT_TRUE(a.G().isApprox(G));
  EXPECT_TRUE(a.C().isApprox(C));
  EXPECT_TRUE(a.H().isApprox(H));
  EXPECT_TRUE(a.Q().isApprox(Q));
  EXPECT_TRUE(a.R().isApprox(R));

  Eigen::Matrix<double, 1, 1> x, u;
  x << 1;
  u << 10;
  EXPECT_TRUE(a.f(x, u).isApprox(F * x + G * u));
  EXPECT_TRUE(a.h(x).isApprox(H * x));

  EXPECT_TRUE(a.F(x, u).isApprox(F));
  EXPECT_TRUE(a.H(x).isApprox(H));

  sia::Gaussian state = a.dynamics(x, u);
  sia::Gaussian observation = a.measurement(x);
  EXPECT_TRUE(state.mean().isApprox(F * x + G * u));
  EXPECT_TRUE(state.covariance().isApprox(C * Q * C.transpose()));
  EXPECT_TRUE(observation.mean().isApprox(H * x));
  EXPECT_TRUE(observation.covariance().isApprox(R));
}

TEST(Models, linearGaussianCT) {
  Eigen::Matrix<double, 1, 1> A, B, C, H, Q, R;
  A << -0.1;
  B << 0.1;
  C << 0.2;
  H << 2;
  Q << 1;
  R << 0.1;

  double dt = 0.1;
  sia::LinearGaussianCT a(A, B, C, H, Q, R, dt);
  EXPECT_TRUE(a.A().isApprox(A));
  EXPECT_TRUE(a.B().isApprox(B));
  EXPECT_TRUE(a.C().isApprox(C));
  EXPECT_TRUE(a.H().isApprox(H));
  EXPECT_TRUE(a.Q().isApprox(Q));
  EXPECT_TRUE(a.R().isApprox(R));
  EXPECT_EQ(a.getType(), sia::LinearGaussianCT::BACKWARD_EULER);
  EXPECT_DOUBLE_EQ(a.getTimeStep(), dt);

  A << -0.2;
  B << 0.2;
  C << 0.5;
  H << 1;
  Q << 2;
  R << 0.2;
  dt = 0.01;
  a.setA(A);
  a.setB(B);
  a.setC(C);
  a.setH(H);
  a.setQ(Q);
  a.setR(R);
  a.setType(sia::LinearGaussianCT::FORWARD_EULER);
  a.setTimeStep(dt);
  EXPECT_TRUE(a.A().isApprox(A));
  EXPECT_TRUE(a.B().isApprox(B));
  EXPECT_TRUE(a.C().isApprox(C));
  EXPECT_TRUE(a.H().isApprox(H));
  EXPECT_TRUE(a.Q().isApprox(Q));
  EXPECT_TRUE(a.R().isApprox(R));
  EXPECT_EQ(a.getType(), sia::LinearGaussianCT::FORWARD_EULER);
  EXPECT_DOUBLE_EQ(a.getTimeStep(), dt);

  Eigen::Matrix<double, 1, 1> x, u;
  x << 1;
  u << 10;
  EXPECT_TRUE(a.f(x, u).isApprox(a.F() * x + a.G() * u));
  EXPECT_TRUE(a.h(x).isApprox(H * x));

  EXPECT_TRUE(a.F(x, u).isApprox(a.F()));
  EXPECT_TRUE(a.H(x).isApprox(H));

  sia::Gaussian state = a.dynamics(x, u);
  sia::Gaussian observation = a.measurement(x);
  EXPECT_TRUE(state.mean().isApprox(a.F() * x + a.G() * u));
  EXPECT_TRUE(state.covariance().isApprox(C * Q * C.transpose() * dt));
  EXPECT_TRUE(observation.mean().isApprox(H * x));
  EXPECT_TRUE(observation.covariance().isApprox(R / dt));
}

TEST(Models, nonlinearGaussian) {
  Eigen::Matrix<double, 1, 1> F, G, C, H, Q, R;
  F << -0.1;
  G << 0.1;
  C << 0.2;
  H << 2;
  Q << 1;
  R << 0.1;

  // Try out the linear case
  auto dynamics = [F, G](const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
    return F * x + G * u;
  };
  auto measurement = [H](const Eigen::VectorXd& x) { return H * x; };

  sia::NonlinearGaussian a(dynamics, measurement, C, Q, R);
  EXPECT_TRUE(a.C().isApprox(C));
  EXPECT_TRUE(a.Q().isApprox(Q));
  EXPECT_TRUE(a.R().isApprox(R));

  C << 0.5;
  Q << 2;
  R << 0.2;
  a.setC(C);
  a.setQ(Q);
  a.setR(R);
  EXPECT_TRUE(a.C().isApprox(C));
  EXPECT_TRUE(a.Q().isApprox(Q));
  EXPECT_TRUE(a.R().isApprox(R));

  Eigen::Matrix<double, 1, 1> x, u;
  x << 1;
  u << 10;
  EXPECT_TRUE(a.f(x, u).isApprox(dynamics(x, u)));
  EXPECT_TRUE(a.h(x).isApprox(measurement(x)));

  EXPECT_NEAR(a.F(x, u)(0, 0), F(0, 0), 1e-6);
  EXPECT_NEAR(a.H(x)(0, 0), H(0, 0), 1e-6);

  sia::Gaussian state = a.dynamics(x, u);
  sia::Gaussian observation = a.measurement(x);
  EXPECT_TRUE(state.mean().isApprox(dynamics(x, u)));
  EXPECT_TRUE(state.covariance().isApprox(C * Q * C.transpose()));
  EXPECT_TRUE(observation.mean().isApprox(measurement(x)));
  EXPECT_TRUE(observation.covariance().isApprox(R));
}

TEST(Models, nonlinearGaussianCT) {
  Eigen::Matrix<double, 1, 1> A, B, C, H, Q, R;
  A << -0.1;
  B << 0.1;
  C << 0.2;
  H << 2;
  Q << 1;
  R << 0.1;

  // Try out the linear case
  auto dynamics = [A, B](const Eigen::VectorXd& x, const Eigen::VectorXd& u) {
    return A * x + B * u;
  };
  auto measurement = [H](const Eigen::VectorXd& x) { return H * x; };

  double dt = 0.1;
  sia::NonlinearGaussianCT a(dynamics, measurement, C, Q, R, dt);
  EXPECT_TRUE(a.C().isApprox(C));
  EXPECT_TRUE(a.Q().isApprox(Q));
  EXPECT_TRUE(a.R().isApprox(R));

  C << 0.5;
  Q << 2;
  R << 0.2;
  a.setC(C);
  a.setQ(Q);
  a.setR(R);
  EXPECT_TRUE(a.C().isApprox(C));
  EXPECT_TRUE(a.Q().isApprox(Q));
  EXPECT_TRUE(a.R().isApprox(R));

  Eigen::Matrix<double, 1, 1> x, u;
  x << 1;
  u << 10;

  // Exact discretization for 1st order system
  double F = exp(A(0, 0) * dt);
  double G = (F - 1) / A(0, 0) * B(0, 0);
  EXPECT_NEAR(a.f(x, u)(0), F * x(0) + G * u(0), 1e-6);
  EXPECT_TRUE(a.h(x).isApprox(measurement(x)));

  EXPECT_NEAR(a.F(x, u)(0, 0), F, 1e-6);
  EXPECT_NEAR(a.H(x)(0, 0), H(0, 0), 1e-6);

  sia::Gaussian state = a.dynamics(x, u);
  sia::Gaussian observation = a.measurement(x);
  EXPECT_NEAR(state.mean()(0), F * x(0) + G * u(0), 1e-6);
  EXPECT_TRUE(state.covariance().isApprox(C * Q * C.transpose() * dt));
  EXPECT_TRUE(observation.mean().isApprox(measurement(x)));
  EXPECT_TRUE(observation.covariance().isApprox(R / dt));

  double dt2 = 0.01;
  a.setTimeStep(dt2);
  EXPECT_DOUBLE_EQ(a.getTimeStep(), dt2);
  state = a.dynamics(x, u);
  observation = a.measurement(x);
  EXPECT_TRUE(state.covariance().isApprox(C * Q * C.transpose() * dt2));
  EXPECT_TRUE(observation.covariance().isApprox(R / dt2));
}
