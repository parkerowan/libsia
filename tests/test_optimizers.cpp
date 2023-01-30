/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include <gtest/gtest.h>
#include <sia/sia.h>

const double SMALL_NUMBER = 1e-6;

double quadratic(const Eigen::VectorXd& x) {
  return x.dot(x);
};

// See the branin function https://www.sfu.ca/~ssurjano/branin.html
double branin(const Eigen::VectorXd& x) {
  double a = 1;
  double b = 5.1 / (4 * pow(M_PI, 2));
  double c = 5 / M_PI;
  double r = 6;
  double s = 10;
  double t = 1 / (8 * M_PI);
  return a * pow(x(1) - b * pow(x(0), 2) + c * x(0) - r, 2) +
         s * (1 - t) * cos(x(0)) + s;
};

TEST(Optimizers, CheckTestFunctions) {
  Eigen::VectorXd xopt = Eigen::VectorXd::Zero(1);
  EXPECT_NEAR(0, quadratic(xopt), SMALL_NUMBER);

  Eigen::VectorXd xopt0 = Eigen::Vector2d{-M_PI, 12.275};
  Eigen::VectorXd xopt1 = Eigen::Vector2d{M_PI, 2.275};
  Eigen::VectorXd xopt2 = Eigen::Vector2d{9.42478, 2.475};
  EXPECT_NEAR(0.397887, branin(xopt0), SMALL_NUMBER);
  EXPECT_NEAR(0.397887, branin(xopt1), SMALL_NUMBER);
  EXPECT_NEAR(0.397887, branin(xopt2), SMALL_NUMBER);
}

TEST(Optimizers, GradientDescent) {
  // bounds inactive at optimimum
  Eigen::VectorXd lb = Eigen::Vector2d{-10, -10};
  Eigen::VectorXd ub = Eigen::Vector2d{10, 10};
  sia::GradientDescent a(lb, ub);
  Eigen::VectorXd xopt = a.minimize(quadratic, Eigen::Vector2d{1.2, -2.5});
  ASSERT_EQ(xopt.size(), 2);
  EXPECT_NEAR(xopt(0), 0, 1e-6);
  EXPECT_NEAR(xopt(1), 0, 1e-6);
  EXPECT_EQ(a.dimension(), 2);

  // bounds inactive at optimimum - 3 manual multiple starts
  xopt = a.minimize(quadratic,
                    std::vector<Eigen::VectorXd>{Eigen::Vector2d{1.2, -2.5},
                                                 Eigen::Vector2d{-15, 13.0},
                                                 Eigen::Vector2d{7.2, -8.0}});
  ASSERT_EQ(xopt.size(), 2);
  EXPECT_NEAR(xopt(0), 0, 1e-6);
  EXPECT_NEAR(xopt(1), 0, 1e-6);

  // bounds active at optimum
  lb = Eigen::Vector2d{-5, -5};
  ub = Eigen::Vector2d{5, -2};
  sia::GradientDescent b(lb, ub);
  xopt = b.minimize(quadratic, Eigen::Vector2d{1.2, -6});
  ASSERT_EQ(xopt.size(), 2);
  EXPECT_NEAR(xopt(0), 0, 1e-6);
  EXPECT_NEAR(xopt(1), -2, 1e-6);

  // bounds active at 2 optima - 10 internal multiple starts
  lb = Eigen::Vector2d{0, -10};
  ub = Eigen::Vector2d{7, 10};
  sia::GradientDescent::Options opt;
  opt.n_starts = 10;
  sia::GradientDescent c(lb, ub, opt);
  xopt = c.minimize(branin);
  ASSERT_EQ(xopt.size(), 2);
  EXPECT_NEAR(xopt(0), M_PI, 1e-3);
  EXPECT_NEAR(xopt(1), 2.275, 1e-3);
}

TEST(Optimizers, BayesianOptimizer) {
  Eigen::VectorXd lb = -Eigen::VectorXd::Ones(1);
  Eigen::VectorXd ub = Eigen::VectorXd::Ones(1);
  sia::NoiseKernel noise_kernel = sia::NoiseKernel(1e-2);
  sia::SEKernel se_kernel = sia::SEKernel(0.1, 1.0);
  sia::CompositeKernel kernel = noise_kernel + se_kernel;
  sia::BayesianOptimizer bo(lb, ub, kernel);
  sia::GradientDescent::Options opt = bo.optimizer().options();
  opt.tol = 1e-2;
  bo.optimizer().setOptions(opt);
  Eigen::VectorXd xopt = bo.getSolution();
  ASSERT_EQ(xopt.size(), 1);

  std::size_t NUM_STEPS = 50;
  for (std::size_t i = 0; i < NUM_STEPS; ++i) {
    Eigen::VectorXd x = bo.selectNextSample();
    double y = -quadratic(x);
    bo.addDataPoint(x, y);
    bool train = false;
    if (i >= NUM_STEPS - 10) {
      train = true;
    }
    bo.updateModel(train);
  }
  xopt = bo.getSolution();
  ASSERT_EQ(xopt.size(), 1);
  EXPECT_NEAR(xopt(0), 0, 2e-2);

  Eigen::VectorXd xopt2 = bo.getSolution();
  EXPECT_TRUE(xopt2.isApprox(xopt));

  // Check acqusition functions
  Eigen::VectorXd xtest = xopt;
  double acq = bo.acquisition(xtest);
  xtest(0) = xopt(0) + 1e-2;
  double acq_pos = bo.acquisition(xtest);
  xtest(0) = xopt(0) - 1e-2;
  double acq_neg = bo.acquisition(xtest);
  EXPECT_LT(acq_pos, acq);
  EXPECT_LT(acq_neg, acq);

  EXPECT_NE(
      bo.acquisition(
          xtest, 0,
          sia::BayesianOptimizer::AcquisitionType::PROBABILITY_IMPROVEMENT),
      0);
  EXPECT_NE(bo.acquisition(
                xtest, 0,
                sia::BayesianOptimizer::AcquisitionType::EXPECTED_IMPROVEMENT),
            0);
  EXPECT_NE(
      bo.acquisition(
          xtest, 0,
          sia::BayesianOptimizer::AcquisitionType::UPPER_CONFIDENCE_BOUND),
      0);

  // Test for 1D of branin free and 1D given (pi)
  Eigen::VectorXd u(1);
  u << M_PI;
  lb = -10 * Eigen::VectorXd::Ones(1);
  ub = 10 * Eigen::VectorXd::Ones(1);
  noise_kernel = sia::NoiseKernel(1e-2);
  se_kernel = sia::SEKernel(1.0, 1.0);
  sia::CompositeKernel kernel2 = noise_kernel + se_kernel;
  sia::BayesianOptimizer bo2(lb, ub, kernel2, 1);
  NUM_STEPS = 50;
  for (std::size_t i = 0; i < NUM_STEPS; ++i) {
    Eigen::VectorXd x = bo2.selectNextSample(u);
    Eigen::VectorXd ux = Eigen::VectorXd(2);
    ux.head<1>() = u;
    ux.tail<1>() = x;
    double y = -branin(ux);
    bo2.addDataPoint(x, y, u);
    bool train = false;
    bo2.updateModel(train);
  }
  xopt = bo2.getSolution(u);
  ASSERT_EQ(xopt.size(), 1);
  EXPECT_NEAR(xopt(0), 2.275, 2e-2);
}
