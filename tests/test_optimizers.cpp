/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include <gtest/gtest.h>
#include <sia/sia.h>

#include <iostream>

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
  sia::GradientDescent a(lb, ub, 1e-6, 0.5);
  Eigen::VectorXd xopt = a.minimize(quadratic, Eigen::Vector2d{1.2, -2.5});
  EXPECT_NEAR(xopt(0), 0, 1e-6);
  EXPECT_NEAR(xopt(1), 0, 1e-6);
  EXPECT_EQ(a.dimension(), 2);

  // bounds inactive at optimimum - multiple starts
  xopt = a.minimize(quadratic,
                    std::vector<Eigen::VectorXd>{Eigen::Vector2d{1.2, -2.5},
                                                 Eigen::Vector2d{-15, 13.0},
                                                 Eigen::Vector2d{7.2, -8.0}});
  EXPECT_NEAR(xopt(0), 0, 1e-6);
  EXPECT_NEAR(xopt(1), 0, 1e-6);

  // bounds active at optimum
  lb = Eigen::Vector2d{-5, -5};
  ub = Eigen::Vector2d{5, -2};
  sia::GradientDescent b(lb, ub, 1e-6, 0.5);
  xopt = b.minimize(quadratic, Eigen::Vector2d{1.2, -6});
  EXPECT_NEAR(xopt(0), 0, 1e-6);
  EXPECT_NEAR(xopt(1), -2, 1e-6);
}

TEST(Optimizers, BayesianOptimizer) {
  Eigen::VectorXd lb = -Eigen::VectorXd::Ones(1);
  Eigen::VectorXd ub = Eigen::VectorXd::Ones(1);
  sia::BayesianOptimizer bo(lb, ub);

  for (std::size_t i = 0; i < 20; ++i) {
    Eigen::VectorXd x = bo.selectNextSample();
    double y = -quadratic(x);
    bo.addDataPoint(x, y);
    bo.updateModel();
  }
  Eigen::VectorXd xopt = bo.getSolution();
  EXPECT_NEAR(xopt(0), 0, 1e-2);
}
