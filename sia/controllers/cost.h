/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/controllers/controllers.h"

#include <Eigen/Dense>

namespace sia {

/// Defines a cost function for an optimal control problem comprised of
/// $J(x,u) = c_f(x_T) + \sum_{i=1}^{T-1} c(x,u,i)$
/// where $c_f$ is the terminal cost and $c$ is the running cost.
class CostFunction {
 public:
  CostFunction() = default;
  virtual ~CostFunction() = default;

  // Running cost $c(x, u, i)$
  virtual double c(const Eigen::VectorXd& x,
                   const Eigen::VectorXd& u,
                   std::size_t i) const = 0;

  // Terminal cost $c_f(x)$
  virtual double cf(const Eigen::VectorXd& x) const = 0;

  // Total cost.  Assume that length of states $x$ is $N$, length of controls
  // $u$ is $N-1$.
  double eval(const std::vector<Eigen::VectorXd>& x,
              const std::vector<Eigen::VectorXd>& u);
};

/// Defines a differentiable cost function, exposing gradients and Hessians of
/// the cost functions.
class DifferentiableCost : public CostFunction {
 public:
  DifferentiableCost() = default;
  virtual ~DifferentiableCost() = default;

  virtual Eigen::VectorXd cx(const Eigen::VectorXd& x,
                             const Eigen::VectorXd& u,
                             std::size_t i) const = 0;
  virtual Eigen::VectorXd cu(const Eigen::VectorXd& x,
                             const Eigen::VectorXd& u,
                             std::size_t i) const = 0;
  virtual Eigen::MatrixXd cxx(const Eigen::VectorXd& x,
                              const Eigen::VectorXd& u,
                              std::size_t i) const = 0;
  virtual Eigen::MatrixXd cux(const Eigen::VectorXd& x,
                              const Eigen::VectorXd& u,
                              std::size_t i) const = 0;
  virtual Eigen::MatrixXd cuu(const Eigen::VectorXd& x,
                              const Eigen::VectorXd& u,
                              std::size_t i) const = 0;
  virtual Eigen::VectorXd cfx(const Eigen::VectorXd& x) const = 0;
  virtual Eigen::MatrixXd cfxx(const Eigen::VectorXd& x) const = 0;
};

/// Implements a quadratic cost function
/// $J(x,u) = l_f(x) + \sum_{i=0}^{T-2} l(x, u)$
/// where:
/// - $l_f(x) = 0.5 (x_{T-1} - x_d)' Q_f (x_{T-1} - x_d)$
/// - $l(x,u) = 0.5 (x-x_d)' Q (x-x_d) + u' R u)$
/// - $Qf$ is the final state cost
/// - $Q$ is the running state cost
/// - $R$ is the running input cost
/// - $x_d$ is the desired state
/// - $T$ is the horizon
class QuadraticCost : public DifferentiableCost {
 public:
  /// Constructor for $x_d = 0$
  explicit QuadraticCost(const Eigen::MatrixXd& Qf,
                         const Eigen::MatrixXd& Q,
                         const Eigen::MatrixXd& R);

  /// Constructor for fixed $x_d$
  explicit QuadraticCost(const Eigen::MatrixXd& Qf,
                         const Eigen::MatrixXd& Q,
                         const Eigen::MatrixXd& R,
                         const Eigen::VectorXd& xd);

  /// Constructor for trajectory of $x_d$
  explicit QuadraticCost(const Eigen::MatrixXd& Qf,
                         const Eigen::MatrixXd& Q,
                         const Eigen::MatrixXd& R,
                         const std::vector<Eigen::VectorXd>& xd);

  virtual ~QuadraticCost() = default;

  // Set the desired trajectory
  void setTrajectory(const std::vector<Eigen::VectorXd>& xd);

  // Running cost $c(x, u, i)$
  double c(const Eigen::VectorXd& x,
           const Eigen::VectorXd& u,
           std::size_t i) const override;

  // Terminal cost $c_f(x)$
  double cf(const Eigen::VectorXd& x) const override;

  // Terms
  const Eigen::VectorXd& xd(std::size_t i) const;
  const Eigen::MatrixXd& Qf() const;
  const Eigen::MatrixXd& Q() const;
  const Eigen::MatrixXd& R() const;

  // Cost partial gradients
  Eigen::VectorXd cx(const Eigen::VectorXd& x,
                     const Eigen::VectorXd& u,
                     std::size_t i) const override;
  Eigen::VectorXd cu(const Eigen::VectorXd& x,
                     const Eigen::VectorXd& u,
                     std::size_t i) const override;
  Eigen::MatrixXd cxx(const Eigen::VectorXd& x,
                      const Eigen::VectorXd& u,
                      std::size_t i) const override;
  Eigen::MatrixXd cux(const Eigen::VectorXd& x,
                      const Eigen::VectorXd& u,
                      std::size_t i) const override;
  Eigen::MatrixXd cuu(const Eigen::VectorXd& x,
                      const Eigen::VectorXd& u,
                      std::size_t i) const override;
  Eigen::VectorXd cfx(const Eigen::VectorXd& x) const override;
  Eigen::MatrixXd cfxx(const Eigen::VectorXd& x) const override;

 private:
  Eigen::MatrixXd m_final_state_cost;
  Eigen::MatrixXd m_state_cost;
  Eigen::MatrixXd m_input_cost;
  std::vector<Eigen::VectorXd> m_desired_states;
};

/// The terminal cost defines the final cost $l_f(x)$.
using TerminalCostFunction = std::function<double(const Eigen::VectorXd&)>;

/// The running cost defines the running cost $l(x, u)$.
using RunningCostFunction =
    std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)>;

/// Implements an arbitrary differentiable cost function using numerical
/// gradients
/// $J(x,u) = c_f(x_T) + \sum_{i=0}^{T-1} c(x, u)$ where:
/// - $c$ is the differentiable running cost
/// - $c_f$ is the differentiable terminal cost
/// The user must ensure that the provided cost functions can be differentiated.
/// Gradients are implemented with central difference.
class FunctionalCost : public DifferentiableCost {
 public:
  explicit FunctionalCost(TerminalCostFunction terminal_cost,
                          RunningCostFunction running_cost);
  virtual ~FunctionalCost() = default;

  // Running cost $c(x, u, i)$
  double c(const Eigen::VectorXd& x,
           const Eigen::VectorXd& u,
           std::size_t i) const override;

  // Terminal cost $c_f(x)$
  double cf(const Eigen::VectorXd& x) const override;

  // Cost partial gradients
  Eigen::VectorXd cx(const Eigen::VectorXd& x,
                     const Eigen::VectorXd& u,
                     std::size_t i) const override;
  Eigen::VectorXd cu(const Eigen::VectorXd& x,
                     const Eigen::VectorXd& u,
                     std::size_t i) const override;
  Eigen::MatrixXd cxx(const Eigen::VectorXd& x,
                      const Eigen::VectorXd& u,
                      std::size_t i) const override;
  Eigen::MatrixXd cux(const Eigen::VectorXd& x,
                      const Eigen::VectorXd& u,
                      std::size_t i) const override;
  Eigen::MatrixXd cuu(const Eigen::VectorXd& x,
                      const Eigen::VectorXd& u,
                      std::size_t i) const override;
  Eigen::VectorXd cfx(const Eigen::VectorXd& x) const override;
  Eigen::MatrixXd cfxx(const Eigen::VectorXd& x) const override;

 private:
  TerminalCostFunction m_terminal_cost;
  RunningCostFunction m_running_cost;
};

}  // namespace sia
