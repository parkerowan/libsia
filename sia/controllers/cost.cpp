/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/controllers/cost.h"
#include "sia/common/exception.h"
#include "sia/common/logger.h"
#include "sia/math/math.h"

namespace sia {

double CostFunction::eval(const std::vector<Eigen::VectorXd>& x,
                          const std::vector<Eigen::VectorXd>& u) const {
  std::size_t N = x.size();
  double J = cf(x.at(N - 1));
  for (std::size_t i = 0; i < N - 1; ++i) {
    J += c(x.at(i), u.at(i), i);
  }
  return J;
}

QuadraticCost::QuadraticCost(const Eigen::MatrixXd& Qf,
                             const Eigen::MatrixXd& Q,
                             const Eigen::MatrixXd& R)
    : m_final_state_cost(Qf), m_state_cost(Q), m_input_cost(R) {
  m_desired_states.emplace_back(Eigen::VectorXd::Zero(Q.rows()));
}

QuadraticCost::QuadraticCost(const Eigen::MatrixXd& Qf,
                             const Eigen::MatrixXd& Q,
                             const Eigen::MatrixXd& R,
                             const Eigen::VectorXd& xd)
    : m_final_state_cost(Qf), m_state_cost(Q), m_input_cost(R) {
  m_desired_states.emplace_back(xd);
}

QuadraticCost::QuadraticCost(const Eigen::MatrixXd& Qf,
                             const Eigen::MatrixXd& Q,
                             const Eigen::MatrixXd& R,
                             const std::vector<Eigen::VectorXd>& xd)
    : m_final_state_cost(Qf),
      m_state_cost(Q),
      m_input_cost(R),
      m_desired_states(xd) {}

void QuadraticCost::setTrajectory(const std::vector<Eigen::VectorXd>& xd) {
  m_desired_states = xd;
}

double QuadraticCost::c(const Eigen::VectorXd& x,
                        const Eigen::VectorXd& u,
                        std::size_t i) const {
  const auto& x_d = xd(i);
  const auto& Q = m_state_cost;
  const auto& R = m_input_cost;
  const auto e = x - x_d;
  return 0.5 * (e.dot(Q * e) + u.dot(R * u));
}

double QuadraticCost::cf(const Eigen::VectorXd& x) const {
  const auto& x_d = m_desired_states.back();
  const auto& Qf = m_final_state_cost;
  const auto e = x - x_d;
  return 0.5 * e.dot(Qf * e);
}

const Eigen::VectorXd& QuadraticCost::xd(std::size_t i) const {
  std::size_t N = m_desired_states.size();
  if (N == 1) {
    return m_desired_states.at(0);
  } else {
    SIA_EXCEPTION(i < N, "Quadratic cost expects i < N");
    return m_desired_states.at(i);
  }
}

const Eigen::MatrixXd& QuadraticCost::Qf() const {
  return m_final_state_cost;
}

const Eigen::MatrixXd& QuadraticCost::Q() const {
  return m_state_cost;
}

const Eigen::MatrixXd& QuadraticCost::R() const {
  return m_input_cost;
}

Eigen::VectorXd QuadraticCost::cx(const Eigen::VectorXd& x,
                                  const Eigen::VectorXd& u,
                                  std::size_t i) const {
  (void)(u);
  const auto& x_d = xd(i);
  return m_state_cost * (x - x_d);
}

Eigen::VectorXd QuadraticCost::cu(const Eigen::VectorXd& x,
                                  const Eigen::VectorXd& u,
                                  std::size_t i) const {
  (void)(x);
  (void)(i);
  return m_input_cost * u;
}

Eigen::MatrixXd QuadraticCost::cxx(const Eigen::VectorXd& x,
                                   const Eigen::VectorXd& u,
                                   std::size_t i) const {
  (void)(x);
  (void)(u);
  (void)(i);
  return m_state_cost;
}

Eigen::MatrixXd QuadraticCost::cux(const Eigen::VectorXd& x,
                                   const Eigen::VectorXd& u,
                                   std::size_t i) const {
  (void)(x);
  (void)(u);
  (void)(i);
  return Eigen::MatrixXd::Zero(m_input_cost.rows(), m_state_cost.rows());
}

Eigen::MatrixXd QuadraticCost::cuu(const Eigen::VectorXd& x,
                                   const Eigen::VectorXd& u,
                                   std::size_t i) const {
  (void)(x);
  (void)(u);
  (void)(i);
  return m_input_cost;
}

Eigen::VectorXd QuadraticCost::cfx(const Eigen::VectorXd& x) const {
  const auto& x_d = m_desired_states.back();
  return m_final_state_cost * (x - x_d);
}

Eigen::MatrixXd QuadraticCost::cfxx(const Eigen::VectorXd& x) const {
  (void)(x);
  return m_final_state_cost;
}

FunctionalCost::FunctionalCost(TerminalCostFunction terminal_cost,
                               RunningCostFunction running_cost)
    : m_terminal_cost(terminal_cost), m_running_cost(running_cost) {}

double FunctionalCost::c(const Eigen::VectorXd& x,
                         const Eigen::VectorXd& u,
                         std::size_t i) const {
  (void)(i);
  return m_running_cost(x, u);
}

double FunctionalCost::cf(const Eigen::VectorXd& x) const {
  return m_terminal_cost(x);
}

Eigen::VectorXd FunctionalCost::cx(const Eigen::VectorXd& x,
                                   const Eigen::VectorXd& u,
                                   std::size_t i) const {
  (void)(i);
  return dfdx(m_running_cost, x, u);
}

Eigen::VectorXd FunctionalCost::cu(const Eigen::VectorXd& x,
                                   const Eigen::VectorXd& u,
                                   std::size_t i) const {
  (void)(i);
  return dfdu(m_running_cost, x, u);
}

Eigen::MatrixXd FunctionalCost::cxx(const Eigen::VectorXd& x,
                                    const Eigen::VectorXd& u,
                                    std::size_t i) const {
  (void)(i);
  return d2fdxx(m_running_cost, x, u);
}

Eigen::MatrixXd FunctionalCost::cux(const Eigen::VectorXd& x,
                                    const Eigen::VectorXd& u,
                                    std::size_t i) const {
  (void)(i);
  return d2fdux(m_running_cost, x, u);
}

Eigen::MatrixXd FunctionalCost::cuu(const Eigen::VectorXd& x,
                                    const Eigen::VectorXd& u,
                                    std::size_t i) const {
  (void)(i);
  return d2fduu(m_running_cost, x, u);
}

Eigen::VectorXd FunctionalCost::cfx(const Eigen::VectorXd& x) const {
  return dfdx(m_terminal_cost, x);
}

Eigen::MatrixXd FunctionalCost::cfxx(const Eigen::VectorXd& x) const {
  return d2fdxx(m_terminal_cost, x);
}

}  // namespace sia
