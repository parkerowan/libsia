/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/models/gmr_models.h"
#include "sia/common/exception.h"
#include "sia/math/math.h"

namespace sia {

GMRDynamics::GMRDynamics(const Eigen::MatrixXd& Xk,
                         const Eigen::MatrixXd& Uk,
                         const Eigen::MatrixXd& Xkp1,
                         std::size_t K)
    : m_prob_dynamics(Xkp1.rows()),
      m_input_indices(indices(0, Xk.rows() + Uk.rows())),
      m_output_indices(
          indices(Xk.rows() + Uk.rows(), Xk.rows() + Uk.rows() + Xkp1.rows())),
      m_gmr(createGMR(Xk, Uk, Xkp1, K, m_input_indices, m_output_indices)) {}

Gaussian& GMRDynamics::dynamics(const Eigen::VectorXd& state,
                                const Eigen::VectorXd& control) {
  std::size_t m = state.size();
  std::size_t n = control.size();
  Eigen::VectorXd xu = Eigen::VectorXd(m + n);
  xu.head(m) = state;
  xu.tail(n) = control;
  m_prob_dynamics = m_gmr.predict(xu);
  m_prob_dynamics.setMean(m_prob_dynamics.mean() + state);
  return m_prob_dynamics;
}

Eigen::VectorXd GMRDynamics::f(const Eigen::VectorXd& state,
                               const Eigen::VectorXd& control) {
  return dynamics(state, control).mean();
}

Eigen::MatrixXd GMRDynamics::Q(const Eigen::VectorXd& state,
                               const Eigen::VectorXd& control) {
  return dynamics(state, control).covariance();
}

double GMRDynamics::negLogLik(const Eigen::MatrixXd& Xk,
                              const Eigen::MatrixXd& Uk,
                              const Eigen::MatrixXd& Xkp1) {
  const Eigen::MatrixXd X = stack(Xk, Uk);
  const Eigen::MatrixXd Y = Xkp1 - Xk;
  return m_gmr.negLogLik(X, Y);
}

GMR& GMRDynamics::gmr() {
  return m_gmr;
}

GMR GMRDynamics::createGMR(
    const Eigen::MatrixXd& Xk,
    const Eigen::MatrixXd& Uk,
    const Eigen::MatrixXd& Xkp1,
    std::size_t K,
    const std::vector<std::size_t>& input_indices,
    const std::vector<std::size_t>& output_indices) const {
  SIA_EXCEPTION(Xk.cols() == Uk.cols(),
                "Training data Xk, Uk expected to have sample number of cols");
  SIA_EXCEPTION(
      Xk.cols() == Xkp1.cols(),
      "Training data Xk, Xkp1 expected to have sample number of cols");
  SIA_EXCEPTION(
      Xk.rows() == Xkp1.rows(),
      "Training data Xk, Xkp1 expected to have sample number of rows");

  const Eigen::MatrixXd X = stack(Xk, Uk);
  const Eigen::MatrixXd Y = Xkp1 - Xk;
  const Eigen::MatrixXd D = stack(X, Y);
  GMM gmm(D, K);
  return GMR(gmm, input_indices, output_indices);
}

GMRMeasurement::GMRMeasurement(const Eigen::MatrixXd& X,
                               const Eigen::MatrixXd& Y,
                               std::size_t K)
    : m_prob_measurement(Y.rows()),
      m_input_indices(indices(0, X.rows())),
      m_output_indices(indices(X.rows(), X.rows() + Y.rows())),
      m_gmr(createGMR(X, Y, K, m_input_indices, m_output_indices)) {}

Gaussian& GMRMeasurement::measurement(const Eigen::VectorXd& state) {
  m_prob_measurement = m_gmr.predict(state);
  return m_prob_measurement;
}

Eigen::VectorXd GMRMeasurement::h(const Eigen::VectorXd& state) {
  return measurement(state).mean();
}

Eigen::MatrixXd GMRMeasurement::R(const Eigen::VectorXd& state) {
  return measurement(state).covariance();
}

double GMRMeasurement::negLogLik(const Eigen::MatrixXd& X,
                                 const Eigen::MatrixXd& Y) {
  return m_gmr.negLogLik(X, Y);
}

GMR& GMRMeasurement::gmr() {
  return m_gmr;
}

GMR GMRMeasurement::createGMR(
    const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y,
    std::size_t K,
    const std::vector<std::size_t>& input_indices,
    const std::vector<std::size_t>& output_indices) const {
  SIA_EXCEPTION(X.cols() == Y.cols(),
                "Training data X, Y expected to have sample number of cols");
  const Eigen::MatrixXd D = stack(X, Y);
  GMM gmm(D, K);
  return GMR(gmm, input_indices, output_indices);
}

}  // namespace sia
