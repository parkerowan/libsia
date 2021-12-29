/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/gpr.h"
#include "sia/common/exception.h"
#include "sia/math/math.h"
#include "sia/optimizers/gradient_descent.h"

#include <glog/logging.h>
#include <limits>

namespace sia {

const double SMALL_NUMBER = 1e-6;
const double LARGE_NUMBER = 1e16;

std::size_t GPR::CovarianceFunction::numHyperparameters() const {
  return hyperparameters().size();
}

Eigen::VectorXd GPR::CovarianceFunction::evalVector(const Eigen::MatrixXd& a,
                                                    const Eigen::VectorXd& b) {
  std::size_t na = a.cols();
  Eigen::VectorXd k(na);
  for (std::size_t i = 0; i < na; ++i) {
    k(i) = eval(a.col(i), b);
  }
  return k;
}

Eigen::MatrixXd GPR::CovarianceFunction::evalMatrix(const Eigen::MatrixXd& a,
                                                    const Eigen::MatrixXd& b) {
  std::size_t na = a.cols();
  std::size_t nb = b.cols();
  Eigen::MatrixXd K(na, nb);
  for (std::size_t i = 0; i < nb; ++i) {
    K.col(i) = evalVector(a, b.col(i));
  }
  return K;
}

std::vector<Eigen::MatrixXd> GPR::CovarianceFunction::gradTensor(
    const Eigen::MatrixXd& a,
    const Eigen::MatrixXd& b) {
  std::size_t na = a.cols();
  std::size_t nb = b.cols();
  std::size_t np = numHyperparameters();
  std::vector<Eigen::MatrixXd> dK(np, Eigen::MatrixXd(na, nb));
  for (std::size_t i = 0; i < nb; ++i) {
    for (std::size_t j = 0; j < na; ++j) {
      Eigen::VectorXd g = grad(a.col(j), b.col(i));
      for (std::size_t k = 0; k < np; ++k) {
        dK[k](j, i) = g(k);
      }
    }
  }
  return dK;
}

GPR::CovarianceFunction* GPR::CovarianceFunction::create(
    GPR::KernelType kernel_type) {
  switch (kernel_type) {
    case GPR::SQUARED_EXPONENTIAL:
      return new SquaredExponential();
    default:
      SIA_EXCEPTION(
          false, "GPR::CovarianceFunction::Type encountered unsupported type");
  }
}

SquaredExponential::SquaredExponential(double length, double signal_var) {
  setHyperparameters(Eigen::Vector2d{length, signal_var});
}

double SquaredExponential::eval(const Eigen::VectorXd& a,
                                const Eigen::VectorXd& b) const {
  const Eigen::VectorXd e = a - b;
  return m_signal_var * exp(-e.dot(e) / pow(m_length, 2) / 2);
}

Eigen::VectorXd SquaredExponential::grad(const Eigen::VectorXd& a,
                                         const Eigen::VectorXd& b) const {
  const Eigen::VectorXd e = (a - b) / m_length;
  double mahal2 = e.dot(e);
  double knorm = exp(-mahal2 / 2);
  double dkdl = m_signal_var * knorm * mahal2 / m_length;
  double dkds = knorm;
  Eigen::VectorXd g(2);
  g << dkdl, dkds;
  return g;
}

Eigen::VectorXd SquaredExponential::hyperparameters() const {
  Eigen::VectorXd p(2);
  p << m_length, m_signal_var;
  return p;
}

void SquaredExponential::setHyperparameters(const Eigen::VectorXd& p) {
  SIA_EXCEPTION(numHyperparameters() == 2,
                "SquaredExponential hyperparameter dim expexted to be 2");
  m_length = p(0);
  m_signal_var = p(1);
  SIA_EXCEPTION(m_length > 0, "SquaredExponential expects length scale > 0");
  SIA_EXCEPTION(m_signal_var > 0, "SquaredExponential expects signal var > 0");
}

GPR::GPR(const Eigen::MatrixXd& input_samples,
         const Eigen::MatrixXd& output_samples,
         GPR::KernelType kernel_type,
         GPR::NoiseType noise_type)
    : m_input_samples(input_samples),
      m_output_samples(output_samples),
      m_belief(output_samples.rows()),
      m_noise_type(noise_type) {
  m_kernel = CovarianceFunction::create(kernel_type);
  assert(m_kernel != nullptr);
  cacheRegressionModels();
}

GPR::~GPR() {
  delete m_kernel;
}

const Gaussian& GPR::predict(const Eigen::VectorXd& x) {
  // Algorithm 2.1 in: http://www.gaussianprocess.org/gpml/chapters/RW.pdf
  // For each output channel
  std::size_t m = outputDimension();
  assert(m == m_models.size());

  Eigen::VectorXd mean(m), var(m);
  const Eigen::VectorXd kstar = m_kernel->evalVector(m_input_samples, x);
  double kxx = m_kernel->eval(x, x);
  for (std::size_t i = 0; i < m; ++i) {
    // Predictive mean, eqn. 2.25
    mean(i) = kstar.dot(m_models[i].m_cached_alpha);

    // Predictive variance, eqn. 2.26
    const Eigen::VectorXd v = m_models[i].m_cached_L_inv * kstar;
    var(i) = kxx - v.dot(v);
  }

  m_belief = Gaussian(mean, var.asDiagonal());
  return m_belief;
}

std::size_t GPR::inputDimension() const {
  return m_input_samples.rows();
}

std::size_t GPR::outputDimension() const {
  return m_output_samples.rows();
}

std::size_t GPR::numSamples() const {
  return m_input_samples.cols();
}

void GPR::train() {
  std::size_t n = numHyperparameters();
  GradientDescent optm(SMALL_NUMBER * Eigen::VectorXd::Ones(n),
                       LARGE_NUMBER * Eigen::VectorXd::Ones(n));
  auto loss = [this](const Eigen::VectorXd& x) {
    this->setHyperparameters(x);
    return this->negLogMarginalLik();
  };
  Eigen::VectorXd p = optm.minimize(loss, this->hyperparameters());
  this->setHyperparameters(p);
}

Eigen::VectorXd GPR::hyperparameters() const {
  return m_kernel->hyperparameters();
}

void GPR::setHyperparameters(const Eigen::VectorXd& p) {
  m_kernel->setHyperparameters(p);
  cacheRegressionModels();
}

std::size_t GPR::numHyperparameters() const {
  return m_kernel->numHyperparameters();
}

double GPR::negLogMarginalLik() {
  double neg_log_lik = 0;
  for (const auto& model : m_models) {
    neg_log_lik -= model.logMarginalLik();
  }
  return neg_log_lik;
}

Eigen::VectorXd GPR::negLogMarginalLikGrad() {
  // TODO: implement
  return Eigen::VectorXd::Zero(m_kernel->numHyperparameters());
}

void GPR::cacheRegressionModels() {
  // Check dimensions
  SIA_EXCEPTION(m_input_samples.cols() == m_output_samples.cols(),
                "Inconsistent number of input cols to output cols");

  // Cache models
  std::size_t m = m_output_samples.rows();
  m_models.clear();
  m_models.reserve(m);
  for (std::size_t i = 0; i < m; ++i) {
    const Eigen::VectorXd& Y = m_output_samples.row(i);
    m_models.emplace_back(GPR::RegressionModel(m_kernel, m_input_samples, Y));
  }
}

GPR::RegressionModel::RegressionModel(CovarianceFunction* kernel,
                                      const Eigen::MatrixXd& X,
                                      const Eigen::VectorXd& y)
    : m_X(X), m_y(y), m_kernel(kernel) {
  // Algorithm 2.1 in http://www.gaussianprocess.org/gpml/chapters/RW.pdf
  assert(m_kernel != nullptr);
  std::size_t n = m_X.cols();
  const Eigen::MatrixXd K = m_kernel->evalMatrix(m_X, m_X);

  // TODO: Figure this out
  double noise_var = 0.1;
  const Eigen::MatrixXd sI = noise_var * Eigen::MatrixXd::Identity(n, n);
  const Eigen::MatrixXd Ksig = K + sI;

  bool r = llt(Ksig, m_cached_L);
  SIA_EXCEPTION(r, "Failed to compute cholesky decomposition of sample matrix");

  m_cached_L_inv = m_cached_L.triangularView<Eigen::Lower>().solve(
      Eigen::MatrixXd::Identity(n, n));
  m_cached_K_inv = m_cached_L_inv.transpose() * m_cached_L_inv;
  m_cached_alpha = m_cached_K_inv * m_y;
}

double GPR::RegressionModel::logMarginalLik() const {
  // Eqn 2.30 in http://www.gaussianprocess.org/gpml/chapters/RW.pdf
  // Note the determinant of a triangular matrix is the product of its diagonal
  // And the log(|K|) = log(det(L))^2 = 2 * sum(log(diag(L)))  // More stable
  double n = m_y.size();
  const double log_kdet = 2 * m_cached_L.diagonal().array().log().sum();
  return -(m_y.transpose() * m_cached_alpha + log_kdet + n * log(2 * M_PI)) / 2;
}

Eigen::VectorXd GPR::RegressionModel::logMarginalLikGrad() const {
  // Eqn 5.9 in http://www.gaussianprocess.org/gpml/chapters/RW.pdf
  const Eigen::MatrixXd aaKinv =
      m_cached_alpha * m_cached_alpha.transpose() - m_cached_K_inv;
  std::size_t np = m_kernel->numHyperparameters();
  Eigen::VectorXd g = Eigen::VectorXd::Zero(np);
  std::vector<Eigen::MatrixXd> dKdp = m_kernel->gradTensor(m_X, m_X);
  for (std::size_t i = 0; i < np; ++i) {
    g(i) = (aaKinv * dKdp.at(i)).trace() / 2.0;
  }
  return g;
}

}  // namespace sia
