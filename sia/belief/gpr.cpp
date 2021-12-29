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

// Evalutes the kernel to construct the na x 1 kernel vector K(a, b) where a,
// b are input samples with cols equal to samples.
static Eigen::VectorXd evalVector(const GPR::KernelFunction& kernel,
                                  const Eigen::MatrixXd& a,
                                  const Eigen::VectorXd& b) {
  std::size_t na = a.cols();
  Eigen::VectorXd k(na);
  for (std::size_t i = 0; i < na; ++i) {
    k(i) = kernel.eval(a.col(i), b);
  }
  return k;
}

// Evaluates the kernel to construct the na x nb kernel matrix K(a, b) where
// a, b are input samples with cols equal to samples.
static Eigen::MatrixXd evalMatrix(const GPR::KernelFunction& kernel,
                                  const Eigen::MatrixXd& a,
                                  const Eigen::MatrixXd& b) {
  std::size_t na = a.cols();
  std::size_t nb = b.cols();
  Eigen::MatrixXd K(na, nb);
  for (std::size_t i = 0; i < nb; ++i) {
    K.col(i) = evalVector(kernel, a, b.col(i));
  }
  return K;
}

// Evaluates the kernel gradient w.r.t. hyperparameters to construct the
// tensor na x nb kernel matrix dK(a, b)/dp where the number of elements of
// the output vector correspond to the number of hyperparameters.
static std::vector<Eigen::MatrixXd> gradTensor(
    const GPR::KernelFunction& kernel,
    const Eigen::MatrixXd& a,
    const Eigen::MatrixXd& b) {
  std::size_t na = a.cols();
  std::size_t nb = b.cols();
  std::size_t np = kernel.numHyperparameters();
  std::vector<Eigen::MatrixXd> dK(np, Eigen::MatrixXd(na, nb));
  for (std::size_t i = 0; i < nb; ++i) {
    for (std::size_t j = 0; j < na; ++j) {
      Eigen::VectorXd g = kernel.grad(a.col(j), b.col(i));
      for (std::size_t k = 0; k < np; ++k) {
        dK[k](j, i) = g(k);
      }
    }
  }
  return dK;
}

// Kernel function factory
static GPR::KernelFunction* createKernel(GPR::KernelType kernel_type) {
  switch (kernel_type) {
    case GPR::SQUARED_EXPONENTIAL:
      return new SquaredExponential();
    default:
      SIA_EXCEPTION(false, "Encountered unsupported GPR::KernelType");
  }
}

GPR::GPR(const Eigen::MatrixXd& input_samples,
         const Eigen::MatrixXd& output_samples,
         GPR::KernelType kernel_type,
         GPR::NoiseType noise_type)
    : m_input_samples(input_samples),
      m_output_samples(output_samples),
      m_belief(output_samples.rows()),
      m_noise_type(noise_type) {
  m_kernel = createKernel(kernel_type);
  assert(m_kernel != nullptr);
  cacheRegressionModel();
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
  const Eigen::VectorXd kstar = evalVector(*m_kernel, m_input_samples, x);
  double kxx = m_kernel->eval(x, x);
  for (std::size_t i = 0; i < m; ++i) {
    // Predictive mean, eqn. 2.25
    mean(i) = kstar.dot(m_models[i].cached_alpha);

    // Predictive variance, eqn. 2.26
    const Eigen::VectorXd v = m_models[i].cached_L_inv * kstar;
    var(i) = kxx - v.dot(v);
  }

  m_belief = Gaussian(mean, var.asDiagonal());
  return m_belief;
}

double GPR::negLogMarginalLik() {
  // Eqn 2.30 in http://www.gaussianprocess.org/gpml/chapters/RW.pdf
  // Note the determinant of a triangular matrix is the product of its diagonal
  // And the log(|K|) = log(det(L))^2 = 2 * sum(log(diag(L)))  // More stable
  double n = numSamples();
  double neg_log_lik = 0;
  for (std::size_t i = 0; i < outputDimension(); ++i) {
    const auto& model = m_models.at(i);
    const Eigen::VectorXd& Y = m_output_samples.row(i);
    double log_kdet = 2 * model.cached_L.diagonal().array().log().sum();
    neg_log_lik +=
        (Y.transpose() * model.cached_alpha + log_kdet + n * log(2 * M_PI)) / 2;
  }
  return neg_log_lik;
}

Eigen::VectorXd GPR::negLogMarginalLikGrad() {
  // Eqn 5.9 in http://www.gaussianprocess.org/gpml/chapters/RW.pdf
  std::size_t np = m_kernel->numHyperparameters();
  Eigen::VectorXd g = Eigen::VectorXd::Zero(m_kernel->numHyperparameters());
  for (const auto& model : m_models) {
    const Eigen::MatrixXd aaKinv =
        model.cached_alpha * model.cached_alpha.transpose() -
        model.cached_K_inv;
    Eigen::VectorXd gm = Eigen::VectorXd::Zero(np);
    for (std::size_t i = 0; i < np; ++i) {
      gm(i) = (aaKinv * model.cached_grad.at(i)).trace() / 2.0;
    }
    g -= gm;
  }
  return g;
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

std::size_t GPR::inputDimension() const {
  return m_input_samples.rows();
}

std::size_t GPR::outputDimension() const {
  return m_output_samples.rows();
}

std::size_t GPR::numSamples() const {
  return m_input_samples.cols();
}

Eigen::VectorXd GPR::hyperparameters() const {
  return m_kernel->hyperparameters();
}

void GPR::setHyperparameters(const Eigen::VectorXd& p) {
  m_kernel->setHyperparameters(p);
  cacheRegressionModel();
}

std::size_t GPR::numHyperparameters() const {
  return m_kernel->numHyperparameters();
}

void GPR::cacheRegressionModel() {
  std::size_t n = numSamples();
  std::size_t m = outputDimension();
  const Eigen::MatrixXd& X = m_input_samples;
  m_models.clear();
  m_models.reserve(m);
  for (std::size_t i = 0; i < m; ++i) {
    // Algorithm 2.1 in http://www.gaussianprocess.org/gpml/chapters/RW.pdf
    const Eigen::VectorXd& Y = m_output_samples.row(i);
    const Eigen::MatrixXd K = evalMatrix(*m_kernel, X, X);

    // Compute noise matrix
    // TODO: based on noise model
    double noise_var = 0.1;
    const Eigen::MatrixXd sI = noise_var * Eigen::MatrixXd::Identity(n, n);

    // Cholesky decomposition of K
    const Eigen::MatrixXd Ksig = K + sI;
    Eigen::MatrixXd L;
    bool r = llt(Ksig, L);
    SIA_EXCEPTION(r,
                  "Failed to compute cholesky decomposition of sample matrix");

    // Compute inverse of K
    const Eigen::MatrixXd L_inv =
        L.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(n, n));
    const Eigen::MatrixXd K_inv = L_inv.transpose() * L_inv;

    // Compute projection to output space
    const Eigen::MatrixXd alpha = K_inv * Y;

    // Pre-compute the gradient tensor
    std::vector<Eigen::MatrixXd> grad = gradTensor(*m_kernel, X, X);
    m_models.emplace_back(RegressionModel(L, L_inv, K_inv, alpha, grad));
  }
}

GPR::RegressionModel::RegressionModel(const Eigen::MatrixXd& L,
                                      const Eigen::MatrixXd& L_inv,
                                      const Eigen::MatrixXd& K_inv,
                                      const Eigen::VectorXd& alpha,
                                      const std::vector<Eigen::MatrixXd>& grad)
    : cached_L(L),
      cached_L_inv(L_inv),
      cached_K_inv(K_inv),
      cached_alpha(alpha),
      cached_grad(grad) {}

std::size_t GPR::KernelFunction::numHyperparameters() const {
  return hyperparameters().size();
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

}  // namespace sia
