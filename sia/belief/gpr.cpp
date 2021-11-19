/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/gpr.h"
#include "sia/common/exception.h"
#include "sia/math/math.h"

#include <glog/logging.h>
#include <limits>

namespace sia {

/// Kernel functions
struct GPR::Kernel {
  Kernel() = default;
  virtual ~Kernel() = default;

  /// Evaluates the kernel k(a, b) where a, b are input samples.
  virtual double eval(const Eigen::VectorXd& a,
                      const Eigen::VectorXd& b,
                      double varf,
                      double length) const = 0;

  /// Evalutes the kernel to construct the na x 1 kernel vector K(a, b) where a,
  /// b are input samples with cols equal to samples.
  Eigen::VectorXd evalVector(const Eigen::MatrixXd& a,
                             const Eigen::VectorXd& b,
                             double varf,
                             double length) {
    std::size_t n = a.cols();
    Eigen::VectorXd K = Eigen::VectorXd::Zero(n);
    for (std::size_t i = 0; i < n; ++i) {
      K(i) = eval(a.col(i), b, varf, length);
    }
    return K;
  }

  /// Evaluates the kernel to construct the na x nb kernel matrix K(a, b) where
  /// a, b are input samples with cols equal to samples.
  Eigen::MatrixXd evalMatrix(const Eigen::MatrixXd& a,
                             const Eigen::MatrixXd& b,
                             double varf,
                             double length) {
    std::size_t na = a.cols();
    std::size_t nb = b.cols();
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(na, nb);
    for (std::size_t i = 0; i < nb; ++i) {
      K.col(i) = evalVector(a, b.col(i), varf, length);
    }
    return K;
  }

  // Forward declaration
  struct SquaredExponential;
};

// Squared exponential kernel
struct GPR::Kernel::SquaredExponential : public GPR::Kernel {
  double eval(const Eigen::VectorXd& a,
              const Eigen::VectorXd& b,
              double varf,
              double length) const override {
    const Eigen::VectorXd e = (a - b) / length;
    return varf * exp(-e.dot(e) / 2);
  }
};

GPR::GPR(const Eigen::MatrixXd& input_samples,
         const Eigen::MatrixXd& output_samples,
         double varn,
         double varf,
         double length,
         GPR::CovFunction type)
    : GPR(input_samples,
          output_samples,
          varn * Eigen::MatrixXd::Ones(output_samples.rows(),
                                       output_samples.cols()),
          varf,
          length,
          type) {
  m_heteroskedastic = false;
}

GPR::GPR(const Eigen::MatrixXd& input_samples,
         const Eigen::MatrixXd& output_samples,
         const Eigen::MatrixXd& varn,
         double varf,
         double length,
         GPR::CovFunction type)
    : m_belief(output_samples.rows()),
      m_input_samples(input_samples),
      m_output_samples(output_samples),
      m_varn(varn),
      m_varf(varf),
      m_length(length) {
  if (type == GPR::SQUARED_EXPONENTIAL) {
    m_kernel = new GPR::Kernel::SquaredExponential();
  } else {
    LOG(ERROR) << "Unsupported covariance function " << type;
  }
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
  Eigen::VectorXd mean(m), var(m);
  for (std::size_t i = 0; i < m; ++i) {
    const Eigen::VectorXd kstar =
        m_kernel->evalVector(m_input_samples, x, m_varf, m_length);

    // Predictive mean, eqn. 2.25
    mean(i) = kstar.dot(m_models[i].m_cached_alpha);

    // Predictive variance, eqn. 2.26
    const Eigen::VectorXd v = m_models[i].m_cached_L_inv * kstar;
    var(i) = m_kernel->eval(x, x, m_varf, m_length) - v.dot(v);
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

double GPR::negLogLikLoss() {
  double neg_log_lik = 0;
  for (std::size_t i = 0; i < numSamples(); ++i) {
    const auto& x = m_input_samples.col(i);
    const auto& y = m_output_samples.col(i);
    neg_log_lik -= predict(x).logProb(y);
  }
  return neg_log_lik;
}

Eigen::VectorXd GPR::getHyperparameters() const {
  if (m_heteroskedastic) {
    LOG(WARNING) << "This GPR uses a heteroskedatic noise model, only "
                    "returning the first element of the noise matrix";
  }
  return Eigen::Vector3d(m_varn(0, 0), m_varf, m_length);
}

void GPR::setHyperparameters(const Eigen::VectorXd& p) {
  SIA_EXCEPTION(p.size() == 3, "Hyperparameter vector dim expected to be 3");

  m_varn = p(0) * Eigen::MatrixXd::Ones(m_output_samples.rows(),
                                        m_output_samples.cols());
  m_varf = p(1);
  m_length = p(2);

  cacheRegressionModels();
}

std::size_t GPR::numSamples() const {
  return m_input_samples.cols();
}

void GPR::cacheRegressionModels() {
  // Check dimensions
  SIA_EXCEPTION(m_input_samples.cols() == m_output_samples.cols(),
                "Inconsistent number of input cols to output cols");
  SIA_EXCEPTION(m_input_samples.cols() == m_varn.cols(),
                "Inconsistent number of input cols to varn cols");
  SIA_EXCEPTION(m_output_samples.rows() == m_varn.rows(),
                "Inconsistent number of output rows to varn rows");

  // Check hyperparameters
  SIA_EXCEPTION(m_varn(0, 0) > 0, "GPR expects hyperparameter varn to be > 0");
  SIA_EXCEPTION(m_varf > 0, "GPR expects hyperparameter varf to be > 0");
  SIA_EXCEPTION(m_length > 0, "GPR expects hyperparameter length to be > 0");

  // Cache models
  std::size_t m = m_output_samples.rows();
  m_models.clear();
  m_models.reserve(m);
  for (std::size_t i = 0; i < m; ++i) {
    const Eigen::VectorXd& Y = m_output_samples.row(i);
    const Eigen::VectorXd& varn = m_varn.row(i);
    m_models.emplace_back(GPR::RegressionModel(m_kernel, m_input_samples, Y,
                                               varn, m_varf, m_length));
  }
}

GPR::RegressionModel::RegressionModel(Kernel* kernel,
                                      const Eigen::MatrixXd& X,
                                      const Eigen::VectorXd& y,
                                      const Eigen::VectorXd& varn,
                                      double varf,
                                      double length) {
  // Algorithm 2.1 in: http://www.gaussianprocess.org/gpml/chapters/RW.pdf
  assert(kernel != nullptr);
  std::size_t n = X.cols();
  assert(std::size_t(varn.size()) == n);
  const Eigen::MatrixXd K = kernel->evalMatrix(X, X, varf, length);
  const Eigen::MatrixXd sig = varn.asDiagonal();
  const Eigen::MatrixXd Ksig = K + sig;

  Eigen::MatrixXd L;
  bool r = llt(Ksig, L);
  SIA_EXCEPTION(r, "Failed to compute cholesky decomposition of sample matrix");

  m_cached_L_inv =
      L.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(n, n));

  m_cached_alpha = m_cached_L_inv.transpose() * m_cached_L_inv * y;
}

}  // namespace sia
