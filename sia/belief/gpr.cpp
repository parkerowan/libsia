/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/gpr.h"
#include "sia/common/exception.h"
#include "sia/common/logger.h"
#include "sia/math/math.h"

#include <limits>

namespace sia {

// Evalutes the kernel with training data to construct the na x 1 kernel vector
// K(x_train, x_test) where a, b are input samples with cols equal to samples.
static Eigen::VectorXd evalTestData(const Kernel& kernel,
                                    const Eigen::MatrixXd& x_train,
                                    const Eigen::VectorXd& x_test,
                                    std::size_t output_index) {
  std::size_t n = x_train.cols();
  Eigen::VectorXd k(n);
  for (std::size_t i = 0; i < n; ++i) {
    k(i) = kernel.eval(x_train.col(i), x_test, output_index);
  }
  return k;
}

// Evaluates the kernel to construct the n x n kernel matrix K(x, y) where
// x, y are input samples with cols equal to samples.
static Eigen::MatrixXd evalTrainData(const Kernel& kernel,
                                     const Eigen::MatrixXd& x_train,
                                     std::size_t output_index) {
  std::size_t n = x_train.cols();
  Eigen::MatrixXd K(n, n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      if (i == j) {
        K(i, j) = kernel.eval(x_train.col(i), output_index);
      } else {
        K(i, j) = kernel.eval(x_train.col(i), x_train.col(j), output_index);
      }
    }
  }
  return K;
}

// Evaluates the kernel gradient w.r.t. hyperparameters to construct the
// tensor n x n kernel matrix dK(x, y)/dp where the number of elements of
// the output vector correspond to the number of hyperparameters.
static std::vector<Eigen::MatrixXd> gradTrainData(
    const Kernel& kernel,
    const Eigen::MatrixXd& x_train,
    std::size_t output_index) {
  std::size_t nx = x_train.cols();
  std::size_t np = kernel.numHyperparameters();
  std::vector<Eigen::MatrixXd> dK(np, Eigen::MatrixXd(nx, nx));
  for (std::size_t i = 0; i < nx; ++i) {
    for (std::size_t j = 0; j < nx; ++j) {
      Eigen::VectorXd g =
          kernel.grad(x_train.col(j), x_train.col(i), output_index);
      for (std::size_t k = 0; k < np; ++k) {
        dK[k](j, i) = g(k);
      }
    }
  }
  return dK;
}

// ----------------------------------------------------------------------------

CompositeKernel CompositeKernel::multiply(Kernel& a, Kernel& b) {
  return CompositeKernel(a, b, Operation::MULTIPLY);
}

CompositeKernel CompositeKernel::add(Kernel& a, Kernel& b) {
  return CompositeKernel(a, b, Operation::ADD);
}

CompositeKernel::CompositeKernel(Kernel& a, Kernel& b, Operation operation)
    : m_kernel_a(a), m_kernel_b(b), m_operation(operation) {}

double CompositeKernel::eval(const Eigen::VectorXd& x,
                             std::size_t output_index) const {
  if (m_operation == Operation::MULTIPLY) {
    return m_kernel_a.eval(x, output_index) * m_kernel_b.eval(x, output_index);
  } else if (m_operation == Operation::ADD) {
    return m_kernel_a.eval(x, output_index) + m_kernel_b.eval(x, output_index);
  } else {
    SIA_THROW_IF_NOT(false, "Kernel Operation not implemented")
  }
}

double CompositeKernel::eval(const Eigen::VectorXd& x,
                             const Eigen::VectorXd& y,
                             std::size_t output_index) const {
  if (m_operation == Operation::MULTIPLY) {
    return m_kernel_a.eval(x, y, output_index) *
           m_kernel_b.eval(x, y, output_index);
  } else if (m_operation == Operation::ADD) {
    return m_kernel_a.eval(x, y, output_index) +
           m_kernel_b.eval(x, y, output_index);
  } else {
    SIA_THROW_IF_NOT(false, "Kernel Operation not implemented")
  }
}

Eigen::VectorXd CompositeKernel::grad(const Eigen::VectorXd& x,
                                      const Eigen::VectorXd& y,
                                      std::size_t output_index) const {
  std::size_t na = m_kernel_a.numHyperparameters();
  std::size_t nb = m_kernel_b.numHyperparameters();
  Eigen::VectorXd p(na + nb);
  if (m_operation == Operation::MULTIPLY) {
    // Product rule (fg)' = f'g + fg'
    p.head(na) = m_kernel_a.grad(x, y, output_index) *
                 m_kernel_b.eval(x, y, output_index);
    p.tail(nb) = m_kernel_b.eval(x, y, output_index) *
                 m_kernel_b.grad(x, y, output_index);
  } else if (m_operation == Operation::ADD) {
    p.head(na) = m_kernel_a.grad(x, y, output_index);
    p.tail(nb) = m_kernel_b.grad(x, y, output_index);
  } else {
    SIA_THROW_IF_NOT(false, "Kernel Operation not implemented")
  }
  return p;
}

Eigen::VectorXd CompositeKernel::hyperparameters() const {
  std::size_t na = m_kernel_a.numHyperparameters();
  std::size_t nb = m_kernel_b.numHyperparameters();
  Eigen::VectorXd p(na + nb);
  p.head(na) = m_kernel_a.hyperparameters();
  p.tail(nb) = m_kernel_b.hyperparameters();
  return p;
}

void CompositeKernel::setHyperparameters(const Eigen::VectorXd& p) {
  int na = m_kernel_a.numHyperparameters();
  int nb = m_kernel_b.numHyperparameters();
  SIA_THROW_IF_NOT(p.size() == na + nb,
                   "CompositeKernel hyperparameter dim expexted to be na + nb");
  m_kernel_a.setHyperparameters(p.head(na));
  m_kernel_b.setHyperparameters(p.tail(nb));
}

std::size_t CompositeKernel::numHyperparameters() const {
  std::size_t na = m_kernel_a.numHyperparameters();
  std::size_t nb = m_kernel_b.numHyperparameters();
  return na + nb;
}

CompositeKernel operator*(Kernel& a, Kernel& b) {
  return CompositeKernel::multiply(a, b);
}

CompositeKernel operator+(Kernel& a, Kernel& b) {
  return CompositeKernel::add(a, b);
}

// ----------------------------------------------------------------------------

SEKernel::SEKernel(double length, double signal_var) {
  setHyperparameters(Eigen::Vector2d{length, signal_var});
}

SEKernel::SEKernel(const Eigen::Vector2d& hyperparameters) {
  setHyperparameters(hyperparameters);
}

double SEKernel::eval(const Eigen::VectorXd& x,
                      std::size_t output_index) const {
  (void)output_index;
  (void)x;
  return m_signal_var;
}

double SEKernel::eval(const Eigen::VectorXd& x,
                      const Eigen::VectorXd& y,
                      std::size_t output_index) const {
  (void)output_index;
  const Eigen::VectorXd e = x - y;
  return m_signal_var * exp(-e.dot(e) / pow(m_length, 2) / 2);
}

Eigen::VectorXd SEKernel::grad(const Eigen::VectorXd& x,
                               const Eigen::VectorXd& y,
                               std::size_t output_index) const {
  (void)output_index;
  const Eigen::VectorXd e = (x - y) / m_length;
  double mahal2 = e.dot(e);
  double knorm = exp(-mahal2 / 2);
  double dkdl = m_signal_var * knorm * mahal2 / m_length;
  double dkds = knorm;
  Eigen::VectorXd g(2);
  g << dkdl, dkds;
  return g;
}

Eigen::VectorXd SEKernel::hyperparameters() const {
  Eigen::VectorXd p(2);
  p << m_length, m_signal_var;
  return p;
}

void SEKernel::setHyperparameters(const Eigen::VectorXd& p) {
  SIA_THROW_IF_NOT(p.size() == 2,
                   "SEKernel hyperparameter dim expexted to be 2");
  m_length = p(0);
  m_signal_var = p(1);
  SIA_THROW_IF_NOT(m_length > 0, "SEKernel expects length scale > 0");
  SIA_THROW_IF_NOT(m_signal_var > 0, "SEKernel expects signal var > 0");
}

std::size_t SEKernel::numHyperparameters() const {
  return 2;
}

// ----------------------------------------------------------------------------

NoiseKernel::NoiseKernel(double noise_var) {
  Eigen::VectorXd p(1);
  p << noise_var;
  setHyperparameters(p);
}

double NoiseKernel::eval(const Eigen::VectorXd& x,
                         std::size_t output_index) const {
  (void)x;
  (void)output_index;
  return m_noise_var;
}

double NoiseKernel::eval(const Eigen::VectorXd& x,
                         const Eigen::VectorXd& y,
                         std::size_t output_index) const {
  (void)x;
  (void)y;
  (void)output_index;
  return 0;
}

Eigen::VectorXd NoiseKernel::grad(const Eigen::VectorXd& x,
                                  const Eigen::VectorXd& y,
                                  std::size_t output_index) const {
  (void)output_index;
  double dkdn = x.isApprox(y) ? 1.0 : 0.0;
  Eigen::VectorXd g(1);
  g << dkdn;
  return g;
}

Eigen::VectorXd NoiseKernel::hyperparameters() const {
  Eigen::VectorXd p(1);
  p << m_noise_var;
  return p;
}

void NoiseKernel::setHyperparameters(const Eigen::VectorXd& p) {
  SIA_THROW_IF_NOT(p.size() == 1,
                   "NoiseKernel hyperparameter dim expexted to be 1");
  m_noise_var = p(0);
  SIA_THROW_IF_NOT(m_noise_var > 0, "NoiseKernel expects noise var > 0");
}

std::size_t NoiseKernel::numHyperparameters() const {
  return 1;
}

// ----------------------------------------------------------------------------

VariableNoiseKernel::VariableNoiseKernel(VarianceFunction var_function)
    : m_var_function(var_function) {}

double VariableNoiseKernel::eval(const Eigen::VectorXd& x,
                                 std::size_t output_index) const {
  const Eigen::VectorXd z = m_var_function(x);
  SIA_THROW_IF_NOT(int(output_index) < z.size(),
                   "VariableNoiseKernel var_function dimension not match GPR "
                   "output dimension");
  return z(output_index);
}

double VariableNoiseKernel::eval(const Eigen::VectorXd& x,
                                 const Eigen::VectorXd& y,
                                 std::size_t output_index) const {
  (void)x;
  (void)y;
  (void)output_index;
  return 0;
}

Eigen::VectorXd VariableNoiseKernel::grad(const Eigen::VectorXd& x,
                                          const Eigen::VectorXd& y,
                                          std::size_t output_index) const {
  (void)x;
  (void)y;
  (void)output_index;
  return Eigen::VectorXd(0);
}

Eigen::VectorXd VariableNoiseKernel::hyperparameters() const {
  return Eigen::VectorXd(0);
}

void VariableNoiseKernel::setHyperparameters(const Eigen::VectorXd& p) {
  SIA_THROW_IF_NOT(p.size() == 0,
                   "VariableNoiseKernel hyperparameter dim expexted to be 0");
}

std::size_t VariableNoiseKernel::numHyperparameters() const {
  return 0;
}

// ----------------------------------------------------------------------------

GPR::GPR(const Eigen::MatrixXd& input_samples,
         const Eigen::MatrixXd& output_samples,
         Kernel& kernel,
         double regularization)
    : m_belief(output_samples.rows()),
      m_kernel(kernel),
      m_regularization(regularization) {
  setData(input_samples, output_samples);
}

GPR::GPR(std::size_t input_dim,
         std::size_t output_dim,
         Kernel& kernel,
         double regularization)
    : m_input_dim(input_dim),
      m_output_dim(output_dim),
      m_belief(output_dim),
      m_kernel(kernel),
      m_regularization(regularization) {}

void GPR::setData(const Eigen::MatrixXd& input_samples,
                  const Eigen::MatrixXd& output_samples) {
  SIA_THROW_IF_NOT(
      input_samples.cols() == output_samples.cols(),
      "GPR training data samples (inputs, outputs) expected to have "
      "same number of columns");
  m_input_dim = input_samples.rows();
  m_output_dim = output_samples.rows();
  m_input_samples = input_samples;
  m_output_samples = output_samples;
  cacheRegressionModels();
}

const Gaussian& GPR::predict(const Eigen::VectorXd& x) {
  // Algorithm 2.1 in: http://www.gaussianprocess.org/gpml/chapters/RW.pdf
  // For each output channel
  std::size_t m = outputDim();
  Eigen::VectorXd mean(m), var(m);

  if (numSamples() == 0) {
    // Prior
    for (std::size_t i = 0; i < m; ++i) {
      double kxx = m_kernel.eval(x, i);
      mean(i) = 0;
      var(i) = kxx;
    }

  } else {
    // Posterior
    assert(m == m_models.size());
    for (std::size_t i = 0; i < m; ++i) {
      const Eigen::VectorXd kstar =
          evalTestData(m_kernel, m_input_samples, x, i);
      double kxx = m_kernel.eval(x, i);

      // Predictive mean, eqn. 2.25
      mean(i) = kstar.dot(m_models[i].cached_alpha);

      // Predictive variance, eqn. 2.26
      // This is unstable for some hyperparameters, see:
      // https://github.com/scikit-learn/scikit-learn/pull/19939
      const Eigen::VectorXd v = m_models[i].cached_L_inv * kstar;

      var(i) = kxx - v.dot(v);
    }
  }

  // Regularize the covariance matrix due to numerical errors in LLT
  for (std::size_t i = 0; i < m; ++i) {
    var(i) = std::max(var(i), m_regularization);
  }

  m_belief = Gaussian(mean, var.asDiagonal());
  return m_belief;
}

double GPR::negLogMarginalLik() const {
  SIA_THROW_IF_NOT(
      numSamples() > 0,
      "GPR negLogMarginalLik cannot be computed because no training "
      "data has been provided");

  // Eqn 2.30 in http://www.gaussianprocess.org/gpml/chapters/RW.pdf
  // Note the determinant of a triangular matrix is the product of its
  // diagonal And the log(|K|) = log(det(L))^2 = 2 * sum(log(diag(L)))
  // More stable
  double n = numSamples();
  double neg_log_lik = 0;
  for (std::size_t i = 0; i < outputDim(); ++i) {
    const auto& model = m_models.at(i);
    const Eigen::VectorXd& Y = m_output_samples.row(i);
    double log_kdet = 2 * model.cached_L.diagonal().array().log().sum();
    neg_log_lik +=
        (Y.transpose() * model.cached_alpha + log_kdet + n * log(2 * M_PI)) / 2;
  }
  return neg_log_lik;
}

Eigen::VectorXd GPR::negLogMarginalLikGrad() const {
  SIA_THROW_IF_NOT(numSamples() > 0,
                   "GPR negLogMarginalLikGrad cannot be computed because no "
                   "training data has been provided");

  // Eqn 5.9 in http://www.gaussianprocess.org/gpml/chapters/RW.pdf
  std::size_t np = m_kernel.numHyperparameters();
  Eigen::VectorXd g = Eigen::VectorXd::Zero(m_kernel.numHyperparameters());
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

void GPR::train(const std::vector<std::size_t>& hp_indices,
                double hp_min,
                double hp_max,
                const GD::Options& options) {
  SIA_THROW_IF_NOT(
      numSamples() > 0,
      "GPR cannot be trained because no training data has been provided");
  SIA_THROW_IF_NOT(hp_min > 0, "GPR expects hp_min > 0");
  SIA_THROW_IF_NOT(hp_max >= hp_min, "GPR expects hp_max >= hp_min");

  std::size_t n = numHyperparameters();
  if (!hp_indices.empty()) {
    SIA_THROW_IF_NOT(
        hp_indices.size() <= n,
        "GPR expects unique indices for trainable hyperparameters");
    n = hp_indices.size();
  }

  // Set up the optimizer.  Here we optimize the log of hyperparameters to
  // promote faster search over the hyperparameter space.  We use the chain rule
  // to compute the jacobian.  dL/d(ln(x)) = x dL/dx
  GD optm(log(hp_min) * Eigen::VectorXd::Ones(n),
                       log(hp_max) * Eigen::VectorXd::Ones(n), options);
  const Eigen::VectorXd x0 = hyperparameters();
  const Eigen::VectorXd ln_x0 = x0.array().log();

  if (hp_indices.empty()) {
    // If no indices are provided, optimize all hyperparameters
    auto loss = [this](const Eigen::VectorXd& ln_x) {
      const Eigen::VectorXd x = ln_x.array().exp();
      this->setHyperparameters(x);
      return this->negLogMarginalLik();
    };
    auto jacobian = [this](const Eigen::VectorXd& ln_x) {
      // Don't need to set x as the optimizer always calls loss before jacobian
      const Eigen::VectorXd x = ln_x.array().exp();
      const Eigen::VectorXd g = this->negLogMarginalLikGrad();
      const Eigen::VectorXd ghat = g.cwiseProduct(x);
      return ghat;
    };
    const Eigen::VectorXd ln_p = optm.minimize(loss, ln_x0, jacobian);
    const Eigen::VectorXd p = ln_p.array().exp();
    this->setHyperparameters(p);

  } else {
    // If indices are provided, optimize the subset of hyperparameters
    auto loss = [this, x0, hp_indices](const Eigen::VectorXd& ln_x) {
      const Eigen::VectorXd x = ln_x.array().exp();
      this->setHyperparameters(replace(x0, x, hp_indices));
      return this->negLogMarginalLik();
    };
    auto jacobian = [this, hp_indices](const Eigen::VectorXd& ln_x) {
      // Don't need to set x as the optimizer always calls loss before jacobian
      const Eigen::VectorXd x = ln_x.array().exp();
      const Eigen::VectorXd g = this->negLogMarginalLikGrad();
      const Eigen::VectorXd ghat = slice(g, hp_indices).cwiseProduct(x);
      return ghat;
    };
    const Eigen::VectorXd ln_p =
        optm.minimize(loss, slice(ln_x0, hp_indices), jacobian);
    const Eigen::VectorXd p = ln_p.array().exp();
    this->setHyperparameters(replace(x0, p, hp_indices));
  }
}

std::size_t GPR::inputDim() const {
  return m_input_dim;
}

std::size_t GPR::outputDim() const {
  return m_output_dim;
}

std::size_t GPR::numSamples() const {
  return m_input_samples.cols();
}

const Kernel& GPR::kernel() const {
  return m_kernel;
}

Eigen::VectorXd GPR::hyperparameters() const {
  return m_kernel.hyperparameters();
}

void GPR::setHyperparameters(const Eigen::VectorXd& hyperparams) {
  m_kernel.setHyperparameters(hyperparams);
  cacheRegressionModels();
}

std::size_t GPR::numHyperparameters() const {
  return m_kernel.numHyperparameters();
}

void GPR::cacheRegressionModels() {
  std::size_t n = numSamples();
  SIA_THROW_IF_NOT(
      n > 0,
      "GPR regression model cannot be computed because no training "
      "data has been provided");

  std::size_t m = outputDim();
  const Eigen::MatrixXd& X = m_input_samples;
  m_models.clear();
  m_models.reserve(m);
  for (std::size_t i = 0; i < m; ++i) {
    // Algorithm 2.1 in http://www.gaussianprocess.org/gpml/chapters/RW.pdf
    const Eigen::MatrixXd K = evalTrainData(m_kernel, X, i);
    const Eigen::VectorXd& Y = m_output_samples.row(i);
    // Cholesky decomposition of K
    Eigen::MatrixXd L;
    bool r = llt(K, L);
    SIA_THROW_IF_NOT(
        r, "Failed to compute cholesky decomposition of Ksample matrix");
    // Compute inverse of K
    const Eigen::MatrixXd L_inv =
        L.triangularView<Eigen::Lower>().solve(Eigen::MatrixXd::Identity(n, n));
    const Eigen::MatrixXd K_inv = L_inv.transpose() * L_inv;
    // Compute projection to output space
    const Eigen::MatrixXd alpha = K_inv * Y;
    // Pre-compute the gradient tensor
    std::vector<Eigen::MatrixXd> grad = gradTrainData(m_kernel, X, i);
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

}  // namespace sia
