/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/estimators/pf.h"
#include "sia/belief/gaussian.h"
#include "sia/belief/uniform.h"
#include "sia/common/logger.h"

namespace sia {

PF::PF(DynamicsModel& dynamics,
       MeasurementModel& measurement,
       const Particles& particles,
       const PF::Options& options)
    : m_dynamics(dynamics),
      m_measurement(measurement),
      m_belief(particles),
      m_options(options) {}

const Particles& PF::belief() const {
  return m_belief;
}

const Particles& PF::estimate(const Eigen::VectorXd& observation,
                              const Eigen::VectorXd& control) {
  m_metrics = PF::Metrics();
  m_belief = predict(control);
  m_belief = correct(observation);
  m_metrics.clockElapsedUs();
  return m_belief;
}

// TODO: Can parallelize
const Particles& PF::predict(const Eigen::VectorXd& control) {
  Eigen::VectorXd wp = m_belief.weights();
  Eigen::MatrixXd xp = m_belief.values();

  // These noise-inducing procedures are traditionally described in the
  // literature after the predict/correct steps, however we perform them before
  // so that we don't tamper with the posterior estimate.
  if (m_first_pass) {
    m_first_pass = false;
  } else {
    // Resample step: if particle variance is high (particles are collapsing)
    double neff = 1.0 / wp.array().pow(2).sum();
    double nt = m_options.resample_threshold * static_cast<double>(wp.size());
    m_metrics.ratio_effective_particles = neff;
    if (neff < nt) {
      systematicResampling(wp, xp);
      m_belief.setWeights(wp);
      m_metrics.resampled = 1;
    }

    // Roughening step: add some noise
    if (m_options.roughening_factor > 0) {
      roughenParticles(xp);
      m_metrics.roughened = 1;
    }
  }

  // Propogate the particles using the proposal density
  for (std::size_t i = 0; i < m_belief.numParticles(); ++i) {
    xp.col(i) = m_dynamics.dynamics(xp.col(i), control).sample();
  }

  m_belief.setValues(xp);
  return m_belief;
}

const Particles& PF::correct(const Eigen::VectorXd& observation) {
  Eigen::VectorXd lp = Eigen::VectorXd::Zero(m_belief.numParticles());
  Eigen::VectorXd wp = m_belief.weights();
  const Eigen::MatrixXd& xp = m_belief.values();

  // Compute the likelihood of each particle given the observation
  for (std::size_t i = 0; i < m_belief.numParticles(); ++i) {
    lp(i) = m_measurement.measurement(xp.col(i)).logProb(observation);
  }

  // Correction step: update the weights using Bayes' rule
  wp = (wp.array().log() + lp.array()).exp();
  wp = wp.array() / wp.sum();

  m_belief.setWeights(wp);
  return m_belief;
}

const PF::Metrics& PF::metrics() const {
  return m_metrics;
}

void PF::systematicResampling(Eigen::VectorXd& wp, Eigen::MatrixXd& xp) const {
  // From Thrun et. al., Table 4.4. page 110.
  Eigen::MatrixXd xpbar = Eigen::MatrixXd::Zero(xp.rows(), xp.cols());
  std::size_t M = wp.size();
  double Minv = 1.0 / static_cast<double>(M);
  Uniform g(0, Minv);
  double r = g.sample()(0);
  double c = wp(0);
  std::size_t i = 0;
  for (std::size_t m = 0; m < M; m++) {
    double u = r + static_cast<double>(m) * Minv;
    while (u > c) {
      i++;
      c += wp(i);
    }
    xpbar.col(m) = xp.col(i);
  }
  xp = xpbar;
  wp = Eigen::VectorXd::Ones(wp.size()) * Minv;
}

void PF::roughenParticles(Eigen::MatrixXd& xp) const {
  // From Crassidis and Junkins, pg 285.
  std::size_t n = xp.rows();  // number of dimensions
  std::size_t N = xp.cols();  // number of particles
  double K = pow(static_cast<double>(N), -1.0 / static_cast<double>(n));
  Eigen::MatrixXd J = Eigen::MatrixXd::Identity(n, n);
  for (std::size_t l = 0; l < n; ++l) {
    double el = xp.row(l).maxCoeff() - xp.row(l).minCoeff();
    J(l, l) = m_options.roughening_factor * el * K;
  }
  Gaussian g(Eigen::VectorXd::Zero(n), J);
  for (std::size_t j = 0; j < N; ++j) {
    xp.col(j) += g.sample();
  }
}

}  // namespace sia
