/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "sia/belief/gaussian.h"
#include "sia/belief/uniform.h"
#include "sia/estimators/pf.h"

#include <glog/logging.h>

namespace sia {

PF::PF(MarkovProcess& system,
       const Particles& particles,
       double resample_threshold,
       double roughening_factor)
    : m_system(system),
      m_belief(particles),
      m_resample_threshold(resample_threshold),
      m_roughening_factor(roughening_factor) {}

void PF::reset(const Particles& particles) {
  m_belief = particles;
  m_first_pass = true;
}

const Particles& PF::getBelief() const {
  return m_belief;
}

const Particles& PF::estimate(const Eigen::VectorXd& observation,
                              const Eigen::VectorXd& control) {
  m_belief = predict(control);
  m_belief = correct(observation);
  return m_belief;
}

const Particles& PF::predict(const Eigen::VectorXd& control) {
  Eigen::MatrixXd& xp = m_belief.m_values;
  Eigen::VectorXd& wp = m_belief.m_weights;

  // These noise-inducing procedures are traditionally described in the
  // literature after the predict/correct steps, however we perform them before
  // so that we don't tamper with the posterior estimate.
  if (m_first_pass) {
    m_first_pass = false;
  } else {
    // Resample step: if particle variance is high (particles are collapsing)
    double neff = 1.0 / wp.array().pow(2).sum();
    double nt = m_resample_threshold * static_cast<double>(wp.size());
    if (neff < nt) {
      VLOG(1) << "Performing resampling, Neff=" << neff << " Nt=" << nt;
      systematicResampling(wp, xp);
    }

    // Roughening step: add some noise
    if (m_roughening_factor > 0) {
      roughenParticles(xp);
    }
  }

  // Propogate the particles using the proposal density
  for (std::size_t i = 0; i < m_belief.numParticles(); ++i) {
    xp.col(i) = m_system.dynamics(xp.col(i), control).sample();
  }
  return m_belief;
}

const Particles& PF::correct(const Eigen::VectorXd& observation) {
  Eigen::VectorXd lp = Eigen::VectorXd::Zero(m_belief.numParticles());
  Eigen::MatrixXd& xp = m_belief.m_values;
  Eigen::VectorXd& wp = m_belief.m_weights;

  // Compute the likelihood of each particle given the observation
  for (std::size_t i = 0; i < m_belief.numParticles(); ++i) {
    lp(i) = m_system.measurement(xp.col(i)).logProb(observation);
  }

  // Correction step: update the weights usinfg Bayes' rule
  wp = (wp.array().log() + lp.array()).exp();
  wp = wp.array() / wp.sum();
  VLOG(2) << "lp(0): " << lp(0) << " wp(0): " << wp(0);
  return m_belief;
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
    J(l, l) = m_roughening_factor * el * K;
  }
  Gaussian g(Eigen::VectorXd::Zero(n), J);
  for (std::size_t j = 0; j < N; ++j) {
    xp.col(j) += g.sample();
  }
}

}  // namespace sia
