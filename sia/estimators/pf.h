/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#pragma once

#include "sia/belief/particles.h"
#include "sia/estimators/estimators.h"
#include "sia/models/models.h"

#include <Eigen/Dense>

namespace sia {

/// The particle filter sub-optimally estimates arbitrary Markov processes
/// and distributions using a sequential monte carlo approach.  The filter uses
/// samples to represent the non-Gaussian belief, and outperforms the extended
/// Kalman filter for highly nonlinear systems and non-Gaussian distributions.
/// Over time the filter exhibits particle collapse, where one particle carries
/// the entire estimate weight.  Resampling and roughening are injected noise
/// that protect against particle collapse.
class PF : public Estimator {
 public:
  /// The resample_threshold is [0, 1], represents a trigger threshold
  /// percentage of number of effective particles.  0 means no resampling is
  /// performed, 1 means it is always performed.
  explicit PF(DynamicsModel& dynamics,
              MeasurementModel& measurement,
              const Particles& particles,
              double resample_threshold = 1.0,
              double roughening_factor = 0.0);
  virtual ~PF() = default;
  void reset(const Particles& particles);
  const Particles& getBelief() const override;

  /// Performs the combined prediction and correction.
  const Particles& estimate(const Eigen::VectorXd& observation,
                            const Eigen::VectorXd& control) override;

  /// Propogates the belief through model dynamics.
  const Particles& predict(const Eigen::VectorXd& control) override;

  /// Propogates the belief through model dynamics.
  const Particles& correct(const Eigen::VectorXd& observation) override;

 private:
  void systematicResampling(Eigen::VectorXd& wp, Eigen::MatrixXd& xp) const;
  void roughenParticles(Eigen::MatrixXd& xp) const;

  DynamicsModel& m_dynamics;
  MeasurementModel& m_measurement;
  Particles m_belief;
  double m_resample_threshold;
  double m_roughening_factor;
  bool m_first_pass{true};
};

}  // namespace sia
