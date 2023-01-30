/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
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
///
/// More information on the parameters is available in [1].
/// - resample_threshold: (>=0, <=1) trigger threshold on effective particles, 0
///   means no resampling is performed, 1 means it is always performed.
/// - roughening_factor: (>=0) added noise to roughen particles.
///
/// References:
/// [1] https://www.irisa.fr/aspi/legland/ensta/ref/arulampalam02a.pdf
/// [2] https://docs.ufpr.br/~danielsantos/ProbabilisticRobotics.pdf
/// [3] J. Crassidis and J. Junkins, Optimal Estimation of Dynamic Systems, 2nd
///     Edition, 2011.
class PF : public Estimator {
 public:
  /// Algorithm options
  struct Options {
    explicit Options() {}
    double resample_threshold = 1.0;
    double roughening_factor = 0.0;
  };

  struct Metrics : public BaseMetrics {
    double ratio_effective_particles{0};
    std::size_t resampled{0};
    std::size_t roughened{0};
  };

  explicit PF(DynamicsModel& dynamics,
              MeasurementModel& measurement,
              const Particles& particles,
              const Options& options = Options());
  virtual ~PF() = default;
  const Particles& belief() const override;

  /// Performs the combined prediction and correction.
  const Particles& estimate(const Eigen::VectorXd& observation,
                            const Eigen::VectorXd& control) override;

  /// Propogates the belief through model dynamics.
  const Particles& predict(const Eigen::VectorXd& control) override;

  /// Propogates the belief through model dynamics.
  const Particles& correct(const Eigen::VectorXd& observation) override;

  /// Return metrics from the latest step
  const Metrics& metrics() const override;

 private:
  void systematicResampling(Eigen::VectorXd& wp, Eigen::MatrixXd& xp) const;
  void roughenParticles(Eigen::MatrixXd& xp) const;

  DynamicsModel& m_dynamics;
  MeasurementModel& m_measurement;
  Particles m_belief;
  Options m_options;
  Metrics m_metrics;
  bool m_first_pass{true};
};

}  // namespace sia
