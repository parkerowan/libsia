/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include <sia/sia.h>
#include <chrono>
#include <fstream>
#include <iostream>

// Test parameters
std::size_t max_dim_power = 5;
std::size_t num_steps = 20;
std::size_t horizon = 20;
std::string datafile = "/libsia/data/benchmarks.csv";

bool parse_args(int argc, char* argv[]) {
  for (int i = 1; i < argc; i += 2) {
    if (std::string(argv[i]) == "--help") {
      std::cout << "  --max_dim_power <value> Power of the max dimension \n";
      std::cout << "  --num_steps <value> Number of simulation steps\n";
      std::cout << "  --horizon <value> MPC optimization horizon\n";
      std::cout << "  --datafile <value> File path the csv data output\n";
      return false;
    } else if (std::string(argv[i]) == "--max_dim_power") {
      max_dim_power = std::atoi(argv[i + 1]);
    } else if (std::string(argv[i]) == "--num_steps") {
      num_steps = std::atoi(argv[i + 1]);
    } else if (std::string(argv[i]) == "--horizon") {
      horizon = std::atoi(argv[i + 1]);
    } else if (std::string(argv[i]) == "--datafile") {
      datafile = std::string(argv[i + 1]);
    }
  }
  return true;
}

using steady_clock = std::chrono::steady_clock;
static unsigned get_elapsed_us(steady_clock::time_point tic,
                               steady_clock::time_point toc) {
  return std::chrono::duration_cast<std::chrono::microseconds>(toc - tic)
      .count();
};

sia::LinearGaussianDynamics create_dynamics(std::size_t nstates,
                                            std::size_t ncontrols) {
  Eigen::MatrixXd F = Eigen::MatrixXd::Random(nstates, nstates);
  Eigen::MatrixXd G = Eigen::MatrixXd::Random(nstates, ncontrols);
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(nstates, nstates);
  return sia::LinearGaussianDynamics(F, G, Q);
}

sia::LinearGaussianMeasurement create_measurement(std::size_t nstates,
                                                  std::size_t nmeas) {
  Eigen::MatrixXd H = Eigen::MatrixXd::Random(nmeas, nstates);
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(nmeas, nmeas);
  return sia::LinearGaussianMeasurement(H, R);
}

sia::QuadraticCost create_cost(std::size_t nstates, std::size_t ncontrols) {
  Eigen::MatrixXd Qf = Eigen::MatrixXd::Identity(nstates, nstates);
  Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(nstates, nstates);
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(ncontrols, ncontrols);
  Eigen::VectorXd xd = Eigen::VectorXd::Random(nstates);
  return sia::QuadraticCost(Qf, Q, R, xd);
}

// write data header
void write_header(std::ofstream& ofs) {
  ofs << "Nstate,Ncontrol,Nmeas,Algorithm,Time (µs)\n";
}

// write to file
void write_data(std::ofstream& ofs,
                std::size_t nstates,
                std::size_t ncontrols,
                std::size_t nmeas,
                double lqr_et_us,
                double ilqr_et_us,
                double mppi_et_us_100,
                double mppi_et_us_500,
                double mppi_et_us_2000,
                double kf_et_us,
                double ekf_et_us,
                double pf_et_us_100,
                double pf_et_us_500,
                double pf_et_us_2000) {
  ofs << nstates << "," << ncontrols << "," << nmeas << ",LQR," << lqr_et_us
      << "\n";
  ofs << nstates << "," << ncontrols << "," << nmeas << ",iLQR," << ilqr_et_us
      << "\n";
  ofs << nstates << "," << ncontrols << "," << nmeas << ",MPPI (100),"
      << mppi_et_us_100 << "\n";
  ofs << nstates << "," << ncontrols << "," << nmeas << ",MPPI (500),"
      << mppi_et_us_500 << "\n";
  ofs << nstates << "," << ncontrols << "," << nmeas << ",MPPI (2000),"
      << mppi_et_us_2000 << "\n";
  ofs << nstates << "," << ncontrols << "," << nmeas << ",KF," << kf_et_us
      << "\n";
  ofs << nstates << "," << ncontrols << "," << nmeas << ",EKF," << ekf_et_us
      << "\n";
  ofs << nstates << "," << ncontrols << "," << nmeas << ",PF (100),"
      << pf_et_us_100 << "\n";
  ofs << nstates << "," << ncontrols << "," << nmeas << ",PF (500),"
      << pf_et_us_500 << "\n";
  ofs << nstates << "," << ncontrols << "," << nmeas << ",PF (2000),"
      << pf_et_us_2000 << "\n";
}

// To profile, run:
// $ valgrind --tool=callgrind ./benchmarks
// $ kcachegrind
int main(int argc, char* argv[]) {
  if (not parse_args(argc, argv)) {
    return 0;
  }

  std::size_t nstates = 4;
  std::size_t ncontrols = 4;
  std::size_t nmeas = 4;
  std::size_t dim_mult = 2;

  // Open file to write test data to
  std::ofstream ofs;
  ofs.open(datafile, std::ofstream::out);
  write_header(ofs);

  for (std::size_t i = 1; i <= max_dim_power; ++i) {
    std::cout << "Running iter " << i << "\n";

    nstates *= dim_mult;
    ncontrols *= dim_mult;
    nmeas *= dim_mult;

    // Init the state
    sia::Gaussian state(nstates);
    sia::Particles particles_100(nstates, 100);
    sia::Particles particles_500(nstates, 500);
    sia::Particles particles_2000(nstates, 2000);

    // Create the system equations
    auto dynamics = create_dynamics(nstates, ncontrols);
    auto measurement = create_measurement(nstates, nmeas);
    auto cost = create_cost(nstates, ncontrols);

    // Create zero control seed
    std::vector<Eigen::VectorXd> u0(horizon, Eigen::VectorXd::Zero(ncontrols));

    // Create the estimators
    sia::KF kf(dynamics, measurement, state);
    sia::EKF ekf(dynamics, measurement, state);
    sia::PF pf_100(dynamics, measurement, particles_100);
    sia::PF pf_500(dynamics, measurement, particles_500);
    sia::PF pf_2000(dynamics, measurement, particles_2000);

    // Create the controllers
    sia::LQR lqr(dynamics, cost, horizon);
    sia::iLQR ilqr(dynamics, cost, u0);

    Eigen::MatrixXd sigma = Eigen::MatrixXd::Identity(ncontrols, ncontrols);
    sia::MPPI mppi_100(dynamics, cost, u0, 100, sigma);
    sia::MPPI mppi_500(dynamics, cost, u0, 500, sigma);
    sia::MPPI mppi_2000(dynamics, cost, u0, 2000, sigma);

    // Initialize the state
    Eigen::VectorXd x = Eigen::VectorXd::Random(nstates);

    // Elapsed time (us)
    unsigned lqr_et_us = 0;
    unsigned ilqr_et_us = 0;
    unsigned mppi_et_us_100 = 0;
    unsigned mppi_et_us_500 = 0;
    unsigned mppi_et_us_2000 = 0;
    unsigned kf_et_us = 0;
    unsigned ekf_et_us = 0;
    unsigned pf_et_us_100 = 0;
    unsigned pf_et_us_500 = 0;
    unsigned pf_et_us_2000 = 0;

    // Compute the algorithms
    for (std::size_t k = 0; k < num_steps; ++k) {
      // Compute a control using the belief in state
      auto tic = steady_clock::now();
      auto u = lqr.policy(state);
      auto toc = steady_clock::now();
      lqr_et_us += get_elapsed_us(tic, toc);

      tic = steady_clock::now();
      u = ilqr.policy(state);
      toc = steady_clock::now();
      ilqr_et_us += get_elapsed_us(tic, toc);

      tic = steady_clock::now();
      u = mppi_100.policy(state);
      toc = steady_clock::now();
      mppi_et_us_100 += get_elapsed_us(tic, toc);

      tic = steady_clock::now();
      u = mppi_500.policy(state);
      toc = steady_clock::now();
      mppi_et_us_500 += get_elapsed_us(tic, toc);

      tic = steady_clock::now();
      u = mppi_2000.policy(state);
      toc = steady_clock::now();
      mppi_et_us_2000 += get_elapsed_us(tic, toc);

      // Simulate the system forward and sample from the propogated
      x = dynamics.dynamics(x, u).sample();
      auto y = measurement.measurement(x).sample();

      // Update the belief based on the measurement we took and the control we
      tic = steady_clock::now();
      state = kf.estimate(y, u);
      toc = steady_clock::now();
      kf_et_us += get_elapsed_us(tic, toc);

      tic = steady_clock::now();
      state = ekf.estimate(y, u);
      toc = steady_clock::now();
      ekf_et_us += get_elapsed_us(tic, toc);

      tic = steady_clock::now();
      particles_100 = pf_100.estimate(y, u);
      toc = steady_clock::now();
      pf_et_us_100 += get_elapsed_us(tic, toc);

      tic = steady_clock::now();
      particles_500 = pf_500.estimate(y, u);
      toc = steady_clock::now();
      pf_et_us_500 += get_elapsed_us(tic, toc);

      tic = steady_clock::now();
      particles_2000 = pf_2000.estimate(y, u);
      toc = steady_clock::now();
      pf_et_us_2000 += get_elapsed_us(tic, toc);
    }

    write_data(ofs, nstates, ncontrols, nmeas,
               double(lqr_et_us) / double(num_steps),
               double(ilqr_et_us) / double(num_steps),
               double(mppi_et_us_100) / double(num_steps),
               double(mppi_et_us_500) / double(num_steps),
               double(mppi_et_us_2000) / double(num_steps),
               double(kf_et_us) / double(num_steps),
               double(ekf_et_us) / double(num_steps),
               double(pf_et_us_100) / double(num_steps),
               double(pf_et_us_500) / double(num_steps),
               double(pf_et_us_2000) / double(num_steps));
  }

  ofs.close();
  return 0;
}
