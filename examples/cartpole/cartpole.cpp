/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include <sia/sia.h>
#include <chrono>
#include <fstream>
#include <iostream>

// constants
const std::size_t STATE_DIM = 4;
const std::size_t INPUT_DIM = 1;

// Simulation parameters
std::size_t num_steps = 200;
std::size_t horizon = 100;
double process_noise = 0;
double measurement_noise = 0;
double dt = 2e-2;  // Time step in seconds (20 ms)
std::string datafile = "/libsia/data/cartpole.csv";

// Cost parameters
double input_cost = 1e-2;
std::string algorithm = "ilqr";

// iLQR parameters
std::size_t max_iter = 10;
std::size_t max_backsteps = 1;
double epsilon = 1e-1;
double tau = 0.5;
double min_z = 1e-1;
double mu = 0;

// MPPI parameters
std::size_t num_samples = 100;
double sigma = 10.0;
double lambda = 1.0;

bool parse_args(int argc, char* argv[]) {
  for (int i = 1; i < argc; i += 2) {
    if (std::string(argv[i]) == "--help") {
      std::cout << "  --num_steps <value> Number of simulation steps\n";
      std::cout << "  --horizon <value> MPC optimization horizon\n";
      std::cout << "  --process_noise <value> Process noise variance\n";
      std::cout << "  --dt <value> Simulation time step (s)\n";
      std::cout << "  --datafile <value> File path the csv data output\n";
      std::cout << "  --input_cost <value> Input cost coefficient\n";
      std::cout << "  --algorithm <value> Options 'ilqr' or 'mppi'\n";
      std::cout << "  --max_iter <value> [iLQR] iterations\n";
      std::cout << "  --max_backsteps <value> [iLQR] backsteps per iteration\n";
      std::cout << "  --epsilon <value> [iLQR] dJ convergence threshold\n";
      std::cout << "  --tau <value> [iLQR] backstep rate\n";
      std::cout << "  --min_z <value> [iLQR] backstep convergence threshold\n";
      std::cout << "  --mu <value> [iLQR] state update regularization\n";
      std::cout << "  --num_samples <value> [MPPI] Number of samples\n";
      std::cout << "  --sigma <value> [MPPI] Control sampling variance\n";
      std::cout << "  --lambda <value> [MPPI] Free energy temperature\n";
      return false;
    } else if (std::string(argv[i]) == "--num_steps") {
      num_steps = std::atoi(argv[i + 1]);
    } else if (std::string(argv[i]) == "--horizon") {
      horizon = std::atoi(argv[i + 1]);
    } else if (std::string(argv[i]) == "--process_noise") {
      process_noise = std::atof(argv[i + 1]);
    } else if (std::string(argv[i]) == "--dt") {
      dt = std::atof(argv[i + 1]);
    } else if (std::string(argv[i]) == "--datafile") {
      datafile = std::string(argv[i + 1]);
    } else if (std::string(argv[i]) == "--input_cost") {
      input_cost = std::atof(argv[i + 1]);
    } else if (std::string(argv[i]) == "--algorithm") {
      algorithm = std::string(argv[i + 1]);
    } else if (std::string(argv[i]) == "--max_iter") {
      max_iter = std::atoi(argv[i + 1]);
    } else if (std::string(argv[i]) == "--max_backsteps") {
      max_backsteps = std::atoi(argv[i + 1]);
    } else if (std::string(argv[i]) == "--epsilon") {
      epsilon = std::atof(argv[i + 1]);
    } else if (std::string(argv[i]) == "--tau") {
      tau = std::atof(argv[i + 1]);
    } else if (std::string(argv[i]) == "--min_z") {
      min_z = std::atof(argv[i + 1]);
    } else if (std::string(argv[i]) == "--mu") {
      mu = std::atof(argv[i + 1]);
    } else if (std::string(argv[i]) == "--num_samples") {
      num_samples = std::atoi(argv[i + 1]);
    } else if (std::string(argv[i]) == "--sigma") {
      sigma = std::atof(argv[i + 1]);
    } else if (std::string(argv[i]) == "--lambda") {
      lambda = std::atof(argv[i + 1]);
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

// write data header
void write_header(std::ofstream& ofs) {
  ofs << "t,";
  ofs << "p,a,v,w,";
  ofs << "f";
  ofs << "\n";
}

// write to file
void write_data(std::ofstream& ofs,
                double t,
                const Eigen::VectorXd& x,
                const Eigen::VectorXd& u) {
  ofs << t << ",";
  ofs << x(0) << "," << x(1) << "," << x(2) << "," << x(3) << ",";
  ofs << u(0);
  ofs << "\n";
}

// Create a system of equations for the cartpole dynamics
// m l \dot{v} \cos(a) + m l^2 \dot{w} - m g l \sin(a) = 0
// (m + m_c) \dot{v} + m l \dot{w} \cos(a) - m l w^2 \sin(a) = f
//
// States: x = [p, a, v, w]
// Inputs: u = [f]
//
// p - cart position (m)
// a - pendulum orientation (rad)
// v - cart velocity (m/s)
// w - pendulum angular velocity (rad/s)
// f - force acting on the cart (N)
const Eigen::VectorXd cartpole_dynamics(const Eigen::VectorXd& x,
                                        const Eigen::VectorXd& u) {
  double g = 9.8;   // Gravitational acceleration
  double l = 0.75;  // Pendulum length
  double m = 0.15;  // Point mass at pendulum tip
  double mc = 1.0;  // Cart mass

  Eigen::MatrixXd A(2, 2);
  Eigen::VectorXd b(2);

  double a = x(1);
  double v = x(2);
  double w = x(3);

  A << cos(a), l, m + mc, m * l * cos(a);
  b << g * sin(a), m * l * pow(w, 2) * sin(a) + u(0);

  // https://eigen.tuxfamily.org/dox/group__TutorialLinearAlgebra.html
  Eigen::VectorXd y = A.colPivHouseholderQr().solve(b);

  Eigen::VectorXd xdot = Eigen::VectorXd::Zero(STATE_DIM);
  xdot(0) = v;
  xdot(1) = w;
  xdot(2) = y(0);
  xdot(3) = y(1);
  return xdot;
}

sia::NonlinearGaussianCT create_system(double q, double r, double dt) {
  // Suppose that noise is added to all channels
  Eigen::MatrixXd C = Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
  Eigen::MatrixXd Q = q * Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);
  Eigen::MatrixXd R = r * Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);

  // Assume we can measure all states
  auto h = [](Eigen::VectorXd x) { return x; };

  // Create the system
  return sia::NonlinearGaussianCT(cartpole_dynamics, h, C, Q, R, dt);
}

sia::QuadraticCost create_cost(double r = 1e-2) {
  Eigen::DiagonalMatrix<double, STATE_DIM> Q;
  Q.diagonal() << 1.25, 1.0, 12.0, 0.25;
  Eigen::MatrixXd R(INPUT_DIM, INPUT_DIM);
  R << r;
  Eigen::VectorXd xd = Eigen::VectorXd::Zero(STATE_DIM);
  return sia::QuadraticCost(Q, Q, R, xd);
}

sia::Controller* create_ilqr_controller(sia::NonlinearGaussian& system,
                                        sia::QuadraticCost& cost,
                                        std::size_t horizon,
                                        std::size_t max_iter,
                                        std::size_t max_backsteps,
                                        double epsilon,
                                        double tau,
                                        double min_z,
                                        double mu) {
  std::vector<Eigen::VectorXd> u0;
  for (std::size_t i = 0; i < horizon; ++i) {
    u0.emplace_back(Eigen::VectorXd::Zero(INPUT_DIM));
  }
  return new sia::iLQR(system, cost, u0, max_iter, max_backsteps, epsilon, tau,
                       min_z, mu);
}

sia::Controller* create_mppi_controller(sia::NonlinearGaussian& system,
                                        sia::QuadraticCost& cost,
                                        std::size_t horizon,
                                        std::size_t num_samples,
                                        double sigma,
                                        double lambda) {
  std::vector<Eigen::VectorXd> u0;
  for (std::size_t i = 0; i < horizon; ++i) {
    u0.emplace_back(Eigen::VectorXd::Zero(INPUT_DIM));
  }
  Eigen::MatrixXd Sigma(INPUT_DIM, INPUT_DIM);
  Sigma << sigma;
  return new sia::MPPI(system, cost, u0, num_samples, Sigma, lambda);
}

Eigen::VectorXd init_state() {
  Eigen::VectorXd mu(STATE_DIM);
  mu << -1, 1, 2, 0;
  return mu;
}

// To profile, run:
// $ valgrind --tool=callgrind ./example-cartpole
// $ kcachegrind
int main(int argc, char* argv[]) {
  if (not parse_args(argc, argv)) {
    return 0;
  }

  // Open file to write test data to
  std::ofstream ofs;
  ofs.open(datafile, std::ofstream::out);
  write_header(ofs);

  // Create the system and cost function
  auto system = create_system(process_noise, measurement_noise, dt);
  auto cost = create_cost(input_cost);

  // Create the controller
  sia::Controller* controller{nullptr};
  if (algorithm == "ilqr") {
    controller = create_ilqr_controller(system, cost, horizon, max_iter,
                                        max_backsteps, epsilon, tau, min_z, mu);
  } else if (algorithm == "mppi") {
    controller = create_mppi_controller(system, cost, horizon, num_samples,
                                        sigma, lambda);
  } else {
    std::cout << "Unknown controller " << algorithm
              << ", running with no control\n";
  }

  // Run the simulation with the controller in the loop
  sia::Gaussian x(STATE_DIM);
  x.setMean(init_state());
  double t = 0;
  for (std::size_t i = 0; i < num_steps; ++i) {
    // Compute the control
    Eigen::VectorXd u = Eigen::VectorXd::Zero(INPUT_DIM);
    if (controller != nullptr) {
      auto tic = steady_clock::now();
      u = controller->policy(x);
      auto toc = steady_clock::now();
      std::cout << algorithm << " - i=" << i << "/" << num_steps
                << ", elapsed=" << get_elapsed_us(tic, toc) << " us\n";
    }

    // Write data
    write_data(ofs, t, x.mean(), u);

    // Integrate forward
    x.setMean(system.f(x.mean(), u));
    t += dt;
  }

  // close data file
  std::cout << "Test complete\n";
  ofs.close();

  delete controller;
  return 0;
}
