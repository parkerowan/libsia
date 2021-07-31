/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include <sia/sia.h>
#include <chrono>
#include <fstream>
#include <iostream>

// constants
const std::size_t NUM_AXES = 2;
const std::size_t STATE_DIM = 8;

// Simulation parameters
std::size_t num_steps = 2000;
std::size_t horizon = 500;
double process_noise = 1e-2;
double measurement_noise = 1e-3;
double dt = 12 * 60;  // Time step in seconds (12 min)
std::string datafile = "/libsia/data/navigator.csv";

// Cost parameters
double input_cost = 1e12;
std::string algorithm = "ilqr";

// iLQR parameters
// FIXME: Increasing the number of iterations causes Quu to not be pos def.
std::size_t max_iter = 3;
std::size_t max_backsteps = 1;
double epsilon = 1e-1;
double tau = 0.5;
double min_z = 1e-1;
double mu = 0;

bool parse_args(int argc, char* argv[]) {
  for (int i = 1; i < argc; i += 2) {
    if (std::string(argv[i]) == "--help") {
      std::cout << "  --num_steps <value> Number of simulation steps\n";
      std::cout << "  --horizon <value> MPC optimization horizon\n";
      std::cout << "  --process_noise <value> Process noise variance\n";
      std::cout << "  --dt <value> Simulation time step (s)\n";
      std::cout << "  --datafile <value> File path the csv data output\n";
      std::cout << "  --input_cost <value> Input cost coefficient\n";
      std::cout << "  --algorithm <value> Options 'ilqr'\n";
      std::cout << "  --max_iter <value> [iLQR] iterations\n";
      std::cout << "  --max_backsteps <value> [iLQR] backsteps per iteration\n";
      std::cout << "  --epsilon <value> [iLQR] dJ convergence threshold\n";
      std::cout << "  --tau <value> [iLQR] backstep rate\n";
      std::cout << "  --min_z <value> [iLQR] backstep convergence threshold\n";
      std::cout << "  --mu <value> [iLQR] state update regularization\n";
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

void print_metrics(const sia::iLQR::Metrics& metrics) {
  std::cout << "iLQR Metrics\n";
  std::cout << "iterations: " << metrics.iter << "\n";
  std::cout << "dJ:         " << metrics.dJ << "\n";
  std::cout << "J:          " << metrics.J << "\n";
  std::cout << "z:          " << metrics.z << "\n";
  std::cout << "elapsed_us: " << metrics.elapsed_us << "\n";
  std::cout << "backsteps:  " << metrics.backstep_iter << "\n";
  std::cout << "alpha:      " << metrics.alpha << "\n";
}

// write data header
void write_header(std::ofstream& ofs) {
  ofs << "t,";
  ofs << "xm0,xm1,xc0,xc1,vm0,vm1,vc0,vc1,";
  ofs << "f0,f1";
  ofs << "\n";
}

// write to file
void write_data(std::ofstream& ofs,
                double t,
                const Eigen::VectorXd& x,
                const Eigen::VectorXd& u) {
  ofs << t << ",";
  ofs << x(0) << "," << x(1) << "," << x(2) << "," << x(3) << "," << x(4) << ","
      << x(5) << "," << x(6) << "," << x(7) << ",";
  ofs << u(0) << "," << u(1);
  ofs << "\n";
}

// Create a system of equations for the celestial dynamics
// \ddot{q}_i = \sum_{j \neq i} G m_j (q_j - q_i) / \|q_j - q_i\|^3
//
// States: x = [xm, xc, vm, vc]
// Inputs: u = [f]
//
// xm - moon position (m)
// xc - craft position (m)
// vm - moon velocity (m/s)
// vc - craft velocity (m/s)
// f - force acting on the craft (N)
const Eigen::VectorXd celestial_dynamics(const Eigen::VectorXd& x,
                                         const Eigen::VectorXd& u) {
  double me = 5.972e24;    // (kg) mass of earth
  double mm = 7.348e22;    // (kg) mass of moon
  double mc = 15103;       // (kg) Apollo 11 lunar module mass
  double G = 6.67408E-11;  // (m3 kg-1 s-2) Gavitational constant

  // states
  Eigen::Vector2d xe = Eigen::Vector2d::Zero();
  Eigen::Vector2d xm = x.segment<2>(0);
  Eigen::Vector2d xc = x.segment<2>(2);
  Eigen::Vector2d vm = x.segment<2>(4);
  Eigen::Vector2d vc = x.segment<2>(6);

  // accelerations
  double dem = pow((xe - xm).norm(), 3);
  double dec = pow((xe - xc).norm(), 3);
  double dmc = pow((xm - xc).norm(), 3);

  // ignore craft effect on moon
  Eigen::Vector2d am = G * (me * (xe - xm) / dem);

  // include full gravitational effects on craft, add thrust force (N)
  Eigen::Vector2d ac =
      G * (me * (xe - xc) / dec + mm * (xm - xc) / dmc) + u / mc;

  Eigen::VectorXd xdot = Eigen::VectorXd::Zero(STATE_DIM);
  xdot.segment<2>(0) = vm;
  xdot.segment<2>(2) = vc;
  xdot.segment<2>(4) = am;
  xdot.segment<2>(6) = ac;
  return xdot;
}

sia::NonlinearGaussianCT create_system(double q, double r, double dt) {
  // Suppose that noise is added to all channels
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(STATE_DIM, NUM_AXES);
  C.block<2, 2>(6, 0) = Eigen::Matrix2d::Identity();
  Eigen::MatrixXd Q = q * Eigen::MatrixXd::Identity(NUM_AXES, NUM_AXES);
  Eigen::MatrixXd R = r * Eigen::MatrixXd::Identity(NUM_AXES, NUM_AXES);

  // Assume we can measure all states
  auto h = [](Eigen::VectorXd x) { return x; };

  // Create the system
  return sia::NonlinearGaussianCT(celestial_dynamics, h, C, Q, R, dt);
}

sia::QuadraticCost create_cost(double r) {
  Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
  Eigen::MatrixXd Qf = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
  Qf.block<2, 2>(0, 0) = I;
  Qf.block<2, 2>(2, 2) = I;
  Qf.block<2, 2>(0, 2) = -I;
  Qf.block<2, 2>(2, 0) = -I;
  Qf.block<2, 2>(4, 4) = I;
  Qf.block<2, 2>(6, 6) = I;
  Qf.block<2, 2>(4, 6) = -I;
  Qf.block<2, 2>(6, 4) = -I;
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(STATE_DIM, STATE_DIM);
  Eigen::MatrixXd R = r * Eigen::MatrixXd::Identity(NUM_AXES, NUM_AXES);
  Eigen::VectorXd xd = Eigen::VectorXd::Zero(STATE_DIM);
  return sia::QuadraticCost(Qf, Q, R, xd);
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
    u0.emplace_back(Eigen::VectorXd::Zero(NUM_AXES));
  }
  return new sia::iLQR(system, cost, u0, max_iter, max_backsteps, epsilon, tau,
                       min_z, mu);
}

Eigen::VectorXd init_state() {
  Eigen::VectorXd mu = Eigen::VectorXd::Zero(STATE_DIM);
  mu.segment<2>(0) << 384400000, 0;
  mu.segment<2>(2) << -67000000, 0;  // 4500 km orbit, includes radius of earth
  mu.segment<2>(4) << 0, 970;
  mu.segment<2>(6) << 0, -10000;
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
  sia::QuadraticCost cost = create_cost(input_cost / horizon);

  // Create the controller
  sia::Controller* controller{nullptr};
  if (algorithm == "ilqr") {
    controller = create_ilqr_controller(system, cost, horizon, max_iter,
                                        max_backsteps, epsilon, tau, min_z, mu);
  } else {
    std::cout << "Unknown controller " << algorithm
              << ", running with no control\n";
  }

  // Run the simulation with the controller in the loop
  sia::Gaussian x(STATE_DIM);
  x.setMean(init_state());
  double t = 0;
  for (std::size_t i = 0; i < num_steps - 1; ++i) {
    // Compute the control
    Eigen::VectorXd u = Eigen::VectorXd::Zero(1);
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
