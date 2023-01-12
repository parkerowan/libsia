/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include <sia/sia.h>
#include <chrono>
#include <fstream>
#include <iostream>

// constants
const std::size_t STATE_DIM = 8;
const std::size_t INPUT_DIM = 2;

// Simulation parameters
std::size_t num_steps = 2000;
std::size_t horizon = 500;
double process_noise = 1e-2;
double dt = 12 * 60;  // Time step in seconds (12 min)
std::string datafile = "/libsia/data/navigator.csv";

// Cost parameters
double input_cost = 5e14;
std::string algorithm = "ilqr";

// iLQR parameters
std::size_t max_lqr_iter = 3;
double cost_tol = 1e-1;

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
      std::cout << "  --max_lqr_iter <value> [iLQR] iterations\n";
      std::cout << "  --cost_tol <value> [iLQR] dJ convergence threshold\n";
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
    } else if (std::string(argv[i]) == "--max_lqr_iter") {
      max_lqr_iter = std::atoi(argv[i + 1]);
    } else if (std::string(argv[i]) == "--cost_tol") {
      cost_tol = std::atof(argv[i + 1]);
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

sia::NonlinearGaussianDynamicsCT create_dynamics(double q, double dt) {
  // Suppose that noise is added to all channels
  Eigen::MatrixXd Qpsd = q * Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);

  // Create the system
  return sia::NonlinearGaussianDynamicsCT(celestial_dynamics, Qpsd, dt,
                                          STATE_DIM, INPUT_DIM);
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
  Eigen::MatrixXd Q = Qf;
  Eigen::MatrixXd R = r * Eigen::MatrixXd::Identity(INPUT_DIM, INPUT_DIM);
  Eigen::VectorXd xd = Eigen::VectorXd::Zero(STATE_DIM);
  return sia::QuadraticCost(Qf, Q, R, xd);
}

sia::Controller* create_ilqr_controller(sia::LinearizableDynamics& dynamics,
                                        sia::QuadraticCost& cost,
                                        std::size_t horizon,
                                        std::size_t max_lqr_iter,
                                        double cost_tol) {
  std::vector<Eigen::VectorXd> u0;
  for (std::size_t i = 0; i < horizon; ++i) {
    u0.emplace_back(Eigen::VectorXd::Zero(INPUT_DIM));
  }
  return new sia::iLQR(dynamics, cost, u0, max_lqr_iter, cost_tol);
}

Eigen::VectorXd init_state() {
  Eigen::VectorXd mu = Eigen::VectorXd::Zero(STATE_DIM);
  mu.segment<2>(0) << 384472282, 0;  // Distance of moon to earth
  mu.segment<2>(2) << -67780000, 0;  // 4000km orbit, includes rad of earth
  mu.segment<2>(4) << 0, 1022;
  mu.segment<2>(6) << 0, -4660;
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
  auto dynamics = create_dynamics(process_noise, dt);
  sia::QuadraticCost cost = create_cost(input_cost / horizon);

  // Create the controller
  sia::Controller* controller{nullptr};
  if (algorithm == "ilqr") {
    controller =
        create_ilqr_controller(dynamics, cost, horizon, max_lqr_iter, cost_tol);
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
    x.setMean(dynamics.f(x.mean(), u));
    t += dt;
  }

  // close data file
  std::cout << "Test complete\n";
  ofs.close();

  delete controller;
  return 0;
}
