/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include <sia/sia.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>

// constants
const std::size_t STATE_DIM = 4;
const std::size_t INPUT_DIM = 1;

// Simulation parameters
unsigned seed = 0;
std::string init_case = "fine";
std::size_t num_trials = 1;
std::size_t num_bootstrap = 10;
std::size_t num_steps = 200;
std::size_t horizon = 100;
double process_noise = 1e-12;
double measurement_noise = 1e-12;
double dt = 2e-2;  // Time step in seconds (20 ms)
std::string datafile = "/libsia/data/cartpole.csv";

// Cost parameters
double input_cost = 1e-3;
std::string algorithm = "ilqr";

// iLQR parameters
std::size_t max_iter = 10;
std::size_t max_backsteps = 1;
double epsilon = 1e-1;
double tau = 0.5;
double min_z = 1e-1;
double mu = 0;

// MPPI parameters
std::size_t num_samples = 20;
double sigma = 2.0;
double lambda = 1.0;

// GMR parameters
std::size_t num_clusters = 10;
double regularization = 1e-1;  // This is super important!

// Data structure to collect data for a single trial
struct Trial {
  explicit Trial(std::size_t num_steps)
      : Xk(Eigen::MatrixXd::Zero(STATE_DIM, num_steps)),
        Uk(Eigen::MatrixXd::Zero(INPUT_DIM, num_steps)),
        Xkp1(Eigen::MatrixXd::Zero(STATE_DIM, num_steps)) {}
  Eigen::MatrixXd Xk;
  Eigen::MatrixXd Uk;
  Eigen::MatrixXd Xkp1;
};

// Data structure to collect multiple trials
struct Dataset {
  explicit Dataset(std::size_t num_steps) : num_steps(num_steps) {}
  void addTrial(const Trial& trial) { trials.emplace_back(trial); }
  Eigen::MatrixXd Xk() const {
    Eigen::MatrixXd Xk =
        Eigen::MatrixXd::Zero(STATE_DIM, trials.size() * num_steps);
    for (std::size_t i = 0; i < trials.size(); ++i) {
      Xk.middleCols(i * num_steps, num_steps) = trials.at(i).Xk;
    }
    return Xk;
  }
  Eigen::MatrixXd Uk() const {
    Eigen::MatrixXd Uk =
        Eigen::MatrixXd::Zero(INPUT_DIM, trials.size() * num_steps);
    for (std::size_t i = 0; i < trials.size(); ++i) {
      Uk.middleCols(i * num_steps, num_steps) = trials.at(i).Uk;
    }
    return Uk;
  }
  Eigen::MatrixXd Xkp1() const {
    Eigen::MatrixXd Xkp1 =
        Eigen::MatrixXd::Zero(STATE_DIM, trials.size() * num_steps);
    for (std::size_t i = 0; i < trials.size(); ++i) {
      Xkp1.middleCols(i * num_steps, num_steps) = trials.at(i).Xkp1;
    }
    return Xkp1;
  }
  std::vector<Trial> trials;
  std::size_t num_steps;
};

bool parse_args(int argc, char* argv[]) {
  // Use a clock to generate a seed
  typedef std::chrono::high_resolution_clock myclock;
  myclock::time_point beginning = myclock::now();
  bool random_seed = true;

  for (int i = 1; i < argc; i += 2) {
    if (std::string(argv[i]) == "--help") {
      std::cout << "  --seed <value> Random number generator\n";
      std::cout << "  --init_case <value> Options 'coarse' or 'fine'\n";
      std::cout << "  --num_trials <value> Num of trials\n";
      std::cout << "  --num_bootstrap <value> Num trials using an oracle\n";
      std::cout << "  --num_steps <value> Num simulation steps per trial\n";
      std::cout << "  --horizon <value> MPC optimization horizon\n";
      std::cout << "  --process_noise <value> Process noise variance\n";
      std::cout << "  --measurement_noise <value> Measurement noise variance\n";
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
      std::cout << "  --num_clusters <value> [GMR] Number of clusters\n";
      std::cout << "  --regularization <value> [GMR] Covariance regularizer\n";
      return false;
    } else if (std::string(argv[i]) == "--seed") {
      seed = std::atoi(argv[i + 1]);
      random_seed = false;
    } else if (std::string(argv[i]) == "--init_case") {
      init_case = std::string(argv[i + 1]);
    } else if (std::string(argv[i]) == "--num_trials") {
      num_trials = std::atoi(argv[i + 1]);
    } else if (std::string(argv[i]) == "--num_bootstrap") {
      num_bootstrap = std::atoi(argv[i + 1]);
    } else if (std::string(argv[i]) == "--num_steps") {
      num_steps = std::atoi(argv[i + 1]);
    } else if (std::string(argv[i]) == "--horizon") {
      horizon = std::atoi(argv[i + 1]);
    } else if (std::string(argv[i]) == "--process_noise") {
      process_noise = std::atof(argv[i + 1]);
    } else if (std::string(argv[i]) == "--measurement_noise") {
      measurement_noise = std::atof(argv[i + 1]);
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
    } else if (std::string(argv[i]) == "--num_clusters") {
      num_clusters = std::atoi(argv[i + 1]);
    } else if (std::string(argv[i]) == "--regularization") {
      regularization = std::atof(argv[i + 1]);
    }
  }

  if (random_seed) {
    myclock::duration d = myclock::now() - beginning;
    seed = d.count();
  }
  sia::Generator::instance().seed(seed);

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

sia::NonlinearGaussianDynamicsCT create_exact_dynamics(double q, double dt) {
  // Suppose that noise is added to all channels
  Eigen::MatrixXd Qpsd = q * Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);

  // Create the system
  return sia::NonlinearGaussianDynamicsCT(cartpole_dynamics, Qpsd, dt);
}

sia::NonlinearGaussianMeasurementCT create_measurement(double r, double dt) {
  // Suppose that noise is added to all channels
  Eigen::MatrixXd Rpsd = r * Eigen::MatrixXd::Identity(STATE_DIM, STATE_DIM);

  // Assume we can measure all states
  auto h = [](Eigen::VectorXd x) { return x; };

  // Create the system
  return sia::NonlinearGaussianMeasurementCT(h, Rpsd, dt);
}

sia::QuadraticCost create_cost(double r) {
  Eigen::DiagonalMatrix<double, STATE_DIM> Q;
  Q.diagonal() << 1.25, 6.0, 12.0, 0.25;
  Eigen::MatrixXd R(INPUT_DIM, INPUT_DIM);
  R << r;
  Eigen::VectorXd xd = Eigen::VectorXd::Zero(STATE_DIM);
  return sia::QuadraticCost(Q, Q, R, xd);
}

sia::Controller* create_ilqr_controller(sia::LinearizableDynamics& dynamics,
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
  return new sia::iLQR(dynamics, cost, u0, max_iter, max_backsteps, epsilon,
                       tau, min_z, mu);
}

sia::Controller* create_mppi_controller(sia::LinearizableDynamics& dynamics,
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
  return new sia::MPPI(dynamics, cost, u0, num_samples, Sigma, lambda);
}

Eigen::VectorXd init_state() {
  Eigen::VectorXd lower(STATE_DIM), upper(STATE_DIM);
  if (init_case == "coarse") {
    lower << -0.5, M_PI - 2, -2, -1;
    upper << +0.5, M_PI + 2, +2, +1;
  } else if (init_case == "fine") {
    lower << -0.3, M_PI / 2 - .3, -.3, -.3;
    upper << +0.3, M_PI / 2 + .3, +.3, +.3;
  } else {
    std::cerr << "Unknown init_case " << init_case << "\n";
  }

  sia::Uniform state(lower, upper);
  return state.sample();
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
  auto dynamics = create_exact_dynamics(process_noise, dt);
  auto measurement = create_measurement(measurement_noise, dt);
  auto cost = create_cost(input_cost);

  // Create the model
  sia::LinearizableDynamics* model = &dynamics;
  sia::GMRDynamics* gmr_dynamics{nullptr};

  // Run trials
  Dataset dataset(num_steps);
  for (std::size_t i = 0; i < num_trials; ++i) {
    // Create the controller
    sia::Controller* controller{nullptr};
    if (algorithm == "ilqr") {
      controller =
          create_ilqr_controller(*model, cost, horizon, max_iter, max_backsteps,
                                 epsilon, tau, min_z, mu);
    } else if (algorithm == "mppi") {
      controller = create_mppi_controller(*model, cost, horizon, num_samples,
                                          sigma, lambda);
    } else {
      std::cout << "Unknown controller " << algorithm
                << ", running with no control\n";
    }

    // Run the simulation with the controller in the loop
    Trial trial(num_steps);
    std::cout << "Trial " << i << " | Algorithm " << algorithm << "\n";

    sia::Gaussian x(STATE_DIM);
    x.setMean(init_state());
    double t = 0;
    for (std::size_t k = 0; k < num_steps; ++k) {
      // Compute the control
      Eigen::VectorXd u = Eigen::VectorXd::Zero(INPUT_DIM);
      if (controller != nullptr) {
        u = controller->policy(x);
      }

      // Write data
      Eigen::VectorXd y = measurement.measurement(x.mean()).sample();
      write_data(ofs, t, y, u);

      // Integrate forward
      trial.Xk.col(k) = x.mean();
      trial.Uk.col(k) = u;
      x.setMean(dynamics.dynamics(x.mean(), u).sample());
      trial.Xkp1.col(k) = x.mean();
      t += dt;
    }

    // Augment the data collection
    dataset.addTrial(trial);

    // If we are past bootstrapping, then learn the model and start using it
    // TODO: adjust the model fit with new data rather than relearn from scratch
    if (i == num_bootstrap - 1) {
      gmr_dynamics =
          new sia::GMRDynamics(dataset.Xk(), dataset.Uk(), dataset.Xkp1(),
                               num_clusters, regularization);
      model = gmr_dynamics;
      std::cout << "Trial " << i << " | Initializing GMR model, MSE "
                << gmr_dynamics->mse(dataset.Xk(), dataset.Uk(), dataset.Xkp1())
                << "\n";
      // } else if (i >= num_bootstrap) {
      // gmr_dynamics->train(dataset.Xk(), dataset.Uk(), dataset.Xkp1());
      // std::cout << "Trial " << i << " | Updating GMR model, MSE "
      //           << gmr_dynamics->mse(dataset.Xk(), dataset.Uk(),
      //           dataset.Xkp1())
      //           << "\n";
    }

    delete controller;
  }

  // close data file
  std::cout << "Test complete\n";
  ofs.close();

  // delete model;
  return 0;
}
