/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include <sia/sia.h>
#include <chrono>
#include <iostream>

// Simulation parameters
unsigned seed = 0;
std::size_t num_tests = 25;

bool parse_args(int argc, char* argv[]) {
  // Use a clock to generate a seed
  typedef std::chrono::high_resolution_clock myclock;
  myclock::time_point beginning = myclock::now();
  bool random_seed = true;

  for (int i = 1; i < argc; i += 2) {
    if (std::string(argv[i]) == "--help") {
      std::cout << "  --seed <value> Random number generator\n";
      std::cout << "  --num_tests <value> Number of tests per evaluation\n";
      return false;
    } else if (std::string(argv[i]) == "--seed") {
      seed = std::atoi(argv[i + 1]);
      random_seed = false;
    } else if (std::string(argv[i]) == "--num_tests") {
      num_tests = std::atoi(argv[i + 1]);
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

void run_gmm_prediction(std::size_t num_tests) {
  std::size_t m = 50;

  std::cout << "=========================================================\n";
  std::cout << "GMM Prediction\n";
  std::vector<std::size_t> dimension{1, 2, 4, 8, 16, 32};
  std::vector<std::size_t> num_clusters{1, 2, 3, 5, 7, 10, 15, 20};

  for (std::size_t i = 0; i < dimension.size(); ++i) {
    std::cout << "---------------------------------------------------------\n";
    std::cout << "Dimension, Samples, Clusters, Gmm Init (us), Gmm Pred (us)\n";
    std::size_t dim = dimension[i];
    std::size_t num_particles = m * dim;

    for (std::size_t j = 0; j < num_clusters.size(); ++j) {
      sia::Gaussian g(dim);
      sia::Particles s = sia::Particles::init(g, num_particles);
      Eigen::VectorXd x = g.sample();
      unsigned gmm_init_et = 0;
      unsigned gmm_pred_et = 0;
      for (std::size_t k = 0; k < num_tests; ++k) {
        auto tic = steady_clock::now();
        sia::GMM gmm(s.values(), num_clusters[j]);
        auto toc = steady_clock::now();
        gmm_init_et += get_elapsed_us(tic, toc);

        tic = steady_clock::now();
        auto y = gmm.classify(x);
        (void)(y);
        toc = steady_clock::now();
        gmm_pred_et += get_elapsed_us(tic, toc);
      }

      // Print the results
      double nt = double(num_tests);
      printf("%9d, %7d, %8d, %13.2f, %13.2f\n", int(dim), int(num_particles),
             int(num_clusters[j]), double(gmm_init_et) / nt,
             double(gmm_pred_et) / nt);
    }
  }
}

void run_gmr_prediction(std::size_t num_tests) {
  std::cout << "=========================================================\n";
  std::cout << "GMR Prediction\n";
  std::vector<std::size_t> dimension{5, 10, 15, 25, 30, 35, 50};
  std::vector<std::size_t> num_clusters{1, 2, 3, 5, 7, 10, 15, 20};

  for (std::size_t i = 0; i < dimension.size(); ++i) {
    std::cout << "---------------------------------------------------------\n";
    std::cout << "In Dim, Out Dim, Clusters, GMR Init (us), GMR Pred (us)\n";
    std::size_t na = double(dimension[i]) / 5;
    std::vector<std::size_t> inputs(na * 3);
    std::iota(inputs.begin(), inputs.end(), 0);
    std::vector<std::size_t> outputs(na * 2);
    std::iota(outputs.begin(), outputs.end(), na * 3);

    for (std::size_t j = 0; j < num_clusters.size(); ++j) {
      sia::Gaussian gx(inputs.size());
      Eigen::VectorXd x = gx.sample();
      sia::GMM gmm(num_clusters[j], dimension[i]);

      unsigned gmr_init_et = 0;
      unsigned gmr_pred_et = 0;
      for (std::size_t k = 0; k < num_tests; ++k) {
        auto tic = steady_clock::now();
        sia::GMR gmr(gmm, inputs, outputs);
        auto toc = steady_clock::now();
        gmr_init_et += get_elapsed_us(tic, toc);

        tic = steady_clock::now();
        auto y = gmr.predict(x);
        (void)(y);
        toc = steady_clock::now();
        gmr_pred_et += get_elapsed_us(tic, toc);
      }

      // Print the results
      double nt = double(num_tests);
      printf("%6d, %7d, %8d, %13.2f, %13.2f\n", int(inputs.size()),
             int(outputs.size()), int(num_clusters[j]),
             double(gmr_init_et) / nt, double(gmr_pred_et) / nt);
    }
  }
}

void run_gpr_prediction(std::size_t num_tests) {
  std::cout << "=========================================================\n";
  std::cout << "GPR Prediction\n";
  std::vector<std::size_t> dimension{5, 10, 15, 25, 30, 35, 50};
  std::vector<std::size_t> num_samples{20, 30, 50, 100, 150, 200, 250, 500};

  for (std::size_t i = 0; i < dimension.size(); ++i) {
    std::cout << "---------------------------------------------------------\n";
    std::cout << "In Dim, Out Dim, Num samples, GPR Init (us), GPR Pred (us)\n";
    std::size_t na = double(dimension[i]) / 5;
    std::size_t x_size = na * 3;
    std::size_t y_size = na * 2;

    for (std::size_t j = 0; j < num_samples.size(); ++j) {
      sia::Gaussian gxy(dimension[i]);
      Eigen::MatrixXd XY = sia::Particles::init(gxy, num_samples[j]).values();
      Eigen::MatrixXd X = XY.topRows(x_size);
      Eigen::MatrixXd Y = XY.bottomRows(y_size);

      sia::Gaussian gx(x_size);
      Eigen::VectorXd x = gx.sample();

      unsigned gpr_init_et = 0;
      unsigned gpr_pred_et = 0;
      for (std::size_t k = 0; k < num_tests; ++k) {
        auto tic = steady_clock::now();
        sia::GPR gpr(X, Y, 1.0, 1.0, 1.0);
        auto toc = steady_clock::now();
        gpr_init_et += get_elapsed_us(tic, toc);

        tic = steady_clock::now();
        auto y = gpr.predict(x);
        (void)(y);
        toc = steady_clock::now();
        gpr_pred_et += get_elapsed_us(tic, toc);
      }

      // Print the results
      double nt = double(num_tests);
      printf("%6d, %7d, %11d, %13.2f, %13.2f\n", int(x_size), int(y_size),
             int(num_samples[j]), double(gpr_init_et) / nt,
             double(gpr_pred_et) / nt);
    }
  }
}

// To profile, run:
// $ valgrind --tool=callgrind ./example-benchmarks
// $ kcachegrind
int main(int argc, char* argv[]) {
  if (not parse_args(argc, argv)) {
    return 0;
  }

  run_gmm_prediction(num_tests);
  run_gmr_prediction(num_tests);
  run_gpr_prediction(num_tests);

  return 0;
}
