/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "python/py_belief.h"

// Define module
void export_py_belief(py::module& m_sup) {
  py::module m = m_sup;

  m.def("DEFAULT_SEED", []() { return sia::DEFAULT_SEED; });

  py::class_<sia::Generator>(m, "Generator")
      .def_static("instance", &sia::Generator::instance)
      .def("seed", &sia::Generator::seed, py::arg("seed") = sia::DEFAULT_SEED);

  py::class_<sia::Distribution, PyDistribution>(m, "Distribution")
      .def(py::init<sia::Generator&>(), py::arg("generator"))
      .def("dimension", &sia::Distribution::dimension)
      .def("sample",
           static_cast<const Eigen::VectorXd (sia::Distribution::*)()>(
               &sia::Distribution::sample))
      .def("logProb",
           static_cast<double (sia::Distribution::*)(const Eigen::VectorXd&)
                           const>(&sia::Distribution::logProb),
           py::arg("x"))
      .def("mean", &sia::Distribution::mean)
      .def("mode", &sia::Distribution::mode)
      .def("covariance", &sia::Distribution::covariance)
      .def("vectorize", &sia::Distribution::vectorize)
      .def("devectorize", &sia::Distribution::devectorize, py::arg("data"))
      .def("samples", &sia::Distribution::samples, py::arg("num_samples"));

  py::class_<sia::Inference, PyInference>(m, "Inference")
      .def(py::init<>())
      .def("predict", &sia::Inference::predict, py::arg("x"))
      .def("inputDimension", &sia::Inference::inputDimension)
      .def("outputDimension", &sia::Inference::outputDimension);

  m.def("logProb", &sia::logProb, py::arg("distribution"), py::arg("x"));

  m.def("logProb1d", &sia::logProb1d, py::arg("distribution"), py::arg("x"));

  m.def("logProb2d", &sia::logProb2d, py::arg("distribution"), py::arg("x"),
        py::arg("y"));

  py::class_<sia::Gaussian, sia::Distribution>(m, "Gaussian")
      .def(py::init<std::size_t>(), py::arg("dimension"))
      .def(py::init<double, double>(), py::arg("mean"), py::arg("variance"))
      .def(py::init<const Eigen::VectorXd&, const Eigen::MatrixXd&>(),
           py::arg("mean"), py::arg("covariance"))
      .def("dimension", &sia::Gaussian::dimension)
      .def("sample", &sia::Gaussian::sample)
      .def("logProb", &sia::Gaussian::logProb, py::arg("x"))
      .def("mean", &sia::Gaussian::mean)
      .def("mode", &sia::Gaussian::mode)
      .def("covariance", &sia::Gaussian::covariance)
      .def("vectorize", &sia::Gaussian::vectorize)
      .def("devectorize", &sia::Gaussian::devectorize, py::arg("data"))
      .def("samples", &sia::Gaussian::samples, py::arg("num_samples"))
      .def("setCovariance", &sia::Gaussian::setCovariance,
           py::arg("covariance"))
      .def("setMean", &sia::Gaussian::setMean, py::arg("mean"))
      .def("setMeanAndCov", &sia::Gaussian::setMeanAndCov, py::arg("mean"),
           py::arg("covariance"))
      .def("mahalanobis", &sia::Gaussian::mahalanobis, py::arg("x"))
      .def("maxLogProb", &sia::Gaussian::maxLogProb)
      .def_static("pdf", &sia::Gaussian::pdf)
      .def_static("cdf", &sia::Gaussian::cdf);

  py::class_<sia::Uniform, sia::Distribution>(m, "Uniform")
      .def(py::init<std::size_t>(), py::arg("dimension"))
      .def(py::init<double, double>(), py::arg("lower"), py::arg("upper"))
      .def(py::init<const Eigen::VectorXd&, const Eigen::VectorXd&>(),
           py::arg("lower"), py::arg("upper"))
      .def("dimension", &sia::Uniform::dimension)
      .def("sample", &sia::Uniform::sample)
      .def("logProb", &sia::Uniform::logProb, py::arg("x"))
      .def("mean", &sia::Uniform::mean)
      .def("mode", &sia::Uniform::mode)
      .def("covariance", &sia::Uniform::covariance)
      .def("vectorize", &sia::Uniform::vectorize)
      .def("devectorize", &sia::Uniform::devectorize, py::arg("data"))
      .def("samples", &sia::Uniform::samples, py::arg("num_samples"))
      .def("lower", &sia::Uniform::lower)
      .def("upper", &sia::Uniform::upper)
      .def("setLower", &sia::Uniform::setLower, py::arg("lower"))
      .def("setUpper", &sia::Uniform::setUpper, py::arg("upper"));

  py::class_<sia::Dirichlet, sia::Distribution>(m, "Dirichlet")
      .def(py::init<std::size_t>(), py::arg("dimension"))
      .def(py::init<double, double>(), py::arg("alpha"), py::arg("beta"))
      .def(py::init<const Eigen::VectorXd&>(), py::arg("alpha"))
      .def("dimension", &sia::Dirichlet::dimension)
      .def("sample", &sia::Dirichlet::sample)
      .def("logProb", &sia::Dirichlet::logProb, py::arg("x"))
      .def("mean", &sia::Dirichlet::mean)
      .def("mode", &sia::Dirichlet::mode)
      .def("covariance", &sia::Dirichlet::covariance)
      .def("vectorize", &sia::Dirichlet::vectorize)
      .def("devectorize", &sia::Dirichlet::devectorize, py::arg("data"))
      .def("samples", &sia::Dirichlet::samples, py::arg("num_samples"))
      .def("categorical", &sia::Dirichlet::categorical)
      .def("classify", &sia::Dirichlet::classify)
      .def("alpha", &sia::Dirichlet::alpha)
      .def("setAlpha", &sia::Dirichlet::setAlpha, py::arg("alpha"));

  py::class_<sia::Categorical, sia::Distribution>(m, "Categorical")
      .def(py::init<std::size_t>(), py::arg("dimension"))
      .def(py::init<const Eigen::VectorXd&>(), py::arg("probs"))
      .def("dimension", &sia::Categorical::dimension)
      .def("sample", &sia::Categorical::sample)
      .def("logProb", &sia::Categorical::logProb, py::arg("x"))
      .def("mean", &sia::Categorical::mean)
      .def("mode", &sia::Categorical::mode)
      .def("covariance", &sia::Categorical::covariance)
      .def("vectorize", &sia::Categorical::vectorize)
      .def("devectorize", &sia::Categorical::devectorize, py::arg("data"))
      .def("samples", &sia::Categorical::samples, py::arg("num_samples"))
      .def("classify", &sia::Categorical::classify)
      .def("probs", &sia::Categorical::probs)
      .def("setProbs", &sia::Categorical::setProbs, py::arg("probs"))
      .def(
          "oneHot",
          static_cast<Eigen::VectorXd (sia::Categorical::*)(std::size_t) const>(
              &sia::Categorical::oneHot),
          py::arg("category"))
      .def("oneHot",
           static_cast<Eigen::MatrixXd (sia::Categorical::*)(
               const Eigen::VectorXi&) const>(&sia::Categorical::oneHot),
           py::arg("category"))
      .def("category", &sia::Categorical::category, py::arg("probs"));

  py::class_<sia::Particles, sia::Distribution>(m, "Particles")
      .def(py::init<std::size_t, std::size_t, bool>(), py::arg("dimension"),
           py::arg("num_particles"), py::arg("weighted_stats") = false)
      .def(py::init<const Eigen::MatrixXd&, const Eigen::VectorXd&, bool>(),
           py::arg("values"), py::arg("weights"),
           py::arg("weighted_stats") = false)
      .def_static("init", &sia::Particles::init, py::arg("distribution"),
                  py::arg("num_particles"), py::arg("weighted_stats") = false)
      .def_static("gaussian", &sia::Particles::gaussian, py::arg("mean"),
                  py::arg("covariance"), py::arg("num_particles"),
                  py::arg("weighted_stats") = false)
      .def_static("uniform", &sia::Particles::uniform, py::arg("lower"),
                  py::arg("upper"), py::arg("num_particles"),
                  py::arg("weighted_stats") = false)
      .def("dimension", &sia::Particles::dimension)
      .def("sample", &sia::Particles::sample)
      .def("logProb", &sia::Particles::logProb, py::arg("x"))
      .def("mean", &sia::Particles::mean)
      .def("mode", &sia::Particles::mode)
      .def("covariance", &sia::Particles::covariance)
      .def("vectorize", &sia::Particles::vectorize)
      .def("devectorize", &sia::Particles::devectorize, py::arg("data"))
      .def("samples", &sia::Particles::samples, py::arg("num_samples"))
      .def("getUseWeightedStats", &sia::Particles::getUseWeightedStats)
      .def("setUseWeightedStats", &sia::Particles::setUseWeightedStats,
           py::arg("weighted_stats"))
      .def("numParticles", &sia::Particles::numParticles)
      .def("setValues", &sia::Particles::setValues, py::arg("values"))
      .def("values", &sia::Particles::values)
      .def("value", &sia::Particles::value, py::arg("i"))
      .def("setWeights", &sia::Particles::setWeights, py::arg("weights"))
      .def("weights", &sia::Particles::weights)
      .def("weight", &sia::Particles::weight, py::arg("i"));

  py::class_<sia::SmoothingKernel, PySmoothingKernel>(m, "SmoothingKernel")
      .def(py::init<>())
      .def("evaluate", &sia::SmoothingKernel::evaluate, py::arg("x"));

  py::class_<sia::UniformKernel, sia::SmoothingKernel>(m, "UniformKernel")
      .def(py::init<std::size_t>(), py::arg("dimension"))
      .def("evaluate", &sia::UniformKernel::evaluate, py::arg("x"));

  py::class_<sia::GaussianKernel, sia::SmoothingKernel>(m, "GaussianKernel")
      .def(py::init<std::size_t>(), py::arg("dimension"))
      .def("evaluate", &sia::GaussianKernel::evaluate, py::arg("x"));

  py::class_<sia::EpanechnikovKernel, sia::SmoothingKernel>(
      m, "EpanechnikovKernel")
      .def(py::init<std::size_t>(), py::arg("dimension"))
      .def("evaluate", &sia::EpanechnikovKernel::evaluate, py::arg("x"));

  py::class_<sia::KernelDensity, sia::Particles> kernel_density(
      m, "KernelDensity");

  py::enum_<sia::KernelDensity::BandwidthMode>(kernel_density, "BandwidthMode")
      .value("SCOTT_RULE", sia::KernelDensity::BandwidthMode::SCOTT_RULE)
      .value("USER_SPECIFIED",
             sia::KernelDensity::BandwidthMode::USER_SPECIFIED)
      .export_values();

  kernel_density
      .def(py::init<const Eigen::MatrixXd&, const Eigen::VectorXd&,
                    sia::SmoothingKernel&, sia::KernelDensity::BandwidthMode,
                    double>(),
           py::arg("values"), py::arg("weights"), py::arg("kernel"),
           py::arg("mode") = sia::KernelDensity::BandwidthMode::SCOTT_RULE,
           py::arg("bandwidth_scaling") = 1.0)
      .def(py::init<const sia::Particles&, sia::SmoothingKernel&,
                    sia::KernelDensity::BandwidthMode, double>(),
           py::arg("particles"), py::arg("kernel"),
           py::arg("mode") = sia::KernelDensity::BandwidthMode::SCOTT_RULE,
           py::arg("bandwidth_scaling") = 1.0)
      .def("probability", &sia::KernelDensity::probability, py::arg("x"))
      .def("dimension", &sia::KernelDensity::dimension)
      .def("sample", &sia::KernelDensity::sample)
      .def("logProb", &sia::KernelDensity::logProb, py::arg("x"))
      .def("mean", &sia::KernelDensity::mean)
      .def("mode", &sia::KernelDensity::mode)
      .def("covariance", &sia::KernelDensity::covariance)
      .def("vectorize", &sia::KernelDensity::vectorize)
      .def("devectorize", &sia::KernelDensity::devectorize, py::arg("data"))
      .def("samples", &sia::KernelDensity::samples, py::arg("num_samples"))
      .def("numParticles", &sia::KernelDensity::numParticles)
      .def("setValues", &sia::KernelDensity::setValues, py::arg("values"))
      .def("values", &sia::KernelDensity::values)
      .def("value", &sia::KernelDensity::value, py::arg("i"))
      .def("setWeights", &sia::KernelDensity::setWeights, py::arg("weights"))
      .def("weights", &sia::KernelDensity::weights)
      .def("weight", &sia::KernelDensity::weight, py::arg("i"))
      .def("setBandwidth",
           static_cast<void (sia::KernelDensity::*)(double)>(
               &sia::KernelDensity::setBandwidth),
           py::arg("h"))
      .def("setBandwidth",
           static_cast<void (sia::KernelDensity::*)(const Eigen::VectorXd&)>(
               &sia::KernelDensity::setBandwidth),
           py::arg("h"))
      .def("bandwidth", &sia::KernelDensity::bandwidth)
      .def("setBandwidthScaling", &sia::KernelDensity::setBandwidthScaling,
           py::arg("scaling"))
      .def("getBandwidthScaling", &sia::KernelDensity::getBandwidthScaling)
      .def("setBandwidthMode", &sia::KernelDensity::setBandwidthMode,
           py::arg("mode"))
      .def("getBandwidthMode", &sia::KernelDensity::getBandwidthMode)
      .def("kernel", &sia::KernelDensity::kernel);

  py::class_<sia::GMM, sia::Distribution, sia::Inference> gmm(m, "GMM");

  gmm.def(py::init<std::size_t, std::size_t>(), py::arg("K"),
          py::arg("dimension"))
      .def(py::init<const std::vector<sia::Gaussian>&,
                    const std::vector<double>&>(),
           py::arg("gaussians"), py::arg("weights"))
      .def(py::init<const Eigen::MatrixXd&, std::size_t, double>(),
           py::arg("samples"), py::arg("K"),
           py::arg("regularization") = sia::GMM::DEFAULT_REGULARIZATION)
      .def("dimension", &sia::GMM::dimension)
      .def("sample", &sia::GMM::sample)
      .def("logProb", &sia::GMM::logProb, py::arg("x"))
      .def("mean", &sia::GMM::mean)
      .def("mode", &sia::GMM::mode)
      .def("covariance", &sia::GMM::covariance)
      .def("vectorize", &sia::GMM::vectorize)
      .def("devectorize", &sia::GMM::devectorize, py::arg("data"))
      .def("predict", &sia::GMM::predict, py::arg("x"))
      .def("inputDimension", &sia::GMM::inputDimension)
      .def("outputDimension", &sia::GMM::outputDimension)
      .def("classify", &sia::GMM::classify, py::arg("x"))
      .def("numClusters", &sia::GMM::numClusters)
      .def("prior", &sia::GMM::prior, py::arg("i"))
      .def("priors", &sia::GMM::priors)
      .def("gaussian", &sia::GMM::gaussian, py::arg("i"))
      .def("gaussians", &sia::GMM::gaussians)
      .def_static("fit", &sia::GMM::fit, py::arg("samples"),
                  py::arg("gaussians"), py::arg("priors"), py::arg("K"),
                  py::arg("fit_method"), py::arg("init_method"),
                  py::arg("regularization"));

  py::enum_<sia::GMM::FitMethod>(gmm, "FitMethod")
      .value("KMEANS", sia::GMM::FitMethod::KMEANS)
      .value("GAUSSIAN_LIKELIHOOD", sia::GMM::FitMethod::GAUSSIAN_LIKELIHOOD)
      .export_values();

  py::enum_<sia::GMM::InitMethod>(gmm, "InitMethod")
      .value("STANDARD_RANDOM", sia::GMM::InitMethod::STANDARD_RANDOM)
      .value("WARM_START", sia::GMM::InitMethod::WARM_START)
      .export_values();

  py::class_<sia::GMR, sia::Inference>(m, "GMR")
      .def(py::init<const std::vector<sia::Gaussian>&,
                    const std::vector<double>&, std::vector<std::size_t>,
                    std::vector<std::size_t>, double>(),
           py::arg("gaussians"), py::arg("weights"), py::arg("input_indices"),
           py::arg("output_indices"),
           py::arg("regularization") = sia::GMM::DEFAULT_REGULARIZATION)
      .def(py::init<const sia::GMM&, std::vector<std::size_t>,
                    std::vector<std::size_t>, double>(),
           py::arg("gmm"), py::arg("input_indices"), py::arg("output_indices"),
           py::arg("regularization") = sia::GMM::DEFAULT_REGULARIZATION)
      .def("predict", &sia::GMR::predict, py::arg("x"))
      .def("inputDimension", &sia::GMR::inputDimension)
      .def("outputDimension", &sia::GMR::outputDimension)
      .def("gmm", &sia::GMR::gmm, py::return_value_policy::reference_internal);

  py::class_<sia::Kernel, PyKernel>(m, "Kernel")
      .def(py::init<>())
      .def("eval",
           static_cast<double (sia::Kernel::*)(
               const Eigen::VectorXd&, std::size_t) const>(&sia::Kernel::eval),
           py::arg("x"), py::arg("output_index"))
      .def("eval",
           static_cast<double (sia::Kernel::*)(
               const Eigen::VectorXd&, const Eigen::VectorXd&, std::size_t)
                           const>(&sia::Kernel::eval),
           py::arg("x"), py::arg("y"), py::arg("output_index"))
      .def("grad", &sia::Kernel::grad, py::arg("a"), py::arg("b"),
           py::arg("output_index"))
      .def("hyperparameters", &sia::Kernel::hyperparameters)
      .def("setHyperparameters", &sia::Kernel::setHyperparameters, py::arg("p"))
      .def("numHyperparameters", &sia::Kernel::numHyperparameters);

  py::class_<sia::CompositeKernel, sia::Kernel>(m, "CompositeKernel")
      .def_static("multiply", &sia::CompositeKernel::multiply, py::arg("a"),
                  py::arg("b"))
      .def_static("add", &sia::CompositeKernel::add, py::arg("a"), py::arg("b"))
      .def("eval",
           static_cast<double (sia::CompositeKernel::*)(const Eigen::VectorXd&,
                                                        std::size_t) const>(
               &sia::CompositeKernel::eval),
           py::arg("x"), py::arg("output_index"))
      .def("eval",
           static_cast<double (sia::CompositeKernel::*)(
               const Eigen::VectorXd&, const Eigen::VectorXd&, std::size_t)
                           const>(&sia::CompositeKernel::eval),
           py::arg("x"), py::arg("y"), py::arg("output_index"))
      .def("grad", &sia::CompositeKernel::grad, py::arg("a"), py::arg("b"),
           py::arg("output_index"))
      .def("hyperparameters", &sia::CompositeKernel::hyperparameters)
      .def("setHyperparameters", &sia::CompositeKernel::setHyperparameters,
           py::arg("p"))
      .def("numHyperparameters", &sia::CompositeKernel::numHyperparameters);

  py::class_<sia::SEKernel, sia::Kernel>(m, "SEKernel")
      .def(py::init<double, double>(), py::arg("length") = 1.0,
           py::arg("signal_var") = 1.0)
      .def(py::init<const Eigen::Vector2d&>(), py::arg("hyperparameters"))
      .def(
          "eval",
          static_cast<double (sia::SEKernel::*)(
              const Eigen::VectorXd&, std::size_t) const>(&sia::SEKernel::eval),
          py::arg("x"), py::arg("output_index"))
      .def("eval",
           static_cast<double (sia::SEKernel::*)(
               const Eigen::VectorXd&, const Eigen::VectorXd&, std::size_t)
                           const>(&sia::SEKernel::eval),
           py::arg("x"), py::arg("y"), py::arg("output_index"))
      .def("grad", &sia::SEKernel::grad, py::arg("a"), py::arg("b"),
           py::arg("output_index"))
      .def("hyperparameters", &sia::SEKernel::hyperparameters)
      .def("setHyperparameters", &sia::SEKernel::setHyperparameters,
           py::arg("p"))
      .def("numHyperparameters", &sia::SEKernel::numHyperparameters);

  py::class_<sia::NoiseKernel, sia::Kernel>(m, "NoiseKernel")
      .def(py::init<double>(), py::arg("noise_var") = 0.1)
      .def("eval",
           static_cast<double (sia::NoiseKernel::*)(const Eigen::VectorXd&,
                                                    std::size_t) const>(
               &sia::NoiseKernel::eval),
           py::arg("x"), py::arg("output_index"))
      .def("eval",
           static_cast<double (sia::NoiseKernel::*)(
               const Eigen::VectorXd&, const Eigen::VectorXd&, std::size_t)
                           const>(&sia::NoiseKernel::eval),
           py::arg("x"), py::arg("y"), py::arg("output_index"))
      .def("grad", &sia::NoiseKernel::grad, py::arg("a"), py::arg("b"),
           py::arg("output_index"))
      .def("hyperparameters", &sia::NoiseKernel::hyperparameters)
      .def("setHyperparameters", &sia::NoiseKernel::setHyperparameters,
           py::arg("p"))
      .def("numHyperparameters", &sia::NoiseKernel::numHyperparameters);

  py::class_<sia::VariableNoiseKernel, sia::Kernel>(m, "VariableNoiseKernel")
      .def(py::init<sia::VariableNoiseKernel::VarianceFunction>(),
           py::arg("var_function"))
      .def("eval",
           static_cast<double (sia::VariableNoiseKernel::*)(
               const Eigen::VectorXd&, std::size_t) const>(
               &sia::VariableNoiseKernel::eval),
           py::arg("x"), py::arg("output_index"))
      .def("eval",
           static_cast<double (sia::VariableNoiseKernel::*)(
               const Eigen::VectorXd&, const Eigen::VectorXd&, std::size_t)
                           const>(&sia::VariableNoiseKernel::eval),
           py::arg("x"), py::arg("y"), py::arg("output_index"))
      .def("grad", &sia::VariableNoiseKernel::grad, py::arg("a"), py::arg("b"),
           py::arg("output_index"))
      .def("hyperparameters", &sia::VariableNoiseKernel::hyperparameters)
      .def("setHyperparameters", &sia::VariableNoiseKernel::setHyperparameters,
           py::arg("p"))
      .def("numHyperparameters", &sia::VariableNoiseKernel::numHyperparameters);

  py::class_<sia::GPR, sia::Inference>(m, "GPR")
      .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&,
                    sia::Kernel&, double>(),
           py::arg("input_samples"), py::arg("output_samples"),
           py::arg("kernel"),
           py::arg("regularization") = sia::GPR::DEFAULT_REGULARIZATION)
      .def(py::init<std::size_t, std::size_t, sia::Kernel&, double>(),
           py::arg("input_dim"), py::arg("output_dim"), py::arg("kernel"),
           py::arg("regularization") = sia::GPR::DEFAULT_REGULARIZATION)
      .def("setData", &sia::GPR::setData, py::arg("input_samples"),
           py::arg("output_samples"))
      .def("predict", &sia::GPR::predict, py::arg("x"))
      .def("negLogMarginalLik", &sia::GPR::negLogMarginalLik)
      .def("negLogMarginalLikGrad", &sia::GPR::negLogMarginalLikGrad)
      .def("train", &sia::GPR::train,
           py::arg("hp_indices") = std::vector<std::size_t>{},
           py::arg("hp_min") = sia::GPR::DEFAULT_HP_MIN,
           py::arg("hp_max") = sia::GPR::DEFAULT_HP_MAX)
      .def("inputDimension", &sia::GPR::inputDimension)
      .def("outputDimension", &sia::GPR::outputDimension)
      .def("numSamples", &sia::GPR::numSamples)
      .def("kernel", &sia::GPR::kernel)
      .def("hyperparameters", &sia::GPR::hyperparameters)
      .def("setHyperparameters", &sia::GPR::setHyperparameters,
           py::arg("hyperparameters"));

  py::class_<sia::GPC, sia::Inference>(m, "GPC")
      .def(py::init<const Eigen::MatrixXd&, const Eigen::VectorXi&,
                    sia::Kernel&, double, double>(),
           py::arg("input_samples"), py::arg("output_samples"),
           py::arg("kernel"),
           py::arg("alpha") = sia::GPC::DEFAULT_CONCENTRATION,
           py::arg("regularization") = sia::GPR::DEFAULT_REGULARIZATION)
      .def(py::init<std::size_t, std::size_t, sia::Kernel&, double, double>(),
           py::arg("input_dim"), py::arg("output_dim"), py::arg("kernel"),
           py::arg("alpha") = sia::GPC::DEFAULT_CONCENTRATION,
           py::arg("regularization") = sia::GPR::DEFAULT_REGULARIZATION)
      .def("setData", &sia::GPC::setData, py::arg("input_samples"),
           py::arg("output_samples"))
      .def("predict", &sia::GPC::predict, py::arg("x"))
      .def("negLogMarginalLik", &sia::GPC::negLogMarginalLik)
      .def("negLogMarginalLikGrad", &sia::GPC::negLogMarginalLikGrad)
      .def("train", &sia::GPC::train,
           py::arg("hp_indices") = std::vector<std::size_t>{},
           py::arg("hp_min") = sia::GPR::DEFAULT_HP_MIN,
           py::arg("hp_max") = sia::GPR::DEFAULT_HP_MAX)
      .def("inputDimension", &sia::GPC::inputDimension)
      .def("outputDimension", &sia::GPC::outputDimension)
      .def("numSamples", &sia::GPC::numSamples)
      .def("kernel", &sia::GPC::kernel)
      .def("hyperparameters", &sia::GPC::hyperparameters)
      .def("setHyperparameters", &sia::GPC::setHyperparameters,
           py::arg("hyperparameters"))
      .def("numHyperparameters", &sia::GPC::numHyperparameters)
      .def("setAlpha", &sia::GPC::setAlpha, py::arg("alpha"))
      .def("alpha", &sia::GPC::alpha);
}
