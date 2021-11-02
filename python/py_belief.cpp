/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
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
      .def("setMean", &sia::Gaussian::setMean, py::arg("mean"))
      .def("setCovariance", &sia::Gaussian::setCovariance,
           py::arg("covariance"))
      .def("setMeanAndCov", &sia::Gaussian::setMeanAndCov, py::arg("mean"),
           py::arg("covariance"))
      .def("mahalanobis", &sia::Gaussian::mahalanobis, py::arg("x"));

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

  py::class_<sia::Kernel, PyKernel> kernel(m, "Kernel");

  py::enum_<sia::Kernel::Type>(kernel, "Type")
      .value("UNIFORM", sia::Kernel::UNIFORM)
      .value("GAUSSIAN", sia::Kernel::GAUSSIAN)
      .value("EPANECHNIKOV", sia::Kernel::EPANECHNIKOV)
      .export_values();

  kernel.def(py::init<>())
      .def("evaluate", &sia::Kernel::evaluate, py::arg("x"))
      .def("type", &sia::Kernel::type);

  py::class_<sia::UniformKernel, sia::Kernel>(m, "UniformKernel")
      .def(py::init<std::size_t>(), py::arg("dimension"))
      .def("evaluate", &sia::UniformKernel::evaluate, py::arg("x"))
      .def("type", &sia::UniformKernel::type);

  py::class_<sia::GaussianKernel, sia::Kernel>(m, "GaussianKernel")
      .def(py::init<std::size_t>(), py::arg("dimension"))
      .def("evaluate", &sia::GaussianKernel::evaluate, py::arg("x"))
      .def("type", &sia::GaussianKernel::type);

  py::class_<sia::EpanechnikovKernel, sia::Kernel>(m, "EpanechnikovKernel")
      .def(py::init<std::size_t>(), py::arg("dimension"))
      .def("evaluate", &sia::EpanechnikovKernel::evaluate, py::arg("x"))
      .def("type", &sia::EpanechnikovKernel::type);

  py::class_<sia::KernelDensity, sia::Particles> kernel_density(
      m, "KernelDensity");

  py::enum_<sia::KernelDensity::BandwidthMode>(kernel_density, "BandwidthMode")
      .value("SCOTT_RULE", sia::KernelDensity::SCOTT_RULE)
      .value("USER_SPECIFIED", sia::KernelDensity::USER_SPECIFIED)
      .export_values();

  kernel_density
      .def(py::init<const Eigen::MatrixXd&, const Eigen::VectorXd&,
                    sia::Kernel::Type, sia::KernelDensity::BandwidthMode,
                    double>(),
           py::arg("values"), py::arg("weights"),
           py::arg("type") = sia::Kernel::EPANECHNIKOV,
           py::arg("mode") = sia::KernelDensity::SCOTT_RULE,
           py::arg("bandwidth_scaling") = 1.0)
      .def(py::init<const sia::Particles&, sia::Kernel::Type,
                    sia::KernelDensity::BandwidthMode, double>(),
           py::arg("particles"), py::arg("type") = sia::Kernel::EPANECHNIKOV,
           py::arg("mode") = sia::KernelDensity::SCOTT_RULE,
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
      .def("setKernelType", &sia::KernelDensity::setKernelType, py::arg("type"))
      .def("getKernelType", &sia::KernelDensity::getKernelType);

  py::class_<sia::GMM, sia::Distribution> gmm(m, "GMM");

  gmm.def(py::init<std::size_t, std::size_t>(), py::arg("K"),
          py::arg("dimension"))
      .def(py::init<const std::vector<sia::Gaussian>&,
                    const std::vector<double>&>(),
           py::arg("gaussians"), py::arg("weights"))
      .def(py::init<const Eigen::MatrixXd&, std::size_t, double>(),
           py::arg("samples"), py::arg("K"), py::arg("regularization") = 1e-6)
      .def("dimension", &sia::GMM::dimension)
      .def("sample", &sia::GMM::sample)
      .def("logProb", &sia::GMM::logProb, py::arg("x"))
      .def("mean", &sia::GMM::mean)
      .def("mode", &sia::GMM::mode)
      .def("covariance", &sia::GMM::covariance)
      .def("vectorize", &sia::GMM::vectorize)
      .def("devectorize", &sia::GMM::devectorize, py::arg("data"))
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

  py::class_<sia::GMR, sia::GMM, sia::Distribution>(m, "GMR")
      .def(py::init<const std::vector<sia::Gaussian>&,
                    const std::vector<double>&, std::vector<std::size_t>,
                    std::vector<std::size_t>, double>(),
           py::arg("gaussians"), py::arg("weights"), py::arg("input_indices"),
           py::arg("output_indices"), py::arg("regularization") = 1e-6)
      .def(py::init<const sia::GMM&, std::vector<std::size_t>,
                    std::vector<std::size_t>, double>(),
           py::arg("gmm"), py::arg("input_indices"), py::arg("output_indices"),
           py::arg("regularization") = 1e-6)
      .def("dimension", &sia::GMR::dimension)
      .def("sample", &sia::GMR::sample)
      .def("logProb", &sia::GMR::logProb, py::arg("x"))
      .def("mean", &sia::GMR::mean)
      .def("mode", &sia::GMR::mode)
      .def("covariance", &sia::GMR::covariance)
      .def("vectorize", &sia::GMR::vectorize)
      .def("devectorize", &sia::GMR::devectorize, py::arg("data"))
      .def("classify", &sia::GMR::classify, py::arg("x"))
      .def("numClusters", &sia::GMR::numClusters)
      .def("prior", &sia::GMR::prior, py::arg("i"))
      .def("gaussian", &sia::GMR::gaussian, py::arg("i"))
      .def("priors", &sia::GMR::priors)
      .def("gaussians", &sia::GMR::gaussians)
      .def("predict", &sia::GMR::predict, py::arg("x"));

  py::class_<sia::GPR> gpr(m, "GPR");

  py::enum_<sia::GPR::CovFunction>(gpr, "CovFunction")
      .value("SQUARED_EXPONENTIAL", sia::GPR::CovFunction::SQUARED_EXPONENTIAL)
      .export_values();

  gpr.def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&, double,
                   double, double, sia::GPR::CovFunction>(),
          py::arg("input_samples"), py::arg("output_samples"), py::arg("varn"),
          py::arg("varf"), py::arg("length"),
          py::arg("type") = sia::GPR::CovFunction::SQUARED_EXPONENTIAL)
      .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&,
                    const Eigen::MatrixXd&, double, double,
                    sia::GPR::CovFunction>(),
           py::arg("input_samples"), py::arg("output_samples"), py::arg("varn"),
           py::arg("varf"), py::arg("length"),
           py::arg("type") = sia::GPR::CovFunction::SQUARED_EXPONENTIAL)
      .def("predict", &sia::GPR::predict, py::arg("x"))
      .def("numSamples", &sia::GPR::numSamples)
      .def("inputDimension", &sia::GPR::inputDimension)
      .def("outputDimension", &sia::GPR::outputDimension);
}
