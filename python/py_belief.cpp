/// Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
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
      .def("covariance", &sia::Distribution::covariance);

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
      .def("setMean", &sia::Gaussian::setMean, py::arg("mean"))
      .def("setCovariance", &sia::Gaussian::setCovariance,
           py::arg("covariance"))
      .def("mahalanobis", &sia::Gaussian::mahalanobis, py::arg("x"))
      .def("checkDimensions", &sia::Gaussian::checkDimensions);

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
      .def("lower", &sia::Uniform::lower)
      .def("upper", &sia::Uniform::upper)
      .def("setLower", &sia::Uniform::setLower, py::arg("lower"))
      .def("setUpper", &sia::Uniform::setUpper, py::arg("upper"))
      .def("checkDimensions", &sia::Uniform::checkDimensions);

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
      .value("SILVERMAN", sia::KernelDensity::SILVERMAN)
      .value("SCOTT", sia::KernelDensity::SCOTT)
      .value("USER_SPECIFIED", sia::KernelDensity::USER_SPECIFIED)
      .export_values();

  m.def("bandwidthSilverman", &sia::bandwidthSilverman, py::arg("sigma"),
        py::arg("num_samples"));

  m.def("bandwidthScott", &sia::bandwidthScott, py::arg("sigma"),
        py::arg("num_samples"));

  kernel_density
      .def(py::init<const Eigen::MatrixXd&, const Eigen::VectorXd&,
                    sia::Kernel::Type, sia::KernelDensity::BandwidthMode,
                    double>(),
           py::arg("values"), py::arg("weights"),
           py::arg("type") = sia::Kernel::EPANECHNIKOV,
           py::arg("mode") = sia::KernelDensity::SILVERMAN,
           py::arg("bandwidth_scaling") = 1.0)
      .def(py::init<const sia::Particles&, sia::Kernel::Type,
                    sia::KernelDensity::BandwidthMode, double>(),
           py::arg("particles"), py::arg("type") = sia::Kernel::EPANECHNIKOV,
           py::arg("mode") = sia::KernelDensity::SILVERMAN,
           py::arg("bandwidth_scaling") = 1.0)
      .def("probability", &sia::KernelDensity::probability, py::arg("x"))
      .def("dimension", &sia::KernelDensity::dimension)
      .def("sample", &sia::KernelDensity::sample)
      .def("logProb", &sia::KernelDensity::logProb, py::arg("x"))
      .def("mean", &sia::KernelDensity::mean)
      .def("mode", &sia::KernelDensity::mode)
      .def("covariance", &sia::KernelDensity::covariance)
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
      .def("getBandwidth", &sia::KernelDensity::getBandwidth)
      .def("setBandwidthScaling", &sia::KernelDensity::setBandwidthScaling,
           py::arg("scaling"))
      .def("getBandwidthScaling", &sia::KernelDensity::getBandwidthScaling)
      .def("setBandwidthMode", &sia::KernelDensity::setBandwidthMode,
           py::arg("mode"))
      .def("getBandwidthMode", &sia::KernelDensity::getBandwidthMode)
      .def("setKernelType", &sia::KernelDensity::setKernelType, py::arg("type"))
      .def("getKernelType", &sia::KernelDensity::getKernelType);
}
