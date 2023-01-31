/// Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "python/py_estimators.h"

// Define module
void export_py_estimators(py::module& m_sup) {
  py::module m = m_sup;

  py::class_<sia::Estimator, PyRecursiveBayesEstimator>(m, "Estimator")
      .def(py::init<>())
      .def("belief", &sia::Estimator::belief)
      .def("estimate", &sia::Estimator::estimate, py::arg("observation"),
           py::arg("control"), py::return_value_policy::reference_internal)
      .def("predict", &sia::Estimator::predict, py::arg("control"),
           py::return_value_policy::reference_internal)
      .def("correct", &sia::Estimator::correct, py::arg("observation"),
           py::return_value_policy::reference_internal)
      .def("metrics", &sia::Estimator::metrics,
           py::return_value_policy::reference_internal);

  auto kf = py::class_<sia::KF, sia::Estimator>(m, "KF");

  py::class_<sia::KF::Metrics, sia::BaseMetrics>(kf, "Metrics")
      .def(py::init<>())
      .def("clockElapsedUs", &sia::KF::Metrics::clockElapsedUs)
      .def_readwrite("elapsed_us", &sia::KF::Metrics::elapsed_us)
      .def_readwrite("kalman_gain_norm", &sia::KF::Metrics::kalman_gain_norm);

  kf.def(py::init<sia::LinearGaussianDynamics&, sia::LinearGaussianMeasurement&,
                  const sia::Gaussian&>(),
         py::arg("dynamics"), py::arg("measurement"), py::arg("state"))
      .def("belief", &sia::KF::belief)
      .def("estimate", &sia::KF::estimate, py::arg("observation"),
           py::arg("control"), py::return_value_policy::reference_internal)
      .def("predict", &sia::KF::predict, py::arg("control"),
           py::return_value_policy::reference_internal)
      .def("correct", &sia::KF::correct, py::arg("observation"),
           py::return_value_policy::reference_internal)
      .def("metrics", &sia::KF::metrics,
           py::return_value_policy::reference_internal);

  auto ekf = py::class_<sia::EKF, sia::Estimator>(m, "EKF");

  py::class_<sia::EKF::Metrics, sia::BaseMetrics>(ekf, "Metrics")
      .def(py::init<>())
      .def("clockElapsedUs", &sia::EKF::Metrics::clockElapsedUs)
      .def_readwrite("elapsed_us", &sia::EKF::Metrics::elapsed_us)
      .def_readwrite("kalman_gain_norm", &sia::EKF::Metrics::kalman_gain_norm);

  ekf.def(py::init<sia::LinearizableDynamics&, sia::LinearizableMeasurement&,
                   const sia::Gaussian&>(),
          py::arg("dynamics"), py::arg("measurement"), py::arg("state"))
      .def("belief", &sia::EKF::belief)
      .def("estimate", &sia::EKF::estimate, py::arg("observation"),
           py::arg("control"), py::return_value_policy::reference_internal)
      .def("predict", &sia::EKF::predict, py::arg("control"),
           py::return_value_policy::reference_internal)
      .def("correct", &sia::EKF::correct, py::arg("observation"),
           py::return_value_policy::reference_internal)
      .def("metrics", &sia::EKF::metrics,
           py::return_value_policy::reference_internal);

  auto pf = py::class_<sia::PF, sia::Estimator>(m, "PF");

  py::class_<sia::PF::Options>(pf, "Options")
      .def(py::init<>())
      .def_readwrite("resample_threshold",
                     &sia::PF::Options::resample_threshold)
      .def_readwrite("roughening_factor", &sia::PF::Options::roughening_factor);

  py::class_<sia::PF::Metrics, sia::BaseMetrics>(pf, "Metrics")
      .def(py::init<>())
      .def("clockElapsedUs", &sia::PF::Metrics::clockElapsedUs)
      .def_readwrite("elapsed_us", &sia::PF::Metrics::elapsed_us)
      .def_readwrite("ratio_effective_particles",
                     &sia::PF::Metrics::ratio_effective_particles)
      .def_readwrite("resampled", &sia::PF::Metrics::resampled)
      .def_readwrite("roughened", &sia::PF::Metrics::roughened);

  pf.def(py::init<sia::DynamicsModel&, sia::MeasurementModel&,
                  const sia::Particles&, const sia::PF::Options&>(),
         py::arg("dynamics"), py::arg("measurement"), py::arg("particles"),
         py::arg("options") = sia::PF::Options())
      .def("belief", &sia::PF::belief)
      .def("estimate", &sia::PF::estimate, py::arg("observation"),
           py::arg("control"), py::return_value_policy::reference_internal)
      .def("predict", &sia::PF::predict, py::arg("control"),
           py::return_value_policy::reference_internal)
      .def("correct", &sia::PF::correct, py::arg("observation"),
           py::return_value_policy::reference_internal)
      .def("metrics", &sia::PF::metrics,
           py::return_value_policy::reference_internal);
}
