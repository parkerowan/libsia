/// Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "python/py_estimators.h"

// Define module
void export_py_estimators(py::module& m_sup) {
  py::module m = m_sup;

  py::class_<sia::RecursiveBayesEstimator, PyRecursiveBayesEstimator>(
      m, "RecursiveBayesEstimator")
      .def("getBelief", &sia::RecursiveBayesEstimator::getBelief)
      .def("estimate", &sia::RecursiveBayesEstimator::estimate,
           py::arg("observation"), py::arg("control"),
           py::return_value_policy::reference_internal)
      .def("predict", &sia::RecursiveBayesEstimator::predict,
           py::arg("control"), py::return_value_policy::reference_internal)
      .def("correct", &sia::RecursiveBayesEstimator::correct,
           py::arg("observation"), py::return_value_policy::reference_internal);

  py::class_<sia::KalmanFilter, sia::RecursiveBayesEstimator>(m, "KalmanFilter")
      .def(py::init<sia::LinearGaussian&, const sia::Gaussian&>(),
           py::arg("system"), py::arg("state"))
      .def("reset", &sia::KalmanFilter::reset, py::arg("state"))
      .def("getBelief", &sia::KalmanFilter::getBelief)
      .def("estimate", &sia::KalmanFilter::estimate, py::arg("observation"),
           py::arg("control"), py::return_value_policy::reference_internal)
      .def("predict", &sia::KalmanFilter::predict, py::arg("control"),
           py::return_value_policy::reference_internal)
      .def("correct", &sia::KalmanFilter::correct, py::arg("observation"),
           py::return_value_policy::reference_internal);

  py::class_<sia::ExtendedKalmanFilter, sia::RecursiveBayesEstimator>(
      m, "ExtendedKalmanFilter")
      .def(py::init<sia::NonlinearGaussian&, const sia::Gaussian&>(),
           py::arg("system"), py::arg("state"))
      .def("reset", &sia::ExtendedKalmanFilter::reset, py::arg("state"))
      .def("getBelief", &sia::ExtendedKalmanFilter::getBelief)
      .def("estimate", &sia::ExtendedKalmanFilter::estimate,
           py::arg("observation"), py::arg("control"),
           py::return_value_policy::reference_internal)
      .def("predict", &sia::ExtendedKalmanFilter::predict, py::arg("control"),
           py::return_value_policy::reference_internal)
      .def("correct", &sia::ExtendedKalmanFilter::correct,
           py::arg("observation"), py::return_value_policy::reference_internal);

  py::class_<sia::ParticleFilter, sia::RecursiveBayesEstimator>(
      m, "ParticleFilter")
      .def(py::init<sia::MarkovProcess&, const sia::Particles&, double,
                    double>(),
           py::arg("system"), py::arg("particles"),
           py::arg("resample_threshold") = 1, py::arg("roughening_factor") = 0)
      .def("reset", &sia::ParticleFilter::reset, py::arg("particles"))
      .def("getBelief", &sia::ParticleFilter::getBelief)
      .def("estimate", &sia::ParticleFilter::estimate, py::arg("observation"),
           py::arg("control"), py::return_value_policy::reference_internal)
      .def("predict", &sia::ParticleFilter::predict, py::arg("control"),
           py::return_value_policy::reference_internal)
      .def("correct", &sia::ParticleFilter::correct, py::arg("observation"),
           py::return_value_policy::reference_internal);
}
