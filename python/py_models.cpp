/// Copyright (c) 2018-2021, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "python/py_models.h"

// Define module
void export_py_models(py::module& m_sup) {
  py::module m = m_sup;

  py::class_<sia::MarkovProcess, PyMarkovProcess>(m, "MarkovProcess")
      .def("dynamics", &sia::MarkovProcess::dynamics, py::arg("state"),
           py::arg("control"), py::return_value_policy::reference_internal)
      .def("measurement", &sia::MarkovProcess::measurement, py::arg("state"),
           py::return_value_policy::reference_internal);

  py::class_<sia::Trajectory>(m, "Trajectory")
      .def_readwrite("states", &sia::Trajectory::states)
      .def_readwrite("controls", &sia::Trajectory::controls)
      .def_readwrite("measurements", &sia::Trajectory::measurements);

  py::class_<sia::Trajectories>(m, "Trajectories")
      .def(py::init<const std::vector<sia::Trajectory>&>(), py::arg("data"))
      .def("data", &sia::Trajectories::data)
      .def("size", &sia::Trajectories::size)
      .def("states", &sia::Trajectories::states, py::arg("k"))
      .def("controls", &sia::Trajectories::controls, py::arg("k"))
      .def("measurements", &sia::Trajectories::measurements, py::arg("k"));

  m.def("simulate",
        static_cast<sia::Trajectory (*)(
            sia::MarkovProcess&, const Eigen::VectorXd&, const Eigen::MatrixXd&,
            bool)>(&sia::simulate),
        py::arg("system"), py::arg("state"), py::arg("controls"),
        py::arg("sample") = true);

  m.def("simulate",
        static_cast<sia::Trajectories (*)(
            sia::MarkovProcess&, const std::vector<Eigen::VectorXd>&,
            const Eigen::MatrixXd&, bool)>(&sia::simulate),
        py::arg("system"), py::arg("states"), py::arg("controls"),
        py::arg("sample") = true);

  m.def("simulate",
        static_cast<sia::Trajectories (*)(
            sia::MarkovProcess&, const std::vector<Eigen::VectorXd>&,
            const std::vector<Eigen::MatrixXd>&, bool)>(&sia::simulate),
        py::arg("system"), py::arg("states"), py::arg("controls"),
        py::arg("sample") = true);

  py::class_<sia::NonlinearGaussian, sia::MarkovProcess>(m, "NonlinearGaussian")
      .def(py::init<sia::DynamicsEquation, sia::MeasurementEquation,
                    const Eigen::MatrixXd&, const Eigen::MatrixXd&,
                    const Eigen::MatrixXd&>(),
           py::arg("dynamics"), py::arg("measurement"), py::arg("C"),
           py::arg("Q"), py::arg("R"))
      .def("dynamics", &sia::NonlinearGaussian::dynamics, py::arg("state"),
           py::arg("control"), py::return_value_policy::reference_internal)
      .def("measurement", &sia::NonlinearGaussian::measurement,
           py::arg("state"), py::return_value_policy::reference_internal)
      .def("f", &sia::NonlinearGaussian::f, py::arg("state"),
           py::arg("control"))
      .def("F", &sia::NonlinearGaussian::F, py::arg("state"),
           py::arg("control"))
      .def("G", &sia::NonlinearGaussian::G, py::arg("state"),
           py::arg("control"))
      .def("h", &sia::NonlinearGaussian::h, py::arg("state"))
      .def("H", &sia::NonlinearGaussian::H, py::arg("state"))
      .def("C", &sia::NonlinearGaussian::C)
      .def("setC", &sia::NonlinearGaussian::setC, py::arg("C"))
      .def("Q", &sia::NonlinearGaussian::Q)
      .def("setQ", &sia::NonlinearGaussian::setQ, py::arg("Q"))
      .def("R", &sia::NonlinearGaussian::R)
      .def("setR", &sia::NonlinearGaussian::setR, py::arg("R"));

  py::class_<sia::NonlinearGaussianCT, sia::NonlinearGaussian,
             sia::MarkovProcess>(m, "NonlinearGaussianCT")
      .def(py::init<sia::DynamicsEquation, sia::MeasurementEquation,
                    const Eigen::MatrixXd&, const Eigen::MatrixXd&,
                    const Eigen::MatrixXd&, double>(),
           py::arg("dynamics"), py::arg("measurement"), py::arg("C"),
           py::arg("Q"), py::arg("R"), py::arg("dt"))
      .def("dynamics", &sia::NonlinearGaussianCT::dynamics, py::arg("state"),
           py::arg("control"), py::return_value_policy::reference_internal)
      .def("measurement", &sia::NonlinearGaussianCT::measurement,
           py::arg("state"), py::return_value_policy::reference_internal)
      .def("f", &sia::NonlinearGaussianCT::f, py::arg("state"),
           py::arg("control"))
      .def("F", &sia::NonlinearGaussianCT::F, py::arg("state"),
           py::arg("control"))
      .def("G", &sia::NonlinearGaussianCT::G, py::arg("state"),
           py::arg("control"))
      .def("h", &sia::NonlinearGaussianCT::h, py::arg("state"))
      .def("H", &sia::NonlinearGaussianCT::H, py::arg("state"))
      .def("C", &sia::NonlinearGaussianCT::C)
      .def("setC", &sia::NonlinearGaussianCT::setC, py::arg("C"))
      .def("Q", &sia::NonlinearGaussianCT::Q)
      .def("setQ", &sia::NonlinearGaussianCT::setQ, py::arg("Q"))
      .def("R", &sia::NonlinearGaussianCT::R)
      .def("setR", &sia::NonlinearGaussianCT::setR, py::arg("R"))
      .def("getTimeStep", &sia::NonlinearGaussianCT::getTimeStep)
      .def("setTimeStep", &sia::NonlinearGaussianCT::setTimeStep,
           py::arg("dt"));

  py::class_<sia::LinearGaussian, sia::NonlinearGaussian, sia::MarkovProcess>(
      m, "LinearGaussian")
      .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&,
                    const Eigen::MatrixXd&, const Eigen::MatrixXd&,
                    const Eigen::MatrixXd&, const Eigen::MatrixXd&>(),
           py::arg("F"), py::arg("G"), py::arg("C"), py::arg("H"), py::arg("Q"),
           py::arg("R"))
      .def("dynamics", &sia::LinearGaussian::dynamics, py::arg("state"),
           py::arg("control"), py::return_value_policy::reference_internal)
      .def("measurement", &sia::LinearGaussian::measurement, py::arg("state"),
           py::return_value_policy::reference_internal)
      .def("f", &sia::LinearGaussian::f, py::arg("state"), py::arg("control"))
      .def("F",
           static_cast<const Eigen::MatrixXd (sia::LinearGaussian::*)(
               const Eigen::VectorXd&, const Eigen::VectorXd&) const>(
               &sia::LinearGaussian::F),
           py::arg("state"), py::arg("control"))
      .def("G",
           static_cast<const Eigen::MatrixXd (sia::LinearGaussian::*)(
               const Eigen::VectorXd&, const Eigen::VectorXd&) const>(
               &sia::LinearGaussian::G),
           py::arg("state"), py::arg("control"))
      .def("h", &sia::LinearGaussian::h, py::arg("state"))
      .def("H",
           static_cast<const Eigen::MatrixXd (sia::LinearGaussian::*)(
               const Eigen::VectorXd&) const>(&sia::LinearGaussian::H),
           py::arg("state"))
      .def("F",
           static_cast<const Eigen::MatrixXd& (sia::LinearGaussian::*)(void)
                           const>(&sia::LinearGaussian::F))
      .def("setF", &sia::LinearGaussian::setF, py::arg("F"))
      .def("G",
           static_cast<const Eigen::MatrixXd& (sia::LinearGaussian::*)(void)
                           const>(&sia::LinearGaussian::G))
      .def("setG", &sia::LinearGaussian::setG, py::arg("G"))
      .def("H",
           static_cast<const Eigen::MatrixXd& (sia::LinearGaussian::*)(void)
                           const>(&sia::LinearGaussian::H))
      .def("setH", &sia::LinearGaussian::setH, py::arg("H"))
      .def("C", &sia::LinearGaussian::C)
      .def("setC", &sia::LinearGaussian::setC, py::arg("C"))
      .def("Q", &sia::LinearGaussian::Q)
      .def("setQ", &sia::LinearGaussian::setQ, py::arg("Q"))
      .def("R", &sia::LinearGaussian::R)
      .def("setR", &sia::LinearGaussian::setR, py::arg("R"));

  py::class_<sia::LinearGaussianCT, sia::LinearGaussian, sia::NonlinearGaussian,
             sia::MarkovProcess>
      lgct(m, "LinearGaussianCT");

  py::enum_<sia::LinearGaussianCT::Type>(lgct, "Type")
      .value("FORWARD_EULER", sia::LinearGaussianCT::FORWARD_EULER)
      .value("BACKWARD_EULER", sia::LinearGaussianCT::BACKWARD_EULER)
      .export_values();

  lgct.def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&,
                    const Eigen::MatrixXd&, const Eigen::MatrixXd&,
                    const Eigen::MatrixXd&, const Eigen::MatrixXd&, double,
                    sia::LinearGaussianCT::Type>(),
           py::arg("A"), py::arg("B"), py::arg("C"), py::arg("H"), py::arg("Q"),
           py::arg("R"), py::arg("dt"),
           py::arg("type") = sia::LinearGaussianCT::BACKWARD_EULER)
      .def("dynamics", &sia::LinearGaussianCT::dynamics, py::arg("state"),
           py::arg("control"), py::return_value_policy::reference_internal)
      .def("measurement", &sia::LinearGaussianCT::measurement, py::arg("state"),
           py::return_value_policy::reference_internal)
      .def("f", &sia::LinearGaussianCT::f, py::arg("state"), py::arg("control"))
      .def("F",
           static_cast<const Eigen::MatrixXd (sia::LinearGaussianCT::*)(
               const Eigen::VectorXd&, const Eigen::VectorXd&) const>(
               &sia::LinearGaussianCT::F),
           py::arg("state"), py::arg("control"))
      .def("h", &sia::LinearGaussianCT::h, py::arg("state"))
      .def("H",
           static_cast<const Eigen::MatrixXd (sia::LinearGaussianCT::*)(
               const Eigen::VectorXd&) const>(&sia::LinearGaussianCT::H),
           py::arg("state"))
      .def("F",
           static_cast<const Eigen::MatrixXd& (sia::LinearGaussianCT::*)(void)
                           const>(&sia::LinearGaussianCT::F))
      .def("setF", &sia::LinearGaussianCT::setF, py::arg("F"))
      .def("G",
           static_cast<const Eigen::MatrixXd& (sia::LinearGaussianCT::*)(void)
                           const>(&sia::LinearGaussianCT::G))
      .def("setG", &sia::LinearGaussianCT::setG, py::arg("G"))
      .def("H",
           static_cast<const Eigen::MatrixXd& (sia::LinearGaussianCT::*)(void)
                           const>(&sia::LinearGaussianCT::H))
      .def("setH", &sia::LinearGaussianCT::setH, py::arg("H"))
      .def("C", &sia::LinearGaussianCT::C)
      .def("setC", &sia::LinearGaussianCT::setC, py::arg("C"))
      .def("Q", &sia::LinearGaussianCT::Q)
      .def("setQ", &sia::LinearGaussianCT::setQ, py::arg("Q"))
      .def("R", &sia::LinearGaussianCT::R)
      .def("setR", &sia::LinearGaussianCT::setR, py::arg("R"))
      .def("A", &sia::LinearGaussianCT::A)
      .def("setA", &sia::LinearGaussianCT::setA, py::arg("A"))
      .def("B", &sia::LinearGaussianCT::B)
      .def("setB", &sia::LinearGaussianCT::setB, py::arg("B"))
      .def("getType", &sia::LinearGaussianCT::getType)
      .def("setType", &sia::LinearGaussianCT::setType, py::arg("type"))
      .def("getTimeStep", &sia::LinearGaussianCT::getTimeStep)
      .def("setTimeStep", &sia::LinearGaussianCT::setTimeStep, py::arg("dt"));
}
