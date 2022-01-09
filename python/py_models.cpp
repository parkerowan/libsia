/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "python/py_models.h"

// Define module
void export_py_models(py::module& m_sup) {
  py::module m = m_sup;

  py::class_<sia::DynamicsModel, PyDynamicsModel>(m, "DynamicsModel")
      .def("dynamics", &sia::DynamicsModel::dynamics, py::arg("state"),
           py::arg("control"), py::return_value_policy::reference_internal);

  py::class_<sia::MeasurementModel, PyMeasurementModel>(m, "MeasurementModel")
      .def("measurement", &sia::MeasurementModel::measurement, py::arg("state"),
           py::return_value_policy::reference_internal);

  py::class_<sia::LinearizableDynamics, PyLinearizableDynamics,
             sia::DynamicsModel>(m, "LinearizableDynamics")
      .def("dynamics", &sia::LinearizableDynamics::dynamics, py::arg("state"),
           py::arg("control"), py::return_value_policy::reference_internal)
      .def("f", &sia::LinearizableDynamics::f, py::arg("state"),
           py::arg("control"))
      .def("Q", &sia::LinearizableDynamics::Q, py::arg("state"),
           py::arg("control"))
      .def("F", &sia::LinearizableDynamics::F, py::arg("state"),
           py::arg("control"))
      .def("G", &sia::LinearizableDynamics::G, py::arg("state"),
           py::arg("control"));

  py::class_<sia::LinearizableMeasurement, PyLinearizableMeasurement,
             sia::MeasurementModel>(m, "LinearizableMeasurement")
      .def("measurement", &sia::LinearizableMeasurement::measurement,
           py::arg("state"), py::return_value_policy::reference_internal)
      .def("h", &sia::LinearizableMeasurement::h, py::arg("state"))
      .def("R", &sia::LinearizableMeasurement::R, py::arg("state"))
      .def("H", &sia::LinearizableMeasurement::H, py::arg("state"));

  //   py::class_<sia::Trajectory>(m, "Trajectory")
  //       .def_readwrite("states", &sia::Trajectory::states)
  //       .def_readwrite("controls", &sia::Trajectory::controls)
  //       .def_readwrite("measurements", &sia::Trajectory::measurements);

  //   py::class_<sia::Trajectories>(m, "Trajectories")
  //       .def(py::init<const std::vector<sia::Trajectory>&>(),
  //       py::arg("data")) .def("data", &sia::Trajectories::data) .def("size",
  //       &sia::Trajectories::size) .def("states", &sia::Trajectories::states,
  //       py::arg("k")) .def("controls", &sia::Trajectories::controls,
  //       py::arg("k")) .def("measurements", &sia::Trajectories::measurements,
  //       py::arg("k"));

  //   m.def("simulate",
  //         static_cast<sia::Trajectory (*)(
  //             sia::DynamicsModel&, sia::MeasurementModel&, const
  //             Eigen::VectorXd&, const Eigen::MatrixXd&,
  //             bool)>(&sia::simulate),
  //         py::arg("dynamics"), py::arg("measurement"), py::arg("state"),
  //         py::arg("controls"), py::arg("sample") = true);

  //   m.def("simulate",
  //         static_cast<sia::Trajectories (*)(
  //             sia::DynamicsModel&, sia::MeasurementModel&,
  //             const std::vector<Eigen::VectorXd>&, const Eigen::MatrixXd&,
  //             bool)>( &sia::simulate),
  //         py::arg("dynamics"), py::arg("measurement"), py::arg("states"),
  //         py::arg("controls"), py::arg("sample") = true);

  //   m.def("simulate",
  //         static_cast<sia::Trajectories (*)(
  //             sia::DynamicsModel&, sia::MeasurementModel&,
  //             const std::vector<Eigen::VectorXd>&,
  //             const std::vector<Eigen::MatrixXd>&, bool)>(&sia::simulate),
  //         py::arg("dynamics"), py::arg("measurement"), py::arg("states"),
  //         py::arg("controls"), py::arg("sample") = true);

  py::class_<sia::NonlinearGaussianDynamics, sia::LinearizableDynamics,
             sia::DynamicsModel>(m, "NonlinearGaussianDynamics")
      .def(py::init<sia::DynamicsEquation, const Eigen::MatrixXd&>(),
           py::arg("dynamics"), py::arg("Q"))
      .def("dynamics", &sia::NonlinearGaussianDynamics::dynamics,
           py::arg("state"), py::arg("control"),
           py::return_value_policy::reference_internal)
      .def("f", &sia::NonlinearGaussianDynamics::f, py::arg("state"),
           py::arg("control"))
      .def("Q",
           static_cast<Eigen::MatrixXd (sia::NonlinearGaussianDynamics::*)(
               const Eigen::VectorXd&, const Eigen::VectorXd&)>(
               &sia::NonlinearGaussianDynamics::Q),
           py::arg("state"), py::arg("control"))
      .def("F", &sia::NonlinearGaussianDynamics::F, py::arg("state"),
           py::arg("control"))
      .def("G", &sia::NonlinearGaussianDynamics::G, py::arg("state"),
           py::arg("control"))
      .def("Q", static_cast<const Eigen::MatrixXd& (
                    sia::NonlinearGaussianDynamics::*)() const>(
                    &sia::NonlinearGaussianDynamics::Q))
      .def("setQ", &sia::NonlinearGaussianDynamics::setQ, py::arg("Q"));

  py::class_<sia::NonlinearGaussianMeasurement, sia::LinearizableMeasurement,
             sia::MeasurementModel>(m, "NonlinearGaussianMeasurement")
      .def(py::init<sia::MeasurementEquation, const Eigen::MatrixXd&>(),
           py::arg("measurement"), py::arg("R"))
      .def("measurement", &sia::NonlinearGaussianMeasurement::measurement,
           py::arg("state"), py::return_value_policy::reference_internal)
      .def("h", &sia::NonlinearGaussianMeasurement::h, py::arg("state"))
      .def("R",
           static_cast<Eigen::MatrixXd (sia::NonlinearGaussianMeasurement::*)(
               const Eigen::VectorXd&)>(&sia::NonlinearGaussianMeasurement::R),
           py::arg("state"))
      .def("H", &sia::NonlinearGaussianMeasurement::H, py::arg("state"))
      .def("R", static_cast<const Eigen::MatrixXd& (
                    sia::NonlinearGaussianMeasurement::*)() const>(
                    &sia::NonlinearGaussianMeasurement::R))
      .def("setR", &sia::NonlinearGaussianMeasurement::setR, py::arg("R"));

  py::class_<sia::NonlinearGaussianDynamicsCT, sia::NonlinearGaussianDynamics,
             sia::LinearizableDynamics, sia::DynamicsModel>(
      m, "NonlinearGaussianDynamicsCT")
      .def(py::init<sia::DynamicsEquation, const Eigen::MatrixXd&, double>(),
           py::arg("dynamics"), py::arg("Qpsd"), py::arg("dt"))
      .def("dynamics", &sia::NonlinearGaussianDynamicsCT::dynamics,
           py::arg("state"), py::arg("control"),
           py::return_value_policy::reference_internal)
      .def("f", &sia::NonlinearGaussianDynamicsCT::f, py::arg("state"),
           py::arg("control"))
      .def("Q",
           static_cast<Eigen::MatrixXd (sia::NonlinearGaussianDynamicsCT::*)(
               const Eigen::VectorXd&, const Eigen::VectorXd&)>(
               &sia::NonlinearGaussianDynamicsCT::Q),
           py::arg("state"), py::arg("control"))
      .def("F", &sia::NonlinearGaussianDynamicsCT::F, py::arg("state"),
           py::arg("control"))
      .def("G", &sia::NonlinearGaussianDynamicsCT::G, py::arg("state"),
           py::arg("control"))
      .def("Q", static_cast<const Eigen::MatrixXd& (
                    sia::NonlinearGaussianDynamicsCT::*)() const>(
                    &sia::NonlinearGaussianDynamicsCT::Q))
      .def("setQ", &sia::NonlinearGaussianDynamicsCT::setQ, py::arg("Q"))
      .def("setQpsd", &sia::NonlinearGaussianDynamicsCT::setQpsd,
           py::arg("Qpsd"))
      .def("getTimeStep", &sia::NonlinearGaussianDynamicsCT::getTimeStep)
      .def("setTimeStep", &sia::NonlinearGaussianDynamicsCT::setTimeStep,
           py::arg("dt"));

  py::class_<sia::NonlinearGaussianMeasurementCT,
             sia::NonlinearGaussianMeasurement, sia::LinearizableMeasurement,
             sia::MeasurementModel>(m, "NonlinearGaussianMeasurementCT")
      .def(py::init<sia::MeasurementEquation, const Eigen::MatrixXd&, double>(),
           py::arg("measurement"), py::arg("Rpsd"), py::arg("dt"))
      .def("measurement", &sia::NonlinearGaussianMeasurementCT::measurement,
           py::arg("state"), py::return_value_policy::reference_internal)
      .def("h", &sia::NonlinearGaussianMeasurementCT::h, py::arg("state"))
      .def(
          "R",
          static_cast<Eigen::MatrixXd (sia::NonlinearGaussianMeasurementCT::*)(
              const Eigen::VectorXd&)>(&sia::NonlinearGaussianMeasurementCT::R),
          py::arg("state"))
      .def("H", &sia::NonlinearGaussianMeasurementCT::H, py::arg("state"))
      .def("R", static_cast<const Eigen::MatrixXd& (
                    sia::NonlinearGaussianMeasurementCT::*)() const>(
                    &sia::NonlinearGaussianMeasurementCT::R))
      .def("setR", &sia::NonlinearGaussianMeasurementCT::setR, py::arg("R"))
      .def("setRpsd", &sia::NonlinearGaussianMeasurementCT::setRpsd,
           py::arg("Rpsd"))
      .def("getTimeStep", &sia::NonlinearGaussianMeasurementCT::getTimeStep)
      .def("setTimeStep", &sia::NonlinearGaussianMeasurementCT::setTimeStep,
           py::arg("dt"));

  py::class_<sia::LinearGaussianDynamics, sia::LinearizableDynamics,
             sia::DynamicsModel>(m, "LinearGaussianDynamics")
      .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&,
                    const Eigen::MatrixXd&>(),
           py::arg("F"), py::arg("G"), py::arg("Q"))
      .def("dynamics", &sia::LinearGaussianDynamics::dynamics, py::arg("state"),
           py::arg("control"), py::return_value_policy::reference_internal)
      .def("f", &sia::LinearGaussianDynamics::f, py::arg("state"),
           py::arg("control"))
      .def("Q",
           static_cast<Eigen::MatrixXd (sia::LinearGaussianDynamics::*)(
               const Eigen::VectorXd&, const Eigen::VectorXd&)>(
               &sia::LinearGaussianDynamics::Q),
           py::arg("state"), py::arg("control"))
      .def("F",
           static_cast<Eigen::MatrixXd (sia::LinearGaussianDynamics::*)(
               const Eigen::VectorXd&, const Eigen::VectorXd&)>(
               &sia::LinearGaussianDynamics::F),
           py::arg("state"), py::arg("control"))
      .def("G",
           static_cast<Eigen::MatrixXd (sia::LinearGaussianDynamics::*)(
               const Eigen::VectorXd&, const Eigen::VectorXd&)>(
               &sia::LinearGaussianDynamics::G),
           py::arg("state"), py::arg("control"))
      .def("Q",
           static_cast<const Eigen::MatrixXd& (sia::LinearGaussianDynamics::*)()
                           const>(&sia::LinearGaussianDynamics::Q))
      .def("F",
           static_cast<const Eigen::MatrixXd& (sia::LinearGaussianDynamics::*)()
                           const>(&sia::LinearGaussianDynamics::F))
      .def("G",
           static_cast<const Eigen::MatrixXd& (sia::LinearGaussianDynamics::*)()
                           const>(&sia::LinearGaussianDynamics::G))
      .def("setQ", &sia::LinearGaussianDynamics::setQ, py::arg("Q"))
      .def("setF", &sia::LinearGaussianDynamics::setF, py::arg("F"))
      .def("setG", &sia::LinearGaussianDynamics::setG, py::arg("G"));

  py::class_<sia::LinearGaussianMeasurement, sia::LinearizableMeasurement,
             sia::MeasurementModel>(m, "LinearGaussianMeasurement")
      .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&>(),
           py::arg("H"), py::arg("R"))
      .def("measurement", &sia::LinearGaussianMeasurement::measurement,
           py::arg("state"), py::return_value_policy::reference_internal)
      .def("h", &sia::LinearGaussianMeasurement::h, py::arg("state"))
      .def("R",
           static_cast<Eigen::MatrixXd (sia::LinearGaussianMeasurement::*)(
               const Eigen::VectorXd&)>(&sia::LinearGaussianMeasurement::R),
           py::arg("state"))
      .def("H",
           static_cast<Eigen::MatrixXd (sia::LinearGaussianMeasurement::*)(
               const Eigen::VectorXd&)>(&sia::LinearGaussianMeasurement::H),
           py::arg("state"))
      .def("R", static_cast<const Eigen::MatrixXd& (
                    sia::LinearGaussianMeasurement::*)() const>(
                    &sia::LinearGaussianMeasurement::R))
      .def("H", static_cast<const Eigen::MatrixXd& (
                    sia::LinearGaussianMeasurement::*)() const>(
                    &sia::LinearGaussianMeasurement::H))
      .def("setR", &sia::LinearGaussianMeasurement::setR, py::arg("R"))
      .def("setH", &sia::LinearGaussianMeasurement::setH, py::arg("H"));

  py::class_<sia::LinearGaussianDynamicsCT, sia::LinearGaussianDynamics,
             sia::LinearizableDynamics, sia::DynamicsModel>
      lgdct(m, "LinearGaussianDynamicsCT");

  py::enum_<sia::LinearGaussianDynamicsCT::Type>(lgdct, "Type")
      .value("FORWARD_EULER", sia::LinearGaussianDynamicsCT::FORWARD_EULER)
      .value("BACKWARD_EULER", sia::LinearGaussianDynamicsCT::BACKWARD_EULER)
      .export_values();

  lgdct
      .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&,
                    const Eigen::MatrixXd&, double,
                    sia::LinearGaussianDynamicsCT::Type>(),
           py::arg("A"), py::arg("B"), py::arg("Qpsd"), py::arg("dt"),
           py::arg("type") = sia::LinearGaussianDynamicsCT::BACKWARD_EULER)
      .def("dynamics", &sia::LinearGaussianDynamicsCT::dynamics,
           py::arg("state"), py::arg("control"),
           py::return_value_policy::reference_internal)
      .def("f", &sia::LinearGaussianDynamicsCT::f, py::arg("state"),
           py::arg("control"))
      .def("Q",
           static_cast<Eigen::MatrixXd (sia::LinearGaussianDynamicsCT::*)(
               const Eigen::VectorXd&, const Eigen::VectorXd&)>(
               &sia::LinearGaussianDynamicsCT::Q),
           py::arg("state"), py::arg("control"))
      .def("F",
           static_cast<Eigen::MatrixXd (sia::LinearGaussianDynamicsCT::*)(
               const Eigen::VectorXd&, const Eigen::VectorXd&)>(
               &sia::LinearGaussianDynamicsCT::F),
           py::arg("state"), py::arg("control"))
      .def("G",
           static_cast<Eigen::MatrixXd (sia::LinearGaussianDynamicsCT::*)(
               const Eigen::VectorXd&, const Eigen::VectorXd&)>(
               &sia::LinearGaussianDynamicsCT::G),
           py::arg("state"), py::arg("control"))
      .def("Q", static_cast<const Eigen::MatrixXd& (
                    sia::LinearGaussianDynamicsCT::*)() const>(
                    &sia::LinearGaussianDynamicsCT::Q))
      .def("F", static_cast<const Eigen::MatrixXd& (
                    sia::LinearGaussianDynamicsCT::*)() const>(
                    &sia::LinearGaussianDynamicsCT::F))
      .def("G", static_cast<const Eigen::MatrixXd& (
                    sia::LinearGaussianDynamicsCT::*)() const>(
                    &sia::LinearGaussianDynamicsCT::G))
      .def("setQ", &sia::LinearGaussianDynamicsCT::setQ, py::arg("Q"))
      .def("setF", &sia::LinearGaussianDynamicsCT::setF, py::arg("F"))
      .def("setG", &sia::LinearGaussianDynamicsCT::setG, py::arg("G"))
      .def("A", &sia::LinearGaussianDynamicsCT::A)
      .def("B", &sia::LinearGaussianDynamicsCT::B)
      .def("setA", &sia::LinearGaussianDynamicsCT::setA, py::arg("A"))
      .def("setB", &sia::LinearGaussianDynamicsCT::setB, py::arg("b"))
      .def("setQpsd", &sia::LinearGaussianDynamicsCT::setQpsd, py::arg("Qpsd"))
      .def("getType", &sia::LinearGaussianDynamicsCT::getType)
      .def("setType", &sia::LinearGaussianDynamicsCT::setType, py::arg("type"))
      .def("getTimeStep", &sia::LinearGaussianDynamicsCT::getTimeStep)
      .def("setTimeStep", &sia::LinearGaussianDynamicsCT::setTimeStep,
           py::arg("dt"));

  py::class_<sia::LinearGaussianMeasurementCT, sia::LinearGaussianMeasurement,
             sia::LinearizableMeasurement, sia::MeasurementModel>(
      m, "LinearGaussianMeasurementCT")
      .def(py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&, double>(),
           py::arg("H"), py::arg("Rpsd"), py::arg("dt"))
      .def("measurement", &sia::LinearGaussianMeasurementCT::measurement,
           py::arg("state"), py::return_value_policy::reference_internal)
      .def("h", &sia::LinearGaussianMeasurementCT::h, py::arg("state"))
      .def("R",
           static_cast<Eigen::MatrixXd (sia::LinearGaussianMeasurementCT::*)(
               const Eigen::VectorXd&)>(&sia::LinearGaussianMeasurementCT::R),
           py::arg("state"))
      .def("H",
           static_cast<Eigen::MatrixXd (sia::LinearGaussianMeasurementCT::*)(
               const Eigen::VectorXd&)>(&sia::LinearGaussianMeasurementCT::H),
           py::arg("state"))
      .def("R", static_cast<const Eigen::MatrixXd& (
                    sia::LinearGaussianMeasurementCT::*)() const>(
                    &sia::LinearGaussianMeasurementCT::R))
      .def("H", static_cast<const Eigen::MatrixXd& (
                    sia::LinearGaussianMeasurementCT::*)() const>(
                    &sia::LinearGaussianMeasurementCT::H))
      .def("setR", &sia::LinearGaussianMeasurementCT::setR, py::arg("R"))
      .def("setH", &sia::LinearGaussianMeasurementCT::setH, py::arg("H"))
      .def("setRpsd", &sia::LinearGaussianMeasurementCT::setRpsd,
           py::arg("Rpsd"))
      .def("getTimeStep", &sia::LinearGaussianMeasurementCT::getTimeStep)
      .def("setTimeStep", &sia::LinearGaussianMeasurementCT::setTimeStep,
           py::arg("dt"));
}
