/// Copyright (c) 2018-2020, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include "python/py_math.h"

// Define module
void export_py_math(py::module& m_sup) {
  py::module m = m_sup;

  m.def("DEFAULT_SINGULAR_TOLERANCE",
        []() { return sia::DEFAULT_SINGULAR_TOLERANCE; });

  m.def("NUMERICAL_DERIVATIVE_STEP",
        []() { return sia::NUMERICAL_DERIVATIVE_STEP; });

  m.def("slice",
        static_cast<const Eigen::VectorXd (*)(const Eigen::VectorXd&,
                                              const std::vector<std::size_t>&)>(
            &sia::slice),
        py::arg("x"), py::arg("indices"));

  m.def("slice",
        static_cast<const Eigen::MatrixXd (*)(
            const Eigen::MatrixXd&, const std::vector<std::size_t>&,
            const std::vector<std::size_t>&)>(&sia::slice),
        py::arg("X"), py::arg("rows"), py::arg("cols"));

  m.def("rk4", &sia::rk4, py::arg("dynamical_system"), py::arg("x"),
        py::arg("u"), py::arg("dt"));
}
