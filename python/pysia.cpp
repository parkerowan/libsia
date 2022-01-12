/// Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
/// Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

#include <pybind11/pybind11.h>

#include "python/py_belief.h"
#include "python/py_controllers.h"
#include "python/py_estimators.h"
#include "python/py_math.h"
#include "python/py_models.h"
#include "python/py_optimizers.h"

namespace py = pybind11;

PYBIND11_MODULE(pysia, m) {
  m.doc() = "Model-based Reinforcement Learning";

  export_py_belief(m);
  export_py_controllers(m);
  export_py_estimators(m);
  export_py_math(m);
  export_py_models(m);
  export_py_optimizers(m);
}
