include(CMakeFindDependencyMacro)

find_dependency(Eigen3 3.3.4 REQUIRED)
find_dependency(Glog 0.4.0 REQUIRED)
find_dependency(GTest REQUIRED)
find_dependency(PythonInterp 3 REQUIRED)
find_dependency(PythonLibs 3 REQUIRED)
find_dependency(pybind11 2.6.1 REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/siaTargets.cmake")
