include(CMakeFindDependencyMacro)

find_dependency(Eigen3 3.3.4 REQUIRED)
find_dependency(Glog 0.4.0 REQUIRED)
find_dependency(GTest REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/siaTargets.cmake")
