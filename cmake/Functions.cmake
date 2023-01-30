# Copyright (c) 2018-2023, Parker Owan.  All rights reserved.
# Licensed under BSD-3 Clause, https://opensource.org/licenses/BSD-3-Clause

# Function For building a C++ library target
#
# TARGET         name of the target library
# DEPENDENCIES   1st party depedencies that need to built first
# LIBRARIES      3rd party libraries to link
# HEADERS        target header files
# SOURCES        target source files
# HEADER_DEST    public header destination relative to install path
# 
function(FUNCTIONS_CREATE_CPP_SHARED_LIB)
  set(options NONE)
  set(oneValueArgs TARGET HEADER_DEST)
  set(multiValueArgs DEPENDENCIES LIBRARIES HEADERS SOURCES)
  cmake_parse_arguments(Functions "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_library(${Functions_TARGET} SHARED
    ${Functions_HEADERS}
    ${Functions_SOURCES}
  )

  if(Functions_DEPENDENCIES)
    add_dependencies(${Functions_TARGET} ${Functions_DEPENDENCIES})
  endif()

  target_link_libraries(${Functions_TARGET} PRIVATE
    ${Functions_DEPENDENCIES}
    ${Functions_LIBRARIES}
  )

  target_include_directories(${Functions_TARGET} PUBLIC 
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/${Functions_TARGET}>
  )

  set_target_properties(${Functions_TARGET} PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib
    PUBLIC_HEADER "${Functions_HEADERS}"
  )

  install(TARGETS ${Functions_TARGET}
    EXPORT "${Functions_TARGET}Targets"
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${Functions_TARGET}
  )

  configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/${Functions_TARGET}Config.cmake.in
    ${CMAKE_BINARY_DIR}/cmake/${Functions_TARGET}Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${Functions_TARGET}
  )

  install(EXPORT ${Functions_TARGET}Targets
    FILE ${Functions_TARGET}Targets.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${Functions_TARGET}
  )

  install(FILES
    ${CMAKE_BINARY_DIR}/cmake/${Functions_TARGET}Config.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${Functions_TARGET}
  )

  write_basic_package_version_file(
    "${PROJECT_SOURCE_DIR}/lib/cmake/${Functions_TARGET}ConfigVersion.cmake"
    VERSION ${PROJ_VERSION}
    COMPATIBILITY SameMajorVersion
  )

  install(
    FILES "${PROJECT_SOURCE_DIR}/lib/cmake/${Functions_TARGET}ConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${Functions_TARGET}
  )

  foreach(file ${Functions_HEADERS})
    get_filename_component(dir ${file} DIRECTORY)
    install(FILES ${file} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${Functions_HEADER_DEST}/${dir})
  endforeach()
endfunction(FUNCTIONS_CREATE_CPP_SHARED_LIB)

# Function For building a pybind11 wrapper
#
# MODULE         name of the python wrapper module
# DEPENDENCIES   1st party depedencies that need to built first
# LIBRARIES      3rd party libraries to link
# SOURCES        python wrapper source files
# 
function(FUNCTIONS_CREATE_PYBIND11_MODULE)
  set(options NONE)
  set(oneValueArgs MODULE)
  set(multiValueArgs DEPENDENCIES LIBRARIES SOURCES)
  cmake_parse_arguments(Functions "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

  pybind11_add_module(${Functions_MODULE}
    MODULE ${Functions_SOURCES}
  )

  if(Functions_DEPENDENCIES)
    add_dependencies(${Functions_MODULE} ${Functions_DEPENDENCIES})
  endif()

  target_link_libraries(${Functions_MODULE} PRIVATE
    ${Functions_DEPENDENCIES}
    ${Functions_LIBRARIES}
  )

  set_target_properties(${Functions_MODULE} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/lib
  )

  # Find python site-packages
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
      from distutils import sysconfig as sc
      print(sc.get_python_lib(prefix='', plat_specific=True))"
    OUTPUT_VARIABLE PYTHON_SITE
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  
  install(TARGETS ${Functions_MODULE}
    EXPORT ${Functions_MODULE}
    LIBRARY DESTINATION "${PYTHON_SITE}" COMPONENT shlib
    ARCHIVE DESTINATION "${PYTHON_SITE}" COMPONENT lib
    COMPONENT dev
  )
endfunction(FUNCTIONS_CREATE_PYBIND11_MODULE)

# Function For building a C++ GTest target
#
# TARGET         name of the test target
# DEPENDENCIES   1st party depedencies that need to built first
# LIBRARIES      3rd party libraries to link
# SOURCES        test source files
# 
function(FUNCTIONS_CREATE_CPP_TEST)
  set(options NONE)
  set(oneValueArgs TARGET)
  set(multiValueArgs DEPENDENCIES LIBRARIES SOURCES)
  cmake_parse_arguments(Functions "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  
  add_executable(${Functions_TARGET} ${Functions_SOURCES})
  
  if(Functions_DEPENDENCIES)
    add_dependencies(${Functions_TARGET} ${Functions_DEPENDENCIES})
  endif()
  
  target_link_libraries(${Functions_TARGET} PRIVATE
    ${Functions_DEPENDENCIES}
    ${Functions_LIBRARIES}
    ${GTEST_LIBRARIES}
    ${GTEST_MAIN_LIBRARIES}
    pthread
  )
  
  set_target_properties(${Functions_TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
  )
  
  add_test(
    NAME ${Functions_TARGET}
    COMMAND ${Functions_TARGET}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
  )
endfunction(FUNCTIONS_CREATE_CPP_TEST)

# Function For building a pytest target
#
# TARGET             name of the pytest target
# WORKING_DIRECTORY  directory where the test is located
# 
function(FUNCTIONS_CREATE_PYTHON_TEST)
  set(options NONE)
  set(oneValueArgs TARGET WORKING_DIRECTORY)
  set(multiValueArgs)
  cmake_parse_arguments(Functions "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_test(
    NAME ${Functions_TARGET}
    COMMAND python3 -m pytest
    WORKING_DIRECTORY ${Functions_WORKING_DIRECTORY}
  )
endfunction(FUNCTIONS_CREATE_PYTHON_TEST)

# Function For building a C++ exectuable
#
# TARGET         name of the executable
# DEPENDENCIES   1st party depedencies that need to built first
# LIBRARIES      3rd party libraries to link
# SOURCES        target source files
# 
function(FUNCTIONS_CREATE_CPP_EXE)
  set(options NONE)
  set(oneValueArgs TARGET)
  set(multiValueArgs DEPENDENCIES LIBRARIES SOURCES)
  cmake_parse_arguments(Functions "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
  )
  
  add_executable(${Functions_TARGET} ${Functions_SOURCES})

  add_dependencies(${Functions_TARGET}
    ${Functions_DEPENDENCIES}
  )

  target_link_libraries(${Functions_TARGET} PRIVATE
    ${Functions_DEPENDENCIES}
    ${Functions_LIBRARIES}
  )
  
  set_target_properties(${Functions_TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin
  )
endfunction(FUNCTIONS_CREATE_CPP_EXE)