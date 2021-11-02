# Copyright (c) 2018-2022, Parker Owan.  All rights reserved.
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

  set_target_properties(${Functions_TARGET} PROPERTIES
    LINKER_LANGUAGE CXX
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
  )

  install(TARGETS ${Functions_TARGET}
    EXPORT "${Functions_TARGET}Targets"
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}" COMPONENT shlib
    ARCHIVE DESTINATION "${INSTALL_LIB_DIR}" COMPONENT lib
    COMPONENT dev
  )

  foreach(file ${Functions_HEADERS})
    get_filename_component(dir ${file} DIRECTORY)
    install(FILES ${file} DESTINATION ${INSTALL_INCLUDE_DIR}/${Functions_HEADER_DEST}/${dir})
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
    glog
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