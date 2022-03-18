cmake_minimum_required (VERSION 3.14)
project (Sophus VERSION 1.1.0)

# Create examples make targets using ctest
option (BUILD_EXAMPLES "Build examples" ON)

include (CMakePackageConfigHelpers)
include (CTest)
include (GNUInstallDirs)

find_package (Ceres 2.1.0)
find_package (Eigen3 3.3.0 REQUIRED)
find_package (fmt 5.3.0 REQUIRED)

# Define interface library target
add_library (sophus INTERFACE)
add_library (Sophus::Sophus ALIAS sophus)

target_sources (sophus PRIVATE
  sophus/average.hpp
  sophus/common.hpp
  sophus/geometry.hpp
  sophus/interpolate.hpp
  sophus/interpolate_details.hpp
  sophus/num_diff.hpp
  sophus/rotation_matrix.hpp
  sophus/rxso2.hpp
  sophus/rxso3.hpp
  sophus/se2.hpp
  sophus/se3.hpp
  sophus/sim2.hpp
  sophus/sim3.hpp
  sophus/sim_details.hpp
  sophus/so2.hpp
  sophus/so3.hpp
  sophus/spline.hpp
  sophus/types.hpp
  sophus/velocities.hpp
)

if (MSVC)
  # Define common math constants if we compile with MSVC
  target_compile_definitions (sophus INTERFACE _USE_MATH_DEFINES)
endif (MSVC)

# Add Eigen interface dependency, depending on available cmake info
target_link_libraries (sophus INTERFACE Eigen3::Eigen)
set (Eigen3_DEPENDENCY "find_dependency (Eigen3 ${Eigen3_VERSION})")

target_link_libraries (sophus INTERFACE fmt::fmt)
set (fmt_DEPENDENCY "find_dependency (fmt ${fmt_VERSION})")

# Associate target with include directory
target_include_directories (sophus INTERFACE
  $<BUILD_INTERFACE:${Sophus_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Declare all used C++14 features
target_compile_features (sophus INTERFACE
  cxx_auto_type
  cxx_decltype
  cxx_nullptr
  cxx_right_angle_brackets
  cxx_variadic_macros
  cxx_variadic_templates
)

# Create 'test' make target using ctest
if (BUILD_TESTING)
  add_subdirectory(test)
endif ()

if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif ()

# Export package for use from the build tree
set(SOPHUS_CMAKE_EXPORT_DIR ${CMAKE_INSTALL_DATADIR}/sophus/cmake)

set_target_properties(sophus PROPERTIES EXPORT_NAME Sophus)

install (TARGETS sophus EXPORT SophusTargets)
install (EXPORT SophusTargets
  NAMESPACE Sophus::
  DESTINATION ${SOPHUS_CMAKE_EXPORT_DIR}
)

export (TARGETS sophus NAMESPACE Sophus:: FILE SophusTargets.cmake)
export (PACKAGE Sophus)

configure_package_config_file (
  SophusConfig.cmake.in
  ${Sophus_BINARY_DIR}/SophusConfig.cmake
  INSTALL_DESTINATION ${SOPHUS_CMAKE_EXPORT_DIR}
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

# Write version to file
write_basic_package_version_file (
  SophusConfigVersion.cmake
  VERSION ${Sophus_VERSION}
  COMPATIBILITY SameMajorVersion
  ARCH_INDEPENDENT
)

# Install cmake targets
install (
  FILES ${Sophus_BINARY_DIR}/SophusConfig.cmake
        ${Sophus_BINARY_DIR}/SophusConfigVersion.cmake
  DESTINATION ${SOPHUS_CMAKE_EXPORT_DIR}
)

# Install header files
install (
  DIRECTORY ${Sophus_SOURCE_DIR}/sophus
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN *.hpp
)
