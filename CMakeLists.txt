# Copyright 2017-2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>
#
# The CMake package configuration of Sophus, which exports the Sophus::Sophus
# target for find_package, is based on the CMake files of Sergiu Deitsch. They
# were adapted and merged upstream in
# https://github.com/strasdat/Sophus/pull/120, and
# https://github.com/strasdat/Sophus/pull/125 states that the change borrows
# extensively from his fork.

cmake_minimum_required (VERSION 3.28...4.4)
project (Sophus
  VERSION 2.0.0
  DESCRIPTION "C++ implementation of Lie Groups using Eigen"
  HOMEPAGE_URL https://github.com/sergiud/Sophus
  LANGUAGES CXX
)

# Tests and examples are built only if Sophus is the top-level project, so that
# projects consuming Sophus through add_subdirectory or FetchContent neither
# build them nor inherit the corresponding cache entries.
if (PROJECT_IS_TOP_LEVEL)
  include (CTest)
  option (BUILD_EXAMPLES "Build examples" ON)
endif (PROJECT_IS_TOP_LEVEL)

# The installed package configuration requires the same dependency versions.
set (SOPHUS_EIGEN3_VERSION 3.4.1...5)
set (SOPHUS_FMT_VERSION 5.3.0)

find_package (Eigen3 ${SOPHUS_EIGEN3_VERSION} REQUIRED CONFIG)
find_package (fmt ${SOPHUS_FMT_VERSION} REQUIRED CONFIG)

if (PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)
  # Ceres follows Eigen3 since its package configuration searches for Eigen3.
  find_package (Ceres 2.2.0 CONFIG)
  find_package (GTest 1.10 REQUIRED CONFIG)
endif (PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)

include (CMakePackageConfigHelpers)
include (GNUInstallDirs)
include (GoogleTest)

add_library (sophus INTERFACE)
add_library (Sophus::Sophus ALIAS sophus)

target_sources (sophus INTERFACE
  FILE_SET HEADERS
  FILES
    sophus/average.hpp
    sophus/cartesian.hpp
    sophus/ceres_manifold.hpp
    sophus/ceres_typetraits.hpp
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
    sophus/trig_details.hpp
    sophus/types.hpp
    sophus/velocities.hpp
)

target_compile_features (sophus INTERFACE cxx_std_17)
target_link_libraries (sophus INTERFACE
  Eigen3::Eigen
  fmt::fmt
)

set_target_properties (sophus PROPERTIES EXPORT_NAME Sophus)

if (PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)
  # Adds a Google Test executable and registers each of its test cases with
  # CTest. Tests of Ceres Jet scalars are enabled if Ceres is available.
  function (sophus_add_test name)
    add_executable (${name} ${ARGN})
    target_include_directories (${name} PRIVATE
      ${PROJECT_SOURCE_DIR}/test/support
    )
    target_link_libraries (${name} PRIVATE
      GTest::gmock_main
      Sophus::Sophus
    )
    # The typed tests of a Debug build exceed the default section limit of
    # MSVC object files.
    target_compile_options (${name} PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
    )

    if (Ceres_FOUND)
      target_compile_definitions (${name} PRIVATE SOPHUS_CERES)
      target_link_libraries (${name} PRIVATE Ceres::ceres)
    endif (Ceres_FOUND)

    # Listing thousands of registered tests takes longer than the default
    # timeout of 5 seconds under an emulator such as Wine.
    gtest_discover_tests (${name}
      DISCOVERY_TIMEOUT 60
      TEST_PREFIX ${name}/
      PROPERTIES LABELS sophus
    )
  endfunction (sophus_add_test)

  sophus_add_test (test_matchers test/support/test_matchers.cpp)

  foreach (test_name IN ITEMS
    test_cartesian2
    test_cartesian3
    test_common
    test_geometry
    test_rxso2
    test_rxso3
    test_se2
    test_se3
    test_sim2
    test_sim3
    test_so2
    test_so3
    test_velocities
  )
    sophus_add_test (${test_name}
      test/core/${test_name}.cpp
      test/core/tests.hpp
    )
  endforeach (test_name)

  # Building the program must fail on the shape check of Cartesian, whose
  # message distinguishes it from unrelated compile errors.
  add_executable (test_cartesian_fixed_shape EXCLUDE_FROM_ALL
    test/compile_fail/cartesian_fixed_shape.cpp
  )
  target_link_libraries (test_cartesian_fixed_shape PRIVATE Sophus::Sophus)
  add_test (NAME test_cartesian_fixed_shape
    COMMAND ${CMAKE_COMMAND} --build ${PROJECT_BINARY_DIR}
            --target test_cartesian_fixed_shape --config $<CONFIG>
  )
  set_tests_properties (test_cartesian_fixed_shape PROPERTIES
    PASS_REGULAR_EXPRESSION "must be an M-vector or an \\(M\\+1\\)x\\(M\\+1\\)"
  )

  # Installs the package into the build tree and builds a project against
  # it, so that the installed package configuration is tested as consumers
  # use it. The previous installation is removed first, since its files
  # would otherwise hide files missing from the new one.
  set (SOPHUS_PACKAGE_TEST_DIR ${PROJECT_BINARY_DIR}/package-test)
  add_test (NAME package_clean
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${SOPHUS_PACKAGE_TEST_DIR}
  )
  add_test (NAME package_install
    COMMAND ${CMAKE_COMMAND} --install ${PROJECT_BINARY_DIR}
            --prefix ${SOPHUS_PACKAGE_TEST_DIR}/install --config $<CONFIG>
  )
  add_test (NAME package_consumer
    COMMAND ${CMAKE_CTEST_COMMAND}
            --build-and-test ${PROJECT_SOURCE_DIR}/test/package
                             ${SOPHUS_PACKAGE_TEST_DIR}/build
            --build-generator ${CMAKE_GENERATOR}
            --build-config $<CONFIG>
            --build-options
              -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
              -DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF
              -DCMAKE_PREFIX_PATH=${SOPHUS_PACKAGE_TEST_DIR}/install
              -DEigen3_DIR=${Eigen3_DIR}
              -Dfmt_DIR=${fmt_DIR}
              -DSOPHUS_SOURCE_DIR=${PROJECT_SOURCE_DIR}
            --test-command consumer
  )
  set_tests_properties (package_clean PROPERTIES FIXTURES_SETUP package_clean)
  set_tests_properties (package_install PROPERTIES
    FIXTURES_REQUIRED package_clean
    FIXTURES_SETUP package
  )
  set_tests_properties (package_consumer PROPERTIES FIXTURES_REQUIRED package)

  if (Ceres_FOUND)
    foreach (test_name IN ITEMS
      test_ceres_rxso2
      test_ceres_rxso3
      test_ceres_se2
      test_ceres_se3
      test_ceres_sim2
      test_ceres_sim3
      test_ceres_so2
      test_ceres_so3
    )
      sophus_add_test (${test_name}
        test/ceres/${test_name}.cpp
        test/ceres/tests.hpp
      )
    endforeach (test_name)
  endif (Ceres_FOUND)
endif (PROJECT_IS_TOP_LEVEL AND BUILD_TESTING)

if (PROJECT_IS_TOP_LEVEL AND BUILD_EXAMPLES)
  add_executable (HelloSO3 examples/HelloSO3.cpp)
  target_link_libraries (HelloSO3 PRIVATE Sophus::Sophus)

  if (BUILD_TESTING)
    # Running the example keeps it working, not only compiling.
    add_test (NAME HelloSO3 COMMAND HelloSO3)
  endif (BUILD_TESTING)
endif (PROJECT_IS_TOP_LEVEL AND BUILD_EXAMPLES)

set (SOPHUS_CMAKE_EXPORT_DIR ${CMAKE_INSTALL_DATADIR}/sophus/cmake)

install (TARGETS sophus
  EXPORT SophusTargets
  FILE_SET HEADERS
)
install (EXPORT SophusTargets
  NAMESPACE Sophus::
  DESTINATION ${SOPHUS_CMAKE_EXPORT_DIR}
)

# Allows to use the package from the build tree without installing it.
export (EXPORT SophusTargets
  NAMESPACE Sophus::
  FILE ${PROJECT_BINARY_DIR}/SophusTargets.cmake
)

configure_package_config_file (SophusConfig.cmake.in
  ${PROJECT_BINARY_DIR}/SophusConfig.cmake
  INSTALL_DESTINATION ${SOPHUS_CMAKE_EXPORT_DIR}
  NO_SET_AND_CHECK_MACRO
)
write_basic_package_version_file (${PROJECT_BINARY_DIR}/SophusConfigVersion.cmake
  COMPATIBILITY SameMajorVersion
  ARCH_INDEPENDENT
)

install (FILES
  ${PROJECT_BINARY_DIR}/SophusConfig.cmake
  ${PROJECT_BINARY_DIR}/SophusConfigVersion.cmake
  DESTINATION ${SOPHUS_CMAKE_EXPORT_DIR}
)
