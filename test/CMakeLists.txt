# Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

find_package (GTest 1.10 REQUIRED NO_MODULE)

include (GoogleTest)

set (SOPHUS_TEST_SUPPORT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/support)

# Adds a Google Test executable and registers each of its test cases with
# CTest. Tests of Ceres Jet scalars are enabled if Ceres is available.
function (sophus_add_test name)
  add_executable (${name} ${ARGN})
  target_include_directories (${name} PRIVATE ${SOPHUS_TEST_SUPPORT_DIR})
  target_link_libraries (${name} PRIVATE GTest::gmock_main sophus)

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

sophus_add_test (test_matchers support/test_matchers.cpp)

add_subdirectory (core)
add_subdirectory (ceres)
