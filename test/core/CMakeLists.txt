# Copyright 2021-2022, 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

foreach (test_name
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
  sophus_add_test (${test_name} ${test_name}.cpp tests.hpp)
endforeach (test_name)
