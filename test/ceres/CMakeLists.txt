# Copyright 2021-2022, 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

if (Ceres_FOUND)
  foreach (test_name
    test_ceres_rxso2
    test_ceres_rxso3
    test_ceres_se2
    test_ceres_se3
    test_ceres_sim2
    test_ceres_sim3
    test_ceres_so2
    test_ceres_so3
  )
    sophus_add_test (${test_name} ${test_name}.cpp tests.hpp)
  endforeach (test_name)
endif (Ceres_FOUND)
