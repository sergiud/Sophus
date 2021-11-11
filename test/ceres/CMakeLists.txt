if( Ceres_FOUND )
  # Tests to run
  SET( TEST_SOURCES test_ceres_se3 test_ceres_so3 test_ceres_sim3 test_ceres_rxso3 test_ceres_se2 test_ceres_so2 test_ceres_rxso2 test_ceres_sim2 )

  FOREACH(test_src ${TEST_SOURCES})
    ADD_EXECUTABLE( ${test_src} ${test_src}.cpp local_parameterization_se3.hpp)
    TARGET_LINK_LIBRARIES( ${test_src} sophus Ceres::Ceres)
    ADD_TEST( ${test_src} ${test_src} )
  ENDFOREACH(test_src)

endif( Ceres_FOUND )
