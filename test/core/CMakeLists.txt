#Tests to run
SET( TEST_SOURCES test_common
                  test_so2
                  test_se2
                  test_rxso2
                  test_sim2
                  test_so3
                  test_se3
                  test_rxso3
                  test_sim3
                  test_velocities
                  test_geometry)

FOREACH(test_src ${TEST_SOURCES})
  ADD_EXECUTABLE( ${test_src} ${test_src}.cpp tests.hpp ${Sophus_SOURCE_DIR}/sophus/test_macros.hpp)
  TARGET_LINK_LIBRARIES( ${test_src} sophus)
  if( Ceres_FOUND )
      TARGET_LINK_LIBRARIES( ${test_src} Ceres::Ceres )
      ADD_DEFINITIONS(-DSOPHUS_CERES)
  endif(Ceres_FOUND)
  ADD_TEST( ${test_src} ${test_src} )
ENDFOREACH(test_src)
