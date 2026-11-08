// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#include <cmath>
#include <vector>

#include <sophus/velocities.hpp>

#include "tests.hpp"

namespace Sophus {
namespace experimental {
namespace {

using ::testing::Pointwise;

template <class Scalar>
class VelocitiesTest : public ::testing::Test {};

TYPED_TEST_SUITE(VelocitiesTest, FloatingPointScalars, ScalarNameGenerator);

template <class Scalar>
SE3<Scalar> fooTBar() {
  return SE3<Scalar>::rotX(Scalar(0.5)) * SE3<Scalar>::rotZ(Scalar(0.2)) *
         SE3<Scalar>::transY(Scalar(2));
}

/// Finite differences of consecutive translations.
template <class Scalar>
StdVector<Vector3<Scalar>> linearVelocities(
    StdVector<SE3<Scalar>> const& a_Ts_b) {
  StdVector<Vector3<Scalar>> linear_vels_a;
  for (std::size_t i = 0; i + 1 < a_Ts_b.size(); ++i) {
    linear_vels_a.push_back(a_Ts_b[i + 1].translation() -
                            a_Ts_b[i].translation());
  }
  return linear_vels_a;
}

TYPED_TEST(VelocitiesTest, TransformsLinearVelocitiesBetweenFrames) {
  using Scalar = TypeParam;
  using std::sqrt;
  int constexpr kNumPoses = 10;
  // Long enough that rotating the steps changes them by far more than the
  // tolerance.
  Scalar const kStepLength(1);
  SE3<Scalar> const foo_T_bar = fooTBar<Scalar>();

  StdVector<SE3<Scalar>> bar_Ts_baz;
  for (int i = 0; i < kNumPoses; ++i) {
    bar_Ts_baz.push_back(SE3<Scalar>::rotX(Scalar(i * 0.001)) *
                         SE3<Scalar>::rotY(Scalar(i * 0.001)) *
                         SE3<Scalar>::transX(kStepLength * Scalar(i)));
  }
  auto const foo_Ts_baz =
      mapped(bar_Ts_baz, [&foo_T_bar](SE3<Scalar> const& bar_T_baz) {
        return SE3<Scalar>(foo_T_bar * bar_T_baz);
      });

  auto const transformed =
      mapped(linearVelocities(bar_Ts_baz),
             [&foo_T_bar](Vector3<Scalar> const& linear_vel_bar) {
               return transformVelocity(foo_T_bar, linear_vel_bar);
             });
  EXPECT_THAT(transformed, Pointwise(IsNear(sqrt(Constants<Scalar>::epsilon())),
                                     linearVelocities(foo_Ts_baz)));
}

TYPED_TEST(VelocitiesTest, TransformsRotationalVelocitiesBetweenFrames) {
  using Scalar = TypeParam;
  using std::sqrt;
  SE3<Scalar> const foo_T_bar = fooTBar<Scalar>();
  // Not too tight, because of the finite difference approximation.
  Scalar const kTolerance = sqrt(Constants<Scalar>::epsilon());
  // The velocities in frames foo and bar differ by more than this.
  Scalar const kMinFrameDifference = Scalar(1e-3);
  std::vector<Scalar> const ts = {Scalar(0), Scalar(0.3), Scalar(1)};

  // One parameter subgroup of SE3, motion through space given time t.
  auto const bar_T_baz = [](Scalar t) -> SE3<Scalar> {
    return SE3<Scalar>::rotX(t * Scalar(0.01)) *
           SE3<Scalar>::rotY(t * Scalar(0.0001)) *
           SE3<Scalar>::transX(t * Scalar(0.0001));
  };
  auto const rot_vels_in_frame_bar = mapped(ts, [&bar_T_baz](Scalar t) {
    return finiteDifferenceRotationalVelocity<Scalar>(bar_T_baz, t);
  });
  auto const rot_vels_in_frame_foo =
      mapped(ts, [&foo_T_bar, &bar_T_baz](Scalar t) {
        return finiteDifferenceRotationalVelocity<Scalar>(
            [&foo_T_bar, &bar_T_baz](Scalar t_bar) -> SE3<Scalar> {
              return foo_T_bar * bar_T_baz(t_bar);
            },
            t);
      });
  auto const transformed_to_bar = mapped(
      rot_vels_in_frame_foo, [&foo_T_bar](Vector3<Scalar> const& vel_foo) {
        return transformVelocity(foo_T_bar.inverse(), vel_foo);
      });

  EXPECT_THAT(transformed_to_bar,
              Pointwise(IsNear(kTolerance), rot_vels_in_frame_bar));

  // The velocities in frames foo and bar differ, since foo != bar.
  EXPECT_THAT(rot_vels_in_frame_foo,
              Pointwise(::testing::Not(IsNear(kMinFrameDifference)),
                        rot_vels_in_frame_bar));

  // The adjoint gives the same result, since
  //   vee(bar_R_foo * hat(vel_foo) * bar_R_foo^T) = bar_R_foo * vel_foo.
  auto const conjugated = mapped(
      rot_vels_in_frame_foo, [&foo_T_bar](Vector3<Scalar> const& vel_foo) {
        return Vector3<Scalar>(SO3<Scalar>::vee(
            foo_T_bar.so3().inverse().matrix() * SO3<Scalar>::hat(vel_foo) *
            foo_T_bar.so3().matrix()));
      });
  EXPECT_THAT(transformed_to_bar,
              Pointwise(IsNear(Constants<Scalar>::epsilon()), conjugated));
}

TYPED_TEST(VelocitiesTest, DefaultStepRecoversRotationalVelocity) {
  using Scalar = TypeParam;
  // The error of a central difference with a step near the cube root of the
  // machine epsilon is far below this tolerance.
  Scalar const kTolerance = Scalar(10) * Constants<Scalar>::epsilon();
  Vector3<Scalar> const velocity(Scalar(0.3), Scalar(-0.2), Scalar(0.1));
  std::function<SO3<Scalar>(Scalar)> const foo_R_bar = [velocity](Scalar t) {
    return SO3<Scalar>::exp(t * velocity);
  };

  EXPECT_THAT(
      finiteDifferenceRotationalVelocity<Scalar>(foo_R_bar, Scalar(0.7)),
      IsNear(velocity, kTolerance));
}

}  // namespace
}  // namespace experimental
}  // namespace Sophus
