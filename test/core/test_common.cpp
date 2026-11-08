// Copyright 2022, 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#include <utility>
#include <vector>

#include <sophus/interpolate.hpp>
#include <sophus/num_diff.hpp>
#include <sophus/se3.hpp>
#include <sophus/spline.hpp>
#include <sophus/trig_details.hpp>

#include "reference.hpp"
#include "tests.hpp"

namespace Sophus {
namespace {

template <class Scalar>
class CommonTest : public ::testing::Test {};

TYPED_TEST_SUITE(CommonTest, FloatingPointScalars, ScalarNameGenerator);

TEST(EnsureDeathTest, ReportsTheFormattedMessageOnStandardError) {
  int const actual = 1;
  EXPECT_DEATH(SOPHUS_ENSURE(actual == 2, "expected {}, got {}", 2, actual),
               "Sophus ensure failed .*\nexpected 2, got 1");
}

TYPED_TEST(CommonTest, TransposeOfScalarIsTheScalar) {
  using Scalar = TypeParam;
  EXPECT_EQ(transpose(Scalar(42)), Scalar(42));
}

TYPED_TEST(CommonTest, TransposeOfVectorIsARowVector) {
  using Scalar = TypeParam;
  Matrix<Scalar, 1, 2> const row =
      transpose(Vector2<Scalar>(Scalar(1), Scalar(7)));
  Matrix<Scalar, 1, 2> const expected_row(Scalar(1), Scalar(7));
  EXPECT_EQ(row, expected_row);
}

TYPED_TEST(CommonTest, OneMinusCosOverThetaSqMatchesReference) {
  using Scalar = TypeParam;
  // Single-precision angles and their values from mpmath at 50 digits.
  std::vector<reference::Case> const cases = {
      {{0x0.0p+0L}, {0.5L}},
      {{0x1.0c6f7ap-20L}, {0.4999999999999583333335437L}},
      {{0x1.92a738p-19L}, {0.499999999999624999973472L}},
      {{0x1.0624dep-10L}, {0.4999999583333307641014252L}},
      {{0x1p+0L}, {0.4596976941318602825990634L}},
      {{0x1.8p+1L}, {0.221110277400049495252397L}},
  };
  reference::expectAccurate<Scalar>(cases, [](reference::Case const& c) {
    return details::oneMinusCosOverThetaSq(reference::input<Scalar>(c));
  });
}

TYPED_TEST(CommonTest, NumericalDerivativesScaleTheStepWithTheArgument) {
  using Scalar = TypeParam;
  // An absolute step is below the spacing of the floating-point numbers
  // around this argument, so that t + h rounds to t.
  Scalar const t = Scalar(1) / Constants<Scalar>::epsilon();
  Scalar const kTolerance = Constants<Scalar>::epsilonSqrt();
  auto const square = [](Scalar x) { return x * x; };

  Scalar const derivative = curveNumDiff(square, t);
  EXPECT_THAT(derivative / (Scalar(2) * t), IsNear(Scalar(1), kTolerance));

  Vector2<Scalar> const a(t, Scalar(1));
  Matrix2<Scalar> const J = vectorFieldNumDiff<Scalar, 2, 2>(
      [square](Vector2<Scalar> const& x) {
        return Vector2<Scalar>(square(x[0]), square(x[1]));
      },
      a);
  EXPECT_THAT(J(0, 0) / (Scalar(2) * t), IsNear(Scalar(1), kTolerance));
  EXPECT_THAT(J(1, 1) / Scalar(2), IsNear(Scalar(1), kTolerance));
}

/// The time steps and spline parameters the derivatives of the spline basis
/// functions are checked at.
template <class Scalar>
std::vector<std::pair<Scalar, Scalar>> splineBasisSamples() {
  std::vector<std::pair<Scalar, Scalar>> samples;
  for (Scalar const delta_t : {Scalar(0.1), Scalar(0.5), Scalar(1.0)}) {
    for (Scalar const u : {Scalar(0.0000), Scalar(0.1), Scalar(0.5),
                           Scalar(0.51), Scalar(0.9), Scalar(0.999)}) {
      samples.emplace_back(delta_t, u);
    }
  }
  return samples;
}

TYPED_TEST(CommonTest, SplineBasisDerivativeMatchesNumericalDerivative) {
  using Scalar = TypeParam;
  using Basis = SplineBasisFunction<Scalar>;
  auto const samples = splineBasisSamples<Scalar>();

  auto const Dt_B = mapped(samples, [](std::pair<Scalar, Scalar> const& s) {
    return Basis::Dt_B(s.second, s.first);
  });
  auto const Dt_B_num = mapped(samples, [](std::pair<Scalar, Scalar> const& s) {
    Scalar const delta_t = s.first;
    return curveNumDiff(
        [delta_t](Scalar u_bar) -> Vector3<Scalar> {
          return Basis::B(u_bar) / delta_t;
        },
        s.second);
  });
  EXPECT_THAT(Dt_B, ::testing::Pointwise(
                        IsNear(Constants<Scalar>::epsilonSqrt()), Dt_B_num))
      << "for each pair of time step and spline parameter";
}

TYPED_TEST(CommonTest, SplineBasisSecondDerivativeMatchesNumericalDerivative) {
  using Scalar = TypeParam;
  using Basis = SplineBasisFunction<Scalar>;
  auto const samples = splineBasisSamples<Scalar>();

  auto const Dt2_B = mapped(samples, [](std::pair<Scalar, Scalar> const& s) {
    return Basis::Dt2_B(s.second, s.first);
  });
  auto const Dt2_B_num =
      mapped(samples, [](std::pair<Scalar, Scalar> const& s) {
        Scalar const delta_t = s.first;
        return curveNumDiff(
            [delta_t](Scalar u_bar) -> Vector3<Scalar> {
              return Basis::Dt_B(u_bar, delta_t) / delta_t;
            },
            s.second);
      });
  EXPECT_THAT(Dt2_B, ::testing::Pointwise(
                         IsNear(Constants<Scalar>::epsilonSqrt()), Dt2_B_num))
      << "for each pair of time step and spline parameter";
}

/// A spline that interpolates between two rigid body motions.
template <class Scalar>
BasisSplineImpl<SE3<Scalar>> se3Spline() {
  using SE3Type = SE3<Scalar>;
  int constexpr kNumIntermediatePoses = 4;
  SE3Type const T_world_foo =
      SE3Type::rotX(Scalar(0.4)) * SE3Type::rotY(Scalar(0.2));
  SE3Type const T_world_bar = SE3Type::rotZ(Scalar(-0.4)) *
                              SE3Type::rotY(Scalar(0.4)) *
                              SE3Type::trans(Scalar(0), Scalar(1), Scalar(-2));

  std::vector<SE3Type> control_poses;
  control_poses.push_back(interpolate(T_world_foo, T_world_bar, Scalar(0)));
  for (int i = 1; i <= kNumIntermediatePoses; ++i) {
    Scalar const p = Scalar(i) / Scalar(kNumIntermediatePoses + 1);
    control_poses.push_back(interpolate(T_world_foo, T_world_bar, p));
  }
  return BasisSplineImpl<SE3Type>(control_poses, Scalar(1));
}

TEST(BasisSplineTest, EndsMapToFirstAndLastSegments) {
  // The start times are ordered such that the midpoint of the spline lies
  // on either side of half its end time.
  for (double const t0 : {-20.0, 0.0, 100.0}) {
    BasisSplineImpl<SE3d> const impl = se3Spline<double>();
    std::vector<SE3d> const control_poses(impl.parent_Ts_control_point());
    BasisSpline<SE3d> const spline(control_poses, t0, impl.delta_t());
    int const last_segment = spline.getNumSegments() - 1;

    IndexAndU const start = spline.index_and_u(spline.t0());
    EXPECT_EQ(start.i, 0) << "t0 = " << t0;
    EXPECT_EQ(start.u, 0.0) << "t0 = " << t0;

    IndexAndU const end = spline.index_and_u(spline.tmax());
    EXPECT_EQ(end.i, last_segment) << "t0 = " << t0;
    EXPECT_EQ(end.u, 1.0) << "t0 = " << t0;
  }
}

TYPED_TEST(CommonTest, SE3SplineSegmentsJoinContinuously) {
  using Scalar = TypeParam;
  Scalar const kEpsSqrt = Constants<Scalar>::epsilonSqrt();
  BasisSplineImpl<SE3<Scalar>> const spline = se3Spline<Scalar>();

  SE3<Scalar> const T = spline.parent_T_spline(0, Scalar(1));
  SE3<Scalar> const T2 = spline.parent_T_spline(1, Scalar(0));
  EXPECT_THAT(T.so3().matrix(), IsNear(T2.so3().matrix(), kEpsSqrt));
  EXPECT_THAT(T.translation(), IsNear(T2.translation(), kEpsSqrt));
}

TYPED_TEST(CommonTest, SE3SplineDerivativesMatchNumericalDerivatives) {
  using Scalar = TypeParam;
  Scalar const kEpsSqrt = Constants<Scalar>::epsilonSqrt();
  Scalar const kU(0.5);
  BasisSplineImpl<SE3<Scalar>> const spline = se3Spline<Scalar>();

  Matrix4<Scalar> const Dt_parent_T_spline2 = curveNumDiff(
      [&spline](Scalar u_bar) -> Matrix4<Scalar> {
        return spline.parent_T_spline(0, u_bar).matrix();
      },
      kU);
  EXPECT_THAT(spline.Dt_parent_T_spline(0, kU),
              IsNear(Dt_parent_T_spline2, kEpsSqrt));

  Matrix4<Scalar> const Dt2_parent_T_spline2 = curveNumDiff(
      [&spline](Scalar u_bar) -> Matrix4<Scalar> {
        return spline.Dt_parent_T_spline(0, u_bar);
      },
      kU);
  EXPECT_THAT(spline.Dt2_parent_T_spline(0, kU),
              IsNear(Dt2_parent_T_spline2, kEpsSqrt));
}

}  // namespace
}  // namespace Sophus
