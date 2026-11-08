// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#include <cmath>
#include <cstdlib>
#include <tuple>

#include <sophus/geometry.hpp>

#include "tests.hpp"

namespace Sophus {
namespace {

using ::testing::Pointwise;

/// The number of random samples of each roundtrip test.
int constexpr kNumSamples = 20;

template <class Scalar>
class GeometryTest : public ::testing::Test {};

TYPED_TEST_SUITE(GeometryTest, FloatingPointScalars, ScalarNameGenerator);

TYPED_TEST(GeometryTest, NormalRoundtripsThroughSO2) {
  using Scalar = TypeParam;
  auto const normals = mapped(
      randomMatrices<Vector2<Scalar>>(kNumSamples),
      [](Vector2<Scalar> const& v) { return Vector2<Scalar>(v.normalized()); });
  EXPECT_THAT(mapped(normals,
                     [](Vector2<Scalar> const& normal_foo) {
                       return normalFromSO2(SO2FromNormal(normal_foo));
                     }),
              Pointwise(IsNear(Constants<Scalar>::epsilon()), normals));
}

/// Random lines or planes in their unique representation.
template <class Hyperplane>
StdVector<Hyperplane> randomHyperplanes() {
  using Vector = typename Hyperplane::VectorType;
  StdVector<Hyperplane> hyperplanes;
  for (int i = 0; i < kNumSamples; ++i) {
    hyperplanes.push_back(makeHyperplaneUnique(
        Hyperplane(Vector(Vector::Random().normalized()), Vector::Random())));
  }
  return hyperplanes;
}

/// Lines and planes compared by their normals and offsets.
template <class Hyperplane>
auto normalsAndOffsets(StdVector<Hyperplane> const& hyperplanes) {
  using Scalar = typename Hyperplane::Scalar;
  int constexpr kDim = Hyperplane::AmbientDimAtCompileTime;
  return mapped(hyperplanes, [](Hyperplane const& h) {
    return Vector<Scalar, kDim + 1>(h.coeffs());
  });
}

TYPED_TEST(GeometryTest, LineRoundtripsThroughSE2) {
  using Scalar = TypeParam;
  auto const lines = randomHyperplanes<Line2<Scalar>>();
  auto const roundtrips = mapped(lines, [](Line2<Scalar> const& line_foo) {
    return lineFromSE2(SE2FromLine(line_foo));
  });
  EXPECT_THAT(normalsAndOffsets(roundtrips),
              Pointwise(IsNear(Constants<Scalar>::epsilon()),
                        normalsAndOffsets(lines)));
}

TYPED_TEST(GeometryTest, SE2RoundtripsThroughLine) {
  using Scalar = TypeParam;
  auto const lines = mapped(getTestSE2s<Scalar>(), [](SE2<Scalar> const& T) {
    return lineFromSE2(T);
  });
  auto const roundtrips = mapped(lines, [](Line2<Scalar> const& line_foo) {
    return lineFromSE2(SE2FromLine(line_foo));
  });
  EXPECT_THAT(normalsAndOffsets(roundtrips),
              Pointwise(IsNear(Constants<Scalar>::epsilon()),
                        normalsAndOffsets(lines)));
}

TYPED_TEST(GeometryTest, RotationFromNormalHasTheNormalAsThirdColumn) {
  using Scalar = TypeParam;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();
  Vector3<Scalar> const normal_foo =
      Vector3<Scalar>(Scalar(1), Scalar(2), Scalar(0)).normalized();

  EXPECT_THAT(rotationFromNormal(normal_foo).col(2),
              IsNear(normal_foo, kEpsilon));
  // The normal is normalized.
  EXPECT_THAT(rotationFromNormal((Scalar(0.9) * normal_foo).eval()).col(2),
              IsNear(normal_foo, kEpsilon));
}

TYPED_TEST(GeometryTest, RotationFromNormalIgnoresTheLengthOfHints) {
  using Scalar = TypeParam;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();
  Scalar const kLong(10);
  Scalar const kShort(0.1);
  // The normal is closer to the y-hint than to the x-hint. A longer x-hint
  // or a shorter y-hint would reverse that unless the hints are normalized,
  // and the frame would be built from the other hint.
  Vector3<Scalar> const normal_foo(Scalar(0.6), Scalar(0.8), Scalar(0));
  Vector3<Scalar> const unit_x(Scalar(1), Scalar(0), Scalar(0));
  Vector3<Scalar> const unit_y(Scalar(0), Scalar(1), Scalar(0));
  Matrix3<Scalar> const expected =
      rotationFromNormal(normal_foo, unit_x, unit_y);

  EXPECT_THAT(
      rotationFromNormal(normal_foo, Vector3<Scalar>(kLong * unit_x), unit_y),
      IsNear(expected, kEpsilon));
  EXPECT_THAT(
      rotationFromNormal(normal_foo, unit_x, Vector3<Scalar>(kShort * unit_y)),
      IsNear(expected, kEpsilon));
}

TYPED_TEST(GeometryTest, RotationFromAxisNormalFollowsTheHints) {
  using Scalar = TypeParam;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();
  Vector3<Scalar> const kUnitX(Scalar(1), Scalar(0), Scalar(0));
  Vector3<Scalar> const kUnitY(Scalar(0), Scalar(1), Scalar(0));

  Matrix3<Scalar> const R_x = rotationFromNormal(kUnitX);
  EXPECT_THAT(R_x.col(2), IsNear(kUnitX, kEpsilon));
  EXPECT_THAT(R_x.col(1), IsNear(kUnitY, kEpsilon));

  Matrix3<Scalar> const R_y = rotationFromNormal(kUnitY);
  EXPECT_THAT(R_y.col(2), IsNear(kUnitY, kEpsilon));
  EXPECT_THAT(R_y.col(0), IsNear(kUnitX, kEpsilon));
}

TYPED_TEST(GeometryTest, NormalRoundtripsThroughSO3) {
  using Scalar = TypeParam;
  auto const normals = mapped(
      randomMatrices<Vector3<Scalar>>(kNumSamples),
      [](Vector3<Scalar> const& v) { return Vector3<Scalar>(v.normalized()); });
  EXPECT_THAT(mapped(normals,
                     [](Vector3<Scalar> const& normal_foo) {
                       return normalFromSO3(SO3FromNormal(normal_foo));
                     }),
              Pointwise(IsNear(Constants<Scalar>::epsilon()), normals));
}

TYPED_TEST(GeometryTest, PlaneRoundtripsThroughSE3) {
  using Scalar = TypeParam;
  auto const planes = randomHyperplanes<Plane3<Scalar>>();
  auto const roundtrips = mapped(planes, [](Plane3<Scalar> const& plane_foo) {
    return planeFromSE3(SE3FromPlane(plane_foo));
  });
  EXPECT_THAT(normalsAndOffsets(roundtrips),
              Pointwise(IsNear(Constants<Scalar>::epsilon()),
                        normalsAndOffsets(planes)));
}

TYPED_TEST(GeometryTest, SE3RoundtripsThroughPlane) {
  using Scalar = TypeParam;
  auto const planes = mapped(getTestSE3s<Scalar>(), [](SE3<Scalar> const& T) {
    return planeFromSE3(T);
  });
  auto const roundtrips = mapped(planes, [](Plane3<Scalar> const& plane_foo) {
    return planeFromSE3(SE3FromPlane(plane_foo));
  });
  EXPECT_THAT(normalsAndOffsets(roundtrips),
              Pointwise(IsNear(Constants<Scalar>::epsilon()),
                        normalsAndOffsets(planes)));
}

TYPED_TEST(GeometryTest, FrameOfPlaneWithNonUnitNormalLiesOnThePlane) {
  using Scalar = TypeParam;
  // The plane 2z - 2 = 0, that is z = 1.
  Plane3<Scalar> const plane_foo(
      Vector3<Scalar>(Scalar(0), Scalar(0), Scalar(2)), Scalar(-2));
  SE3<Scalar> const T_foo_plane = SE3FromPlane(plane_foo);
  EXPECT_THAT(T_foo_plane.translation(),
              IsNear(Vector3<Scalar>(Scalar(0), Scalar(0), Scalar(1)),
                     Constants<Scalar>::epsilon()));
}

TYPED_TEST(GeometryTest, FrameOfLineWithNonUnitNormalLiesOnTheLine) {
  using Scalar = TypeParam;
  // The line 3y - 3 = 0, that is y = 1.
  Line2<Scalar> const line_foo(Vector2<Scalar>(Scalar(0), Scalar(3)),
                               Scalar(-3));
  SE2<Scalar> const T_foo_line = SE2FromLine(line_foo);
  EXPECT_THAT(T_foo_line.translation(),
              IsNear(Vector2<Scalar>(Scalar(0), Scalar(1)),
                     Constants<Scalar>::epsilon()));
}

/// Expects both signs of a plane to be made unique as the expected
/// coefficients, which flipping the sign computes exactly.
template <class Scalar>
void expectUniquePlane(Vector3<Scalar> const& normal, Scalar offset,
                       Vector4<Scalar> const& expected) {
  EXPECT_EQ(makeHyperplaneUnique(Plane3<Scalar>(normal, offset)).coeffs(),
            expected)
      << "for normal " << normal.transpose() << " and offset " << offset;
  EXPECT_EQ(makeHyperplaneUnique(Plane3<Scalar>(-normal, -offset)).coeffs(),
            expected)
      << "for normal " << -normal.transpose() << " and offset " << -offset;
}

TYPED_TEST(GeometryTest, UniqueHyperplaneThroughOriginIgnoresNormalSign) {
  using Scalar = TypeParam;
  // The first nonzero coefficient of the normal becomes positive.
  expectUniquePlane(
      Vector3<Scalar>(Scalar(0), Scalar(-0.6), Scalar(0.8)), Scalar(0),
      Vector4<Scalar>(Scalar(0), Scalar(0.6), Scalar(-0.8), Scalar(0)));
  expectUniquePlane(
      Vector3<Scalar>(Scalar(-0.6), Scalar(0.8), Scalar(0)), Scalar(0),
      Vector4<Scalar>(Scalar(0.6), Scalar(-0.8), Scalar(0), Scalar(0)));
  expectUniquePlane(
      Vector3<Scalar>(Scalar(1), Scalar(0), Scalar(0)), Scalar(0),
      Vector4<Scalar>(Scalar(1), Scalar(0), Scalar(0), Scalar(0)));
}

TYPED_TEST(GeometryTest, UniqueHyperplaneHasNonnegativeOffset) {
  using Scalar = TypeParam;
  // The plane z = 2, whose offset is negative for the normal along z.
  expectUniquePlane(
      Vector3<Scalar>(Scalar(0), Scalar(0), Scalar(1)), Scalar(-2),
      Vector4<Scalar>(Scalar(0), Scalar(0), Scalar(-1), Scalar(2)));
}

TEST(GeometryDeathTest, PerpendicularityOfHintsIgnoresTheirLength) {
  Vector3<double> const normal(0, 0, 1);
  // Long hints whose angle differs from 90 degrees by about 1e-12.
  EXPECT_EXIT(
      {
        std::ignore = rotationFromNormal(normal, Vector3<double>(1e6, 0, 0),
                                         Vector3<double>(1e-6, 1e6, 0));
        std::exit(0);
      },
      ::testing::ExitedWithCode(0), "");
  // Short hints at an angle of about 66 and 114 degrees, whose dot products
  // are below the tolerance on the cosine of their angle.
  double constexpr kShortLength = 1.2e-5;
  double constexpr kCosine = 0.4;
  double const sine = std::sqrt(1 - kCosine * kCosine);
  Vector3<double> const short_x(kShortLength, 0, 0);
  EXPECT_DEATH(
      std::ignore = rotationFromNormal(
          normal, short_x,
          Vector3<double>(kShortLength * kCosine, kShortLength * sine, 0)),
      "perpendicular");
  EXPECT_DEATH(
      std::ignore = rotationFromNormal(
          normal, short_x,
          Vector3<double>(-kShortLength * kCosine, kShortLength * sine, 0)),
      "perpendicular");
}

}  // namespace
}  // namespace Sophus
