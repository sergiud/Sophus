// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

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
  // The normal and the hint directions are normalized.
  EXPECT_THAT(rotationFromNormal((Scalar(0.9) * normal_foo).eval()).col(2),
              IsNear(normal_foo, kEpsilon));
  EXPECT_THAT(
      rotationFromNormal(normal_foo,
                         Vector3<Scalar>(Scalar(0.9), Scalar(0), Scalar(0)),
                         Vector3<Scalar>(Scalar(0), Scalar(1.1), Scalar(0)))
          .col(2),
      IsNear(normal_foo, kEpsilon));
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

}  // namespace
}  // namespace Sophus
