// Copyright 2022, 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#include <sophus/se2.hpp>

#include "reference_se2.hpp"
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::SE2<double>>;
template class Map<Sophus::SE2<double> const>;
}  // namespace Eigen

namespace Sophus {

template class SE2<double, Eigen::AutoAlign>;
template class SE2<double, Eigen::DontAlign>;
#if SOPHUS_CERES
template class SE2<ceres::Jet<double, 3>>;
#endif

template <class Scalar>
struct LieGroupTestData<SE2<Scalar>> {
  using SE2Type = SE2<Scalar>;
  using SO2Type = SO2<Scalar>;
  using Point = typename SE2Type::Point;
  using Tangent = typename SE2Type::Tangent;

  static StdVector<SE2Type> groupElements() {
    Scalar const kPi = Constants<Scalar>::pi();
    StdVector<SE2Type> se2_vec;
    se2_vec.push_back(
        SE2Type(SO2Type(Scalar(0.0)), Point(Scalar(0), Scalar(0))));
    se2_vec.push_back(
        SE2Type(SO2Type(Scalar(0.2)), Point(Scalar(10), Scalar(0))));
    se2_vec.push_back(
        SE2Type(SO2Type(Scalar(0.)), Point(Scalar(0), Scalar(100))));
    se2_vec.push_back(
        SE2Type(SO2Type(Scalar(-1.)), Point(Scalar(20), -Scalar(1))));
    se2_vec.push_back(
        SE2Type(SO2Type(Scalar(0.00001)),
                Point(Scalar(-0.00000001), Scalar(0.0000000001))));
    se2_vec.push_back(
        SE2Type(SO2Type(Scalar(0.2)), Point(Scalar(0), Scalar(0))) *
        SE2Type(SO2Type(kPi), Point(Scalar(0), Scalar(0))) *
        SE2Type(SO2Type(Scalar(-0.2)), Point(Scalar(0), Scalar(0))));
    se2_vec.push_back(
        SE2Type(SO2Type(Scalar(0.3)), Point(Scalar(2), Scalar(0))) *
        SE2Type(SO2Type(kPi), Point(Scalar(0), Scalar(0))) *
        SE2Type(SO2Type(Scalar(-0.3)), Point(Scalar(0), Scalar(6))));
    return se2_vec;
  }

  static StdVector<Tangent> tangents() {
    StdVector<Tangent> tangent_vec;
    tangent_vec.push_back(Tangent(Scalar(0), Scalar(0), Scalar(0)));
    tangent_vec.push_back(Tangent(Scalar(1), Scalar(0), Scalar(0)));
    tangent_vec.push_back(Tangent(Scalar(0), Scalar(1), Scalar(1)));
    tangent_vec.push_back(Tangent(Scalar(-1), Scalar(1), Scalar(0)));
    tangent_vec.push_back(Tangent(Scalar(20), Scalar(-1), Scalar(-1)));
    tangent_vec.push_back(Tangent(Scalar(30), Scalar(5), Scalar(20)));
    return tangent_vec;
  }

  static StdVector<Point> points() {
    StdVector<Point> point_vec;
    point_vec.push_back(Point(Scalar(1), Scalar(2)));
    point_vec.push_back(Point(Scalar(1), Scalar(-3)));
    return point_vec;
  }
};

namespace {

[[maybe_unused]] bool const kRegistered =
    registerLieGroupTests<SE2<double>, SE2<float>
#if SOPHUS_CERES
                          ,
                          SE2<ceres::Jet<double, 3>>
#endif
                          >("SE2");

template <class Scalar>
class SE2Test : public ::testing::Test {};

using SE2Scalars = ::testing::Types<double, float
#if SOPHUS_CERES
                                    ,
                                    ceres::Jet<double, 3>
#endif
                                    >;
TYPED_TEST_SUITE(SE2Test, SE2Scalars, ScalarNameGenerator);

// Catastrophic cancellation in SE2::log(): for theta in roughly [1e-5, 1e-3],
// computing the V_inv coefficient as sin(theta) / (cos(theta) - 1) cancels
// almost all significant digits, since cos(theta) - 1 is about -theta^2/2.
// The small-angle Taylor series threshold does not cover this range. This
// test is separate from the elements of LieGroupTestData, so that its
// tolerance is not diluted by the pairwise interpolation and mean checks
// against the other, much larger elements there.
TYPED_TEST(SE2Test, ExpInvertsLogForSmallAngles) {
  using Scalar = TypeParam;
  using SE2Type = SE2<Scalar>;
  using Point = typename SE2Type::Point;

  Scalar const theta(0.0001);
  Point const translation(Scalar(0.1), Scalar(-0.05));
  SE2Type const T(SO2<Scalar>(theta), translation);
  EXPECT_THAT(SE2Type::exp(T.log()), IsNear(T, Constants<Scalar>::epsilon()));
}

TYPED_TEST(SE2Test, MapsAccessTheUnderlyingData) {
  using Scalar = TypeParam;
  using SE2Type = SE2<Scalar>;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();

  Vector4<Scalar> raw(Scalar(0), Scalar(1), Scalar(0), Scalar(3));
  Eigen::Map<SE2Type const> const const_se2_map(raw.data());
  EXPECT_THAT(const_se2_map.unit_complex(),
              IsNear(raw.template head<2>(), kEpsilon));
  EXPECT_THAT(const_se2_map.translation(),
              IsNear(raw.template tail<2>(), kEpsilon));
  EXPECT_EQ(const_se2_map.unit_complex().data(), raw.data());
  EXPECT_EQ(const_se2_map.translation().data(), raw.data() + 2);
  Eigen::Map<SE2Type const> const const_shallow_copy = const_se2_map;
  EXPECT_EQ(const_shallow_copy.unit_complex().eval(),
            const_se2_map.unit_complex().eval());
  EXPECT_EQ(const_shallow_copy.translation().eval(),
            const_se2_map.translation().eval());

  Vector4<Scalar> const raw2(Scalar(1), Scalar(0), Scalar(3), Scalar(1));
  Eigen::Map<SE2Type> map_of_se2(raw.data());
  map_of_se2.setComplex(raw2.template head<2>());
  map_of_se2.translation() = raw2.template tail<2>();
  EXPECT_THAT(map_of_se2.unit_complex(),
              IsNear(raw2.template head<2>(), kEpsilon));
  EXPECT_THAT(map_of_se2.translation(),
              IsNear(raw2.template tail<2>(), kEpsilon));
  EXPECT_EQ(map_of_se2.unit_complex().data(), raw.data());
  EXPECT_EQ(map_of_se2.translation().data(), raw.data() + 2);
  EXPECT_NE(map_of_se2.unit_complex().data(), raw2.data());
  Eigen::Map<SE2Type> const shallow_copy = map_of_se2;
  EXPECT_EQ(shallow_copy.unit_complex().eval(),
            map_of_se2.unit_complex().eval());
  EXPECT_EQ(shallow_copy.translation().eval(), map_of_se2.translation().eval());
  Eigen::Map<SE2Type> const const_map_of_se2 = map_of_se2;
  EXPECT_EQ(const_map_of_se2.unit_complex().eval(),
            map_of_se2.unit_complex().eval());
  EXPECT_EQ(const_map_of_se2.translation().eval(),
            map_of_se2.translation().eval());
  // Writing through the map changed the underlying data.
  EXPECT_EQ(raw, raw2);
}

TYPED_TEST(SE2Test, DataAccessesTheParameters) {
  using Scalar = TypeParam;
  using SE2Type = SE2<Scalar>;

  Vector4<Scalar> const raw(Scalar(0), Scalar(1), Scalar(0), Scalar(3));
  Vector4<Scalar> const raw2(Scalar(1), Scalar(0), Scalar(3), Scalar(1));
  SE2Type const const_se2(raw2.template head<2>().eval(),
                          raw2.template tail<2>().eval());
  EXPECT_EQ(rawParams(const_se2), raw2);

  SE2Type se2(raw2.template head<2>().eval(), raw2.template tail<2>().eval());
  EXPECT_EQ(rawParams(se2), raw2);
  Eigen::Map<Vector<Scalar, 4>>(se2.data()) = raw;
  EXPECT_EQ(rawParams(se2), raw);
}

TYPED_TEST(SE2Test, TranslationsAlongAxes) {
  using Scalar = TypeParam;
  using SE2Type = SE2<Scalar>;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();

  EXPECT_THAT(SE2Type::transX(Scalar(0.2)).translation(),
              IsNear(Vector2<Scalar>(Scalar(0.2), Scalar(0)), kEpsilon));
  EXPECT_THAT(SE2Type::transY(Scalar(0.7)).translation(),
              IsNear(Vector2<Scalar>(Scalar(0), Scalar(0.7)), kEpsilon));
}

TYPED_TEST(SE2Test, AssignsBetweenMapsAndElements) {
  using Scalar = TypeParam;
  using SE2Type = SE2<Scalar>;

  Vector4<Scalar> data1(Scalar(0), Scalar(1), Scalar(1), Scalar(2));
  Vector4<Scalar> data2(Scalar(1), Scalar(0), Scalar(2), Scalar(1));
  Eigen::Map<SE2Type> map1(data1.data());
  Eigen::Map<SE2Type> map2(data2.data());

  map2 = map1;
  EXPECT_EQ(map2.matrix(), map1.matrix()) << "map -> map assignment";

  SE2Type copy;
  copy = map1;
  EXPECT_EQ(copy.matrix(), map1.matrix()) << "map -> type assignment";

  copy = SE2Type::trans(Scalar(4), Scalar(5)) * SE2Type::rot(Scalar(0.5));
  map1 = copy;
  EXPECT_EQ(map1.matrix(), copy.matrix()) << "type -> map assignment";
}

TYPED_TEST(SE2Test, SetRotationMatrix) {
  using Scalar = TypeParam;
  using SE2Type = SE2<Scalar>;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();

  SO2<Scalar> const R(Scalar(0.2));
  SE2Type se2;
  se2.setRotationMatrix(R.matrix());
  EXPECT_THAT(se2.rotationMatrix(), IsNear(R.matrix(), kEpsilon));

  Vector4<Scalar> raw(Scalar(1), Scalar(0), Scalar(3), Scalar(1));
  Eigen::Map<SE2Type> map_of_se2(raw.data());
  map_of_se2.setRotationMatrix(R.matrix());
  EXPECT_THAT(map_of_se2.rotationMatrix(), IsNear(R.matrix(), kEpsilon));
}

TYPED_TEST(SE2Test, Constructors) {
  using Scalar = TypeParam;
  using SE2Type = SE2<Scalar>;
  using Point = typename SE2Type::Point;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();

  EXPECT_EQ(SE2Type().matrix(), Matrix3<Scalar>::Identity().eval());

  SE2Type const se2 = LieGroupTestData<SE2Type>::groupElements().front();
  Point const translation = se2.translation();
  SO2<Scalar> const so2 = se2.so2();

  EXPECT_THAT(SE2Type(so2.log(), translation), IsNear(se2, kEpsilon));
  EXPECT_THAT(SE2Type(so2, translation), IsNear(se2, kEpsilon));
  EXPECT_THAT(SE2Type(so2.matrix(), translation), IsNear(se2, kEpsilon));
  EXPECT_THAT(SE2Type(so2.unit_complex(), translation), IsNear(se2, kEpsilon));
  EXPECT_THAT(SE2Type(se2.matrix()), IsNear(se2, kEpsilon));
}

template <class Scalar>
class SE2FloatingPointTest : public ::testing::Test {};

TYPED_TEST_SUITE(SE2FloatingPointTest, FloatingPointScalars,
                 ScalarNameGenerator);

TYPED_TEST(SE2FloatingPointTest, FitToSE2IsIdempotent) {
  using Scalar = TypeParam;
  using SE2Type = SE2<Scalar>;
  int constexpr kNumSamples = 100;

  auto const fitted =
      mapped(randomMatrices<Matrix3<Scalar>>(kNumSamples),
             [](Matrix3<Scalar> const& M) { return SE2Type::fitToSE2(M); });
  auto const refitted = mapped(
      fitted, [](SE2Type const& g) { return SE2Type::fitToSE2(g.matrix()); });
  EXPECT_THAT(refitted, ::testing::Pointwise(
                            IsNear(Constants<Scalar>::epsilon()), fitted));
}

TYPED_TEST(SE2FloatingPointTest, LogMatchesReference) {
  reference::expectAccurateLog<SE2<TypeParam>>(reference::kSE2Log);
}

TYPED_TEST(SE2FloatingPointTest, ExpMatchesReference) {
  reference::expectAccurateExp<SE2<TypeParam>>(reference::kSE2Exp);
}

TYPED_TEST(SE2FloatingPointTest, DxExpXMatchesReference) {
  reference::expectAccurateDxExpX<SE2<TypeParam>>(reference::kSE2DxExpX);
}

}  // namespace
}  // namespace Sophus
