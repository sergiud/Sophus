// Copyright 2022, 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#include <sophus/so2.hpp>

#include "reference_so2.hpp"
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::SO2<double>>;
template class Map<Sophus::SO2<double> const>;
}  // namespace Eigen

namespace Sophus {

template class SO2<double, Eigen::AutoAlign>;
template class SO2<float, Eigen::DontAlign>;
#if SOPHUS_CERES
template class SO2<ceres::Jet<double, 3>>;
#endif

template <class Scalar>
struct LieGroupTestData<SO2<Scalar>> {
  using SO2Type = SO2<Scalar>;
  using Point = typename SO2Type::Point;
  using Tangent = typename SO2Type::Tangent;

  static StdVector<SO2Type> groupElements() {
    Scalar const kPi = Constants<Scalar>::pi();
    StdVector<SO2Type> so2_vec;
    so2_vec.push_back(SO2Type::exp(Scalar(0.0)));
    so2_vec.push_back(SO2Type::exp(Scalar(0.2)));
    so2_vec.push_back(SO2Type::exp(Scalar(10.)));
    so2_vec.push_back(SO2Type::exp(Scalar(0.00001)));
    so2_vec.push_back(SO2Type::exp(kPi));
    so2_vec.push_back(SO2Type::exp(Scalar(0.2)) * SO2Type::exp(kPi) *
                      SO2Type::exp(Scalar(-0.2)));
    so2_vec.push_back(SO2Type::exp(Scalar(-0.3)) * SO2Type::exp(kPi) *
                      SO2Type::exp(Scalar(0.3)));
    return so2_vec;
  }

  static StdVector<Tangent> tangents() {
    Scalar const kPi = Constants<Scalar>::pi();
    StdVector<Tangent> tangent_vec;
    tangent_vec.push_back(Tangent(Scalar(0)));
    tangent_vec.push_back(Tangent(Scalar(1)));
    tangent_vec.push_back(Tangent(Scalar(kPi / 2.)));
    tangent_vec.push_back(Tangent(Scalar(-1)));
    tangent_vec.push_back(Tangent(Scalar(20)));
    tangent_vec.push_back(Tangent(Scalar(kPi / 2. + 0.0001)));
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
    registerLieGroupTests<SO2<double>, SO2<float>
#if SOPHUS_CERES
                          ,
                          SO2<ceres::Jet<double, 3>>
#endif
                          >("SO2");

template <class Scalar>
class SO2Test : public ::testing::Test {};

using SO2Scalars = ::testing::Types<double, float
#if SOPHUS_CERES
                                    ,
                                    ceres::Jet<double, 3>
#endif
                                    >;
TYPED_TEST_SUITE(SO2Test, SO2Scalars, ScalarNameGenerator);

TYPED_TEST(SO2Test, RepeatedProductsKeepUnitMagnitude) {
  using Scalar = TypeParam;
  using SO2Type = SO2<Scalar>;
  int constexpr kNumRepetitions = 1000;

  StdVector<SO2Type> const so2_vec = LieGroupTestData<SO2Type>::groupElements();
  SO2Type current_q;
  for (int i = 0; i < kNumRepetitions; ++i) {
    for (SO2Type const& q : so2_vec) {
      current_q *= q;
    }
  }
  EXPECT_THAT(current_q.unit_complex().norm(),
              IsNear(Scalar(1), Constants<Scalar>::epsilon()));
}

TYPED_TEST(SO2Test, MapsAccessTheUnderlyingData) {
  using Scalar = TypeParam;
  using SO2Type = SO2<Scalar>;

  Vector2<Scalar> raw(Scalar(0), Scalar(1));
  Eigen::Map<SO2Type const> const map_of_const_so2(raw.data());
  EXPECT_THAT(map_of_const_so2.unit_complex(),
              IsNear(raw, Constants<Scalar>::epsilon()));
  EXPECT_EQ(map_of_const_so2.unit_complex().data(), raw.data());
  Eigen::Map<SO2Type const> const const_shallow_copy = map_of_const_so2;
  EXPECT_EQ(const_shallow_copy.unit_complex().eval(),
            map_of_const_so2.unit_complex().eval());

  Vector2<Scalar> const raw2(Scalar(1), Scalar(0));
  Eigen::Map<SO2Type> map_of_so2(raw.data());
  map_of_so2.setComplex(raw2);
  EXPECT_THAT(map_of_so2.unit_complex(),
              IsNear(raw2, Constants<Scalar>::epsilon()));
  EXPECT_EQ(map_of_so2.unit_complex().data(), raw.data());
  EXPECT_NE(map_of_so2.unit_complex().data(), raw2.data());
  Eigen::Map<SO2Type> const shallow_copy = map_of_so2;
  EXPECT_EQ(shallow_copy.unit_complex().eval(),
            map_of_so2.unit_complex().eval());
}

TYPED_TEST(SO2Test, DataAccessesTheParameters) {
  using Scalar = TypeParam;
  using SO2Type = SO2<Scalar>;

  Vector2<Scalar> const raw(Scalar(0), Scalar(1));
  Vector2<Scalar> const raw2(Scalar(1), Scalar(0));
  SO2Type const const_so2(raw2);
  EXPECT_EQ(rawParams(const_so2), raw2);

  SO2Type so2(raw2);
  Eigen::Map<Vector2<Scalar>>(so2.data()) = raw;
  EXPECT_EQ(rawParams(so2), raw);
}

TYPED_TEST(SO2Test, AssignsBetweenMapsAndElements) {
  using Scalar = TypeParam;
  using SO2Type = SO2<Scalar>;

  Vector2<Scalar> data1(Scalar(1), Scalar(0));
  Vector2<Scalar> data2(Scalar(0), Scalar(1));
  Eigen::Map<SO2Type> map1(data1.data());
  Eigen::Map<SO2Type> map2(data2.data());

  map2 = map1;
  EXPECT_EQ(map2.matrix(), map1.matrix()) << "map -> map assignment";

  SO2Type copy;
  copy = map1;
  EXPECT_EQ(copy.matrix(), map1.matrix()) << "map -> type assignment";

  copy = SO2Type(Scalar(0.5));
  map1 = copy;
  EXPECT_EQ(map1.matrix(), copy.matrix()) << "type -> map assignment";
}

template <class Scalar>
class SO2FloatingPointTest : public ::testing::Test {};

TYPED_TEST_SUITE(SO2FloatingPointTest, FloatingPointScalars,
                 ScalarNameGenerator);

TYPED_TEST(SO2FloatingPointTest, FitToSO2IsIdempotent) {
  using Scalar = TypeParam;
  using SO2Type = SO2<Scalar>;
  int constexpr kNumSamples = 100;

  auto const fitted =
      mapped(randomMatrices<Matrix2<Scalar>>(kNumSamples),
             [](Matrix2<Scalar> const& R) { return SO2Type::fitToSO2(R); });
  auto const refitted = mapped(fitted, [](SO2Type const& so2) {
    return SO2Type::fitToSO2(so2.matrix());
  });
  EXPECT_THAT(refitted, ::testing::Pointwise(
                            IsNear(Constants<Scalar>::epsilon()), fitted));
}

TYPED_TEST(SO2FloatingPointTest, FitToSO2RecoversScaledRotations) {
  using Scalar = TypeParam;
  using SO2Type = SO2<Scalar>;
  Scalar const kScale(2.5);

  StdVector<SO2Type> const elements =
      LieGroupTestData<SO2Type>::groupElements();
  auto const fitted = mapped(elements, [kScale](SO2Type const& g) {
    return SO2Type::fitToSO2(kScale * g.matrix());
  });
  EXPECT_THAT(fitted, ::testing::Pointwise(IsNear(Constants<Scalar>::epsilon()),
                                           elements));
}

TYPED_TEST(SO2FloatingPointTest, ExpMatchesReference) {
  reference::expectAccurateExp<SO2<TypeParam>>(reference::kSO2Exp);
}

TYPED_TEST(SO2FloatingPointTest, DxExpXMatchesReference) {
  reference::expectAccurateDxExpX<SO2<TypeParam>>(reference::kSO2DxExpX);
}

TYPED_TEST(SO2FloatingPointTest, LogMatchesReference) {
  reference::expectAccurateLog<SO2<TypeParam>>(reference::kSO2Log);
}

TYPED_TEST(SO2FloatingPointTest, CompoundMultiplicationReturnsTheElement) {
  using Scalar = TypeParam;
  using SO2Type = SO2<Scalar>;
  SO2Type const b = SO2Type::exp(Scalar(0.25));
  SO2Type const c = SO2Type::exp(Scalar(-0.75));
  SO2Type a = SO2Type::exp(Scalar(1));

  (a *= b) *= c;
  EXPECT_THAT(a.log(), IsNear(Scalar(0.5), Constants<Scalar>::epsilon()));
}

}  // namespace
}  // namespace Sophus
