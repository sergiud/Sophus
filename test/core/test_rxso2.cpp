// Copyright 2022, 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#include <cmath>
#include <limits>
#include <random>

#include <sophus/rxso2.hpp>

#include "reference_rxso2.hpp"
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::RxSO2<double>>;
template class Map<Sophus::RxSO2<double> const>;
}  // namespace Eigen

namespace Sophus {

template class RxSO2<double, Eigen::AutoAlign>;
template class RxSO2<float, Eigen::DontAlign>;
#if SOPHUS_CERES
template class RxSO2<ceres::Jet<double, 3>>;
#endif

template <class Scalar>
struct LieGroupTestData<RxSO2<Scalar>> {
  using RxSO2Type = RxSO2<Scalar>;
  using Point = typename RxSO2Type::Point;
  using Tangent = typename RxSO2Type::Tangent;

  static StdVector<RxSO2Type> groupElements() {
    Scalar const kPi = Constants<Scalar>::pi();
    StdVector<RxSO2Type> rxso2_vec;
    rxso2_vec.push_back(RxSO2Type::exp(Tangent(Scalar(0.2), Scalar(1.))));
    rxso2_vec.push_back(RxSO2Type::exp(Tangent(Scalar(0.2), Scalar(1.1))));
    rxso2_vec.push_back(RxSO2Type::exp(Tangent(Scalar(0.), Scalar(1.1))));
    rxso2_vec.push_back(RxSO2Type::exp(Tangent(Scalar(0.00001), Scalar(0.))));
    rxso2_vec.push_back(
        RxSO2Type::exp(Tangent(Scalar(0.00001), Scalar(0.00001))));
    rxso2_vec.push_back(RxSO2Type::exp(Tangent(kPi, Scalar(0.9))));
    rxso2_vec.push_back(RxSO2Type::exp(Tangent(Scalar(0.2), Scalar(0))) *
                        RxSO2Type::exp(Tangent(kPi, Scalar(0.0))) *
                        RxSO2Type::exp(Tangent(Scalar(-0.2), Scalar(0))));
    rxso2_vec.push_back(RxSO2Type::exp(Tangent(Scalar(0.3), Scalar(0))) *
                        RxSO2Type::exp(Tangent(kPi, Scalar(0.001))) *
                        RxSO2Type::exp(Tangent(Scalar(-0.3), Scalar(0))));
    return rxso2_vec;
  }

  static StdVector<Tangent> tangents() {
    StdVector<Tangent> tangent_vec;
    tangent_vec.push_back(Tangent(Scalar(0), Scalar(0)));
    tangent_vec.push_back(Tangent(Scalar(1), Scalar(0)));
    tangent_vec.push_back(Tangent(Scalar(1), Scalar(0.1)));
    tangent_vec.push_back(Tangent(Scalar(0), Scalar(0.1)));
    tangent_vec.push_back(Tangent(Scalar(0), Scalar(-0.1)));
    tangent_vec.push_back(Tangent(Scalar(-1), Scalar(-0.1)));
    tangent_vec.push_back(Tangent(Scalar(20), Scalar(2)));
    return tangent_vec;
  }

  static StdVector<Point> points() {
    StdVector<Point> point_vec;
    point_vec.push_back(Point(Scalar(1), Scalar(4)));
    point_vec.push_back(Point(Scalar(1), Scalar(-3)));
    return point_vec;
  }
};

namespace {

[[maybe_unused]] bool const kRegistered =
    registerLieGroupTests<RxSO2<double>, RxSO2<float>
#if SOPHUS_CERES
                          ,
                          RxSO2<ceres::Jet<double, 3>>
#endif
                          >("RxSO2");

template <class Scalar>
class RxSO2Test : public ::testing::Test {};

using RxSO2Scalars = ::testing::Types<double, float
#if SOPHUS_CERES
                                      ,
                                      ceres::Jet<double, 3>
#endif
                                      >;
TYPED_TEST_SUITE(RxSO2Test, RxSO2Scalars, ScalarNameGenerator);

TYPED_TEST(RxSO2Test, ProductOfSmallElementsSaturatesTheScale) {
  using Scalar = TypeParam;
  using RxSO2Type = RxSO2<Scalar>;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();

  RxSO2Type const small1(Scalar(1.1) * kEpsilon, SO2<Scalar>());
  RxSO2Type const small2(Scalar(1.1) * kEpsilon,
                         SO2<Scalar>::exp(Constants<Scalar>::pi()));
  RxSO2Type const saturated_product = small1 * small2;
  EXPECT_THAT(saturated_product.scale(), IsNear(kEpsilon, kEpsilon));
  EXPECT_THAT(saturated_product.so2(),
              IsNear(small1.so2() * small2.so2(), kEpsilon));
}

TYPED_TEST(RxSO2Test, ExpOfLargeScaleCanBeMultiplied) {
  using Scalar = TypeParam;
  using RxSO2Type = RxSO2<Scalar>;
  using Tangent = typename RxSO2Type::Tangent;

  Tangent const large_log(Scalar(1.), std::numeric_limits<Scalar>::max());
  Tangent const regular_log(Scalar(2.), Scalar(0.));
  RxSO2Type const large = RxSO2Type::exp(large_log);
  RxSO2Type const regular = RxSO2Type::exp(regular_log);
  RxSO2Type const product = regular * large;
  EXPECT_THAT(large.scale(), IsFinite());
  EXPECT_THAT(product.scale(), IsFinite());
}

TYPED_TEST(RxSO2Test, ExpSaturatesTheScaleDespiteRoundingOfTheRotation) {
  using Scalar = TypeParam;
  using RxSO2Type = RxSO2<Scalar>;
  using Tangent = typename RxSO2Type::Tangent;
  using std::cos;
  using std::log;
  using std::sin;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();

  // Looks for an angle whose cosine and sine have a squared norm below 1
  // after rounding. The angle is sampled in double and converted to Scalar
  // for compatibility with ceres::Jet.
  std::mt19937 rng;
  std::uniform_real_distribution<double> uniform(0., Constants<double>::pi());
  Tangent small_log;
  while (true) {
    Scalar const phi = Scalar(uniform(rng));
    Scalar const c = cos(phi);
    Scalar const s = sin(phi);
    if (c * c + s * s < Scalar(1.)) {
      small_log[0] = phi;
      break;
    }
  }
  small_log[1] = log(kEpsilon / Scalar(2.));

  RxSO2Type const small_exp = RxSO2Type::exp(small_log);
  EXPECT_THAT(small_exp.scale(), IsNear(kEpsilon, kEpsilon));
}

TYPED_TEST(RxSO2Test, MapsAccessTheUnderlyingData) {
  using Scalar = TypeParam;
  using RxSO2Type = RxSO2<Scalar>;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();

  Vector2<Scalar> const raw(Scalar(0), Scalar(1));
  Eigen::Map<RxSO2Type const> const map_of_const_rxso2(raw.data());
  EXPECT_THAT(map_of_const_rxso2.complex(), IsNear(raw, kEpsilon));
  EXPECT_EQ(map_of_const_rxso2.complex().data(), raw.data());
  Eigen::Map<RxSO2Type const> const const_shallow_copy = map_of_const_rxso2;
  EXPECT_EQ(const_shallow_copy.complex().eval(),
            map_of_const_rxso2.complex().eval());

  Vector2<Scalar> raw2(Scalar(1), Scalar(0));
  Eigen::Map<RxSO2Type> map_of_rxso2(raw2.data());
  EXPECT_THAT(map_of_rxso2.complex(), IsNear(raw2, kEpsilon));
  EXPECT_EQ(map_of_rxso2.complex().data(), raw2.data());
  Eigen::Map<RxSO2Type> const shallow_copy = map_of_rxso2;
  EXPECT_EQ(shallow_copy.complex().eval(), map_of_rxso2.complex().eval());
}

TYPED_TEST(RxSO2Test, DataAccessesTheParameters) {
  using Scalar = TypeParam;
  using RxSO2Type = RxSO2<Scalar>;

  Vector2<Scalar> const raw(Scalar(0), Scalar(1));
  Vector2<Scalar> const raw2(Scalar(1), Scalar(0));
  RxSO2Type const const_rxso2(raw2);
  EXPECT_EQ(rawParams(const_rxso2), raw2);

  RxSO2Type rxso2(raw2);
  Eigen::Map<Vector2<Scalar>>(rxso2.data()) = raw;
  EXPECT_EQ(rawParams(rxso2), raw);
}

TYPED_TEST(RxSO2Test, RotationMatrixLeavesTheScaleOfMapsAsIs) {
  using Scalar = TypeParam;
  using RxSO2Type = RxSO2<Scalar>;
  using RotationMatrixType = typename SO2<Scalar>::Transformation;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();

  // rotationMatrix() must not change the underlying value of a map, and must
  // compile for a map of const.
  Vector2<Scalar> raw3(Scalar(2), Scalar(0));
  Eigen::Map<RxSO2Type> const map_of_rxso2_3(raw3.data());
  Eigen::Map<RxSO2Type const> const const_map_of_rxso2_3(raw3.data());
  RxSO2Type const rxso2_copy3 = map_of_rxso2_3;
  RotationMatrixType const r_ref = map_of_rxso2_3.so2().matrix();

  RotationMatrixType const r = map_of_rxso2_3.rotationMatrix();
  EXPECT_THAT(r, IsNear(r_ref, kEpsilon));
  EXPECT_THAT(map_of_rxso2_3.complex(),
              IsNear(rxso2_copy3.complex(), kEpsilon));

  RotationMatrixType const r_const = const_map_of_rxso2_3.rotationMatrix();
  EXPECT_THAT(r_const, IsNear(r_ref, kEpsilon));
  EXPECT_THAT(const_map_of_rxso2_3.complex(),
              IsNear(rxso2_copy3.complex(), kEpsilon));
}

TYPED_TEST(RxSO2Test, AssignsBetweenMapsAndElements) {
  using Scalar = TypeParam;
  using RxSO2Type = RxSO2<Scalar>;
  using Tangent = typename RxSO2Type::Tangent;

  Vector2<Scalar> data1(Scalar(.1), Scalar(.2));
  Vector2<Scalar> data2(Scalar(.5), Scalar(.4));
  Eigen::Map<RxSO2Type> map1(data1.data());
  Eigen::Map<RxSO2Type> map2(data2.data());

  map2 = map1;
  EXPECT_EQ(map2.matrix(), map1.matrix()) << "map -> map assignment";

  RxSO2Type copy;
  copy = map1;
  EXPECT_EQ(copy.matrix(), map1.matrix()) << "map -> type assignment";

  copy = RxSO2Type::exp(Tangent(Scalar(0.2), Scalar(0.5)));
  map1 = copy;
  EXPECT_EQ(map1.matrix(), copy.matrix()) << "type -> map assignment";
}

TYPED_TEST(RxSO2Test, ConstructorsAndSetters) {
  using Scalar = TypeParam;
  using RxSO2Type = RxSO2<Scalar>;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();

  RxSO2Type rxso2;
  Scalar const scale(1.2);
  rxso2.setScale(scale);
  EXPECT_THAT(rxso2.scale(), IsNear(scale, kEpsilon)) << "setScale";
  Scalar const angle(0.2);
  rxso2.setAngle(angle);
  EXPECT_THAT(rxso2.angle(), IsNear(angle, kEpsilon)) << "setAngle";
  EXPECT_THAT(rxso2.scale(), IsNear(scale, kEpsilon))
      << "setAngle leaves the scale as is";

  SO2<Scalar> const so2 =
      LieGroupTestData<RxSO2Type>::groupElements().front().so2();
  rxso2.setSO2(so2);
  EXPECT_THAT(rxso2.scale(), IsNear(scale, kEpsilon))
      << "setSO2 leaves the scale as is";
  EXPECT_THAT(RxSO2Type(scale, so2), IsNear(rxso2, kEpsilon))
      << "RxSO2(scale, SO2)";
  EXPECT_THAT(RxSO2Type(scale, so2.matrix()), IsNear(rxso2, kEpsilon))
      << "RxSO2(scale, rotation matrix)";

  Matrix2<Scalar> const R = SO2<Scalar>::exp(Scalar(0.2)).matrix();
  Matrix2<Scalar> const sR = R * Scalar(1.3);
  EXPECT_THAT(RxSO2Type(sR).matrix(), IsNear(sR, kEpsilon)) << "RxSO2(sR)";
  rxso2.setScaledRotationMatrix(sR);
  EXPECT_THAT(rxso2.matrix(), IsNear(sR, kEpsilon))
      << "setScaledRotationMatrix";
  rxso2.setScale(scale);
  rxso2.setRotationMatrix(R);
  EXPECT_THAT(rxso2.rotationMatrix(), IsNear(R, kEpsilon))
      << "setRotationMatrix";
  EXPECT_THAT(rxso2.scale(), IsNear(scale, kEpsilon))
      << "setRotationMatrix leaves the scale as is";
}

template <class Scalar>
class RxSO2FloatingPointTest : public ::testing::Test {};

TYPED_TEST_SUITE(RxSO2FloatingPointTest, FloatingPointScalars,
                 ScalarNameGenerator);

TYPED_TEST(RxSO2FloatingPointTest, ScaledRotationsAreScaledOrthogonal) {
  using Scalar = TypeParam;
  using ::testing::Each;
  using ::testing::Not;
  int constexpr kNumSamples = 10;

  StdVector<Matrix2<Scalar>> scaled_rotations;
  for (Matrix2<Scalar> const& M :
       randomMatrices<Matrix2<Scalar>>(kNumSamples)) {
    Matrix2<Scalar> const R = makeRotationMatrix(M);
    for (Scalar const scale :
         {Scalar(0.01), Scalar(0.99), Scalar(1), Scalar(10)}) {
      scaled_rotations.push_back(scale * R);
    }
  }
  auto const reflections =
      mapped(scaled_rotations, [](Matrix2<Scalar> const& sR) {
        Matrix2<Scalar> sR_cols_swapped;
        sR_cols_swapped << sR.col(1), sR.col(0);
        return sR_cols_swapped;
      });

  EXPECT_THAT(scaled_rotations, Each(IsScaledOrthogonalAndPositive()));
  EXPECT_THAT(reflections, Each(Not(IsScaledOrthogonalAndPositive())));
}

TYPED_TEST(RxSO2FloatingPointTest, ExpMatchesReference) {
  reference::expectAccurateExp<RxSO2<TypeParam>>(reference::kRxSO2Exp);
}

TYPED_TEST(RxSO2FloatingPointTest, DxExpXMatchesReference) {
  reference::expectAccurateDxExpX<RxSO2<TypeParam>>(reference::kRxSO2DxExpX);
}

TYPED_TEST(RxSO2FloatingPointTest, LogMatchesReference) {
  reference::expectAccurateLog<RxSO2<TypeParam>>(reference::kRxSO2Log);
}

}  // namespace
}  // namespace Sophus
