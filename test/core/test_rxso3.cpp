// Copyright 2022, 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#include <cmath>
#include <limits>
#include <optional>
#include <random>
#include <tuple>

#include <sophus/rxso3.hpp>

#include "reference_rxso3.hpp"
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::RxSO3<double>>;
template class Map<Sophus::RxSO3<double> const>;
}  // namespace Eigen

namespace Sophus {

template class RxSO3<double, Eigen::AutoAlign>;
template class RxSO3<float, Eigen::DontAlign>;
#if SOPHUS_CERES
template class RxSO3<ceres::Jet<double, 3>>;
#endif

template <class Scalar>
struct LieGroupTestData<RxSO3<Scalar>> {
  using RxSO3Type = RxSO3<Scalar>;
  using Point = typename RxSO3Type::Point;
  using Tangent = typename RxSO3Type::Tangent;

  static StdVector<RxSO3Type> groupElements() {
    Scalar const kPi = Constants<Scalar>::pi();
    StdVector<RxSO3Type> rxso3_vec;
    rxso3_vec.push_back(RxSO3Type::exp(
        Tangent(Scalar(0.2), Scalar(0.5), Scalar(0.0), Scalar(1.))));
    rxso3_vec.push_back(RxSO3Type::exp(
        Tangent(Scalar(0.2), Scalar(0.5), Scalar(-1.0), Scalar(1.1))));
    rxso3_vec.push_back(RxSO3Type::exp(
        Tangent(Scalar(0.), Scalar(0.), Scalar(0.), Scalar(1.1))));
    rxso3_vec.push_back(RxSO3Type::exp(
        Tangent(Scalar(0.), Scalar(0.), Scalar(0.00001), Scalar(0.))));
    rxso3_vec.push_back(RxSO3Type::exp(
        Tangent(Scalar(0.), Scalar(0.), Scalar(0.00001), Scalar(0.00001))));
    rxso3_vec.push_back(RxSO3Type::exp(
        Tangent(Scalar(0.), Scalar(0.), Scalar(0.00001), Scalar(0))));
    rxso3_vec.push_back(
        RxSO3Type::exp(Tangent(kPi, Scalar(0), Scalar(0), Scalar(0.9))));
    rxso3_vec.push_back(
        RxSO3Type::exp(
            Tangent(Scalar(0.2), Scalar(-0.5), Scalar(0), Scalar(0))) *
        RxSO3Type::exp(Tangent(kPi, Scalar(0), Scalar(0), Scalar(0))) *
        RxSO3Type::exp(
            Tangent(-Scalar(0.2), Scalar(-0.5), Scalar(0), Scalar(0))));
    rxso3_vec.push_back(
        RxSO3Type::exp(
            Tangent(Scalar(0.3), Scalar(0.5), Scalar(0.1), Scalar(0))) *
        RxSO3Type::exp(Tangent(kPi, Scalar(0), Scalar(0), Scalar(0))) *
        RxSO3Type::exp(
            Tangent(Scalar(-0.3), Scalar(-0.5), Scalar(-0.1), Scalar(0))));
    return rxso3_vec;
  }

  static StdVector<Tangent> tangents() {
    StdVector<Tangent> tangent_vec;
    tangent_vec.push_back(Tangent(Scalar(0), Scalar(0), Scalar(0), Scalar(0)));
    tangent_vec.push_back(Tangent(Scalar(1), Scalar(0), Scalar(0), Scalar(0)));
    tangent_vec.push_back(
        Tangent(Scalar(1), Scalar(0), Scalar(0), Scalar(0.1)));
    tangent_vec.push_back(
        Tangent(Scalar(0), Scalar(1), Scalar(0), Scalar(0.1)));
    tangent_vec.push_back(
        Tangent(Scalar(0), Scalar(0), Scalar(1), Scalar(-0.1)));
    tangent_vec.push_back(
        Tangent(Scalar(-1), Scalar(1), Scalar(0), Scalar(-0.1)));
    tangent_vec.push_back(
        Tangent(Scalar(20), Scalar(-1), Scalar(0), Scalar(2)));
    return tangent_vec;
  }

  static StdVector<Point> points() {
    StdVector<Point> point_vec;
    point_vec.push_back(Point(Scalar(1), Scalar(2), Scalar(4)));
    point_vec.push_back(Point(Scalar(1), Scalar(-3), Scalar(0.5)));
    point_vec.push_back(Point(Scalar(-5), Scalar(-6), Scalar(7)));
    return point_vec;
  }
};

namespace {

[[maybe_unused]] bool const kRegistered =
    registerLieGroupTests<RxSO3<double>, RxSO3<float>
#if SOPHUS_CERES
                          ,
                          RxSO3<ceres::Jet<double, 3>>
#endif
                          >("RxSO3");

/**
 * Returns the tolerance of a saturated scale `expected`. Saturated scales
 * differ by a factor of 1 + epsilon from the bounds of the class invariant,
 * which half of this margin still tells apart.
 */
template <class Scalar>
Scalar saturationTolerance(Scalar const& expected) {
  return expected * Constants<Scalar>::epsilon() / Scalar(2);
}

template <class Scalar>
class RxSO3Test : public ::testing::Test {};

using RxSO3Scalars = ::testing::Types<double, float
#if SOPHUS_CERES
                                      ,
                                      ceres::Jet<double, 3>
#endif
                                      >;
TYPED_TEST_SUITE(RxSO3Test, RxSO3Scalars, ScalarNameGenerator);

TYPED_TEST(RxSO3Test, ProductOfSmallElementsSaturatesTheScale) {
  using Scalar = TypeParam;
  using RxSO3Type = RxSO3<Scalar>;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();

  RxSO3Type const small1(kEpsilon, SO3<Scalar>());
  RxSO3Type const small2(
      kEpsilon, SO3<Scalar>::exp(Vector3<Scalar>(Constants<Scalar>::pi(),
                                                 Scalar(0), Scalar(0))));
  RxSO3Type const saturated_product = small1 * small2;
  Scalar const kEpsilonPlus = Constants<Scalar>::epsilonPlus();
  EXPECT_THAT(saturated_product.scale(),
              IsNear(kEpsilonPlus, saturationTolerance(kEpsilonPlus)));
  EXPECT_THAT(saturated_product.so3(),
              IsNear(small1.so3() * small2.so3(), kEpsilon));
}

TYPED_TEST(RxSO3Test, ExpOfLargeScaleCanBeMultiplied) {
  using Scalar = TypeParam;
  using RxSO3Type = RxSO3<Scalar>;
  using Tangent = typename RxSO3Type::Tangent;

  Tangent const large_log(Scalar(1.), Scalar(2.), Scalar(3.),
                          std::numeric_limits<Scalar>::max());
  Tangent const regular_log(Scalar(4.), Scalar(5.), Scalar(6.), Scalar(0.));
  RxSO3Type const large = RxSO3Type::exp(large_log);
  RxSO3Type const regular = RxSO3Type::exp(regular_log);
  RxSO3Type const product = regular * large;
  EXPECT_THAT(large.scale(), IsFinite());
  EXPECT_THAT(product.scale(), IsFinite());
}

TYPED_TEST(RxSO3Test, ExpSaturatesTheScaleDespiteRoundingOfTheRotation) {
  using Scalar = TypeParam;
  using RxSO3Type = RxSO3<Scalar>;
  using SO3Type = SO3<Scalar>;
  using Tangent = typename RxSO3Type::Tangent;
  using std::log;
  using std::sqrt;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();
  Scalar const kEpsilonPlus = Constants<Scalar>::epsilonPlus();
  int constexpr kMaxSamples = 1000;

  // Looks for a rotation whose unit quaternion, scaled by the square root of
  // epsilon, rounds to a scale below epsilon, such that saturating the scale
  // at epsilon itself would violate the class invariant. The tangent is
  // sampled in double and converted to Scalar for compatibility with
  // ceres::Jet.
  std::mt19937 rng;
  std::uniform_real_distribution<double> uniform(-1., 1.);
  std::optional<typename SO3Type::Tangent> rounding_tangent;
  for (int i = 0; i < kMaxSamples && !rounding_tangent; ++i) {
    typename SO3Type::Tangent const so3_tangent(
        Scalar(uniform(rng)), Scalar(uniform(rng)), Scalar(uniform(rng)));
    Vector4<Scalar> const scaled =
        SO3Type::exp(so3_tangent).unit_quaternion().coeffs() * sqrt(kEpsilon);
    if (scaled.squaredNorm() < kEpsilon) {
      rounding_tangent = so3_tangent;
    }
  }
  ASSERT_TRUE(rounding_tangent.has_value())
      << "no rounding rotation among " << kMaxSamples << " samples";

  Tangent small_log;
  small_log << *rounding_tangent, log(kEpsilon / Scalar(2.));
  RxSO3Type const small_exp = RxSO3Type::exp(small_log);
  EXPECT_THAT(small_exp.scale(),
              IsNear(kEpsilonPlus, saturationTolerance(kEpsilonPlus)));
}

TYPED_TEST(RxSO3Test, MapsAccessTheUnderlyingData) {
  using Scalar = TypeParam;
  using RxSO3Type = RxSO3<Scalar>;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();

  Vector4<Scalar> raw(Scalar(0), Scalar(1), Scalar(0), Scalar(0));
  Eigen::Map<RxSO3Type const> const map_of_const_rxso3(raw.data());
  EXPECT_THAT(map_of_const_rxso3.quaternion().coeffs(), IsNear(raw, kEpsilon));
  EXPECT_EQ(map_of_const_rxso3.quaternion().coeffs().data(), raw.data());
  Eigen::Map<RxSO3Type const> const const_shallow_copy = map_of_const_rxso3;
  EXPECT_EQ(const_shallow_copy.quaternion().coeffs().eval(),
            map_of_const_rxso3.quaternion().coeffs().eval());

  Vector4<Scalar> const raw2(Scalar(1), Scalar(0), Scalar(0), Scalar(0));
  Eigen::Map<RxSO3Type> map_of_rxso3(raw.data());
  Eigen::Quaternion<Scalar> quat;
  quat.coeffs() = raw2;
  map_of_rxso3.setQuaternion(quat);
  EXPECT_THAT(map_of_rxso3.quaternion().coeffs(), IsNear(raw2, kEpsilon));
  EXPECT_EQ(map_of_rxso3.quaternion().coeffs().data(), raw.data());
  EXPECT_NE(map_of_rxso3.quaternion().coeffs().data(), quat.coeffs().data());
  Eigen::Map<RxSO3Type> const shallow_copy = map_of_rxso3;
  EXPECT_EQ(shallow_copy.quaternion().coeffs().eval(),
            map_of_rxso3.quaternion().coeffs().eval());
}

TYPED_TEST(RxSO3Test, DataAccessesTheParameters) {
  using Scalar = TypeParam;
  using RxSO3Type = RxSO3<Scalar>;

  Vector4<Scalar> const raw(Scalar(0), Scalar(1), Scalar(0), Scalar(0));
  Vector4<Scalar> const raw2(Scalar(1), Scalar(0), Scalar(0), Scalar(0));
  Eigen::Quaternion<Scalar> quat;
  quat.coeffs() = raw2;

  RxSO3Type const const_rxso3(quat);
  EXPECT_EQ(rawParams(const_rxso3), raw2);

  RxSO3Type rxso3(quat);
  Eigen::Map<Vector4<Scalar>>(rxso3.data()) = raw;
  EXPECT_EQ(rawParams(rxso3), raw);
}

TYPED_TEST(RxSO3Test, RotationMatrixLeavesTheScaleOfMapsAsIs) {
  using Scalar = TypeParam;
  using RxSO3Type = RxSO3<Scalar>;
  using RotationMatrixType = typename SO3<Scalar>::Transformation;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();

  // rotationMatrix() must not change the underlying value of a map, and must
  // compile for a map of const.
  Vector4<Scalar> raw3(Scalar(2), Scalar(0), Scalar(0), Scalar(0));
  Eigen::Map<RxSO3Type> const map_of_rxso3_3(raw3.data());
  Eigen::Map<RxSO3Type const> const const_map_of_rxso3_3(raw3.data());
  RxSO3Type const rxso3_copy3 = map_of_rxso3_3;
  RotationMatrixType const r_ref = map_of_rxso3_3.so3().matrix();

  RotationMatrixType const r = map_of_rxso3_3.rotationMatrix();
  EXPECT_THAT(r, IsNear(r_ref, kEpsilon));
  EXPECT_THAT(map_of_rxso3_3.quaternion().coeffs(),
              IsNear(rxso3_copy3.quaternion().coeffs(), kEpsilon));

  RotationMatrixType const r_const = const_map_of_rxso3_3.rotationMatrix();
  EXPECT_THAT(r_const, IsNear(r_ref, kEpsilon));
  EXPECT_THAT(const_map_of_rxso3_3.quaternion().coeffs(),
              IsNear(rxso3_copy3.quaternion().coeffs(), kEpsilon));
}

TYPED_TEST(RxSO3Test, AssignsBetweenMapsAndElements) {
  using Scalar = TypeParam;
  using RxSO3Type = RxSO3<Scalar>;
  using Tangent = typename RxSO3Type::Tangent;

  Vector4<Scalar> data1(Scalar(.1), Scalar(.2), Scalar(.3), Scalar(.4));
  Vector4<Scalar> data2(Scalar(.5), Scalar(.4), Scalar(.3), Scalar(.2));
  Eigen::Map<RxSO3Type> map1(data1.data());
  Eigen::Map<RxSO3Type> map2(data2.data());

  map2 = map1;
  EXPECT_EQ(map2.matrix(), map1.matrix()) << "map -> map assignment";

  RxSO3Type copy;
  copy = map1;
  EXPECT_EQ(copy.matrix(), map1.matrix()) << "map -> type assignment";

  copy = RxSO3Type::exp(
      Tangent(Scalar(0.2), Scalar(0.5), Scalar(-1.0), Scalar(1.1)));
  map1 = copy;
  EXPECT_EQ(map1.matrix(), copy.matrix()) << "type -> map assignment";
}

TYPED_TEST(RxSO3Test, ConstructorsAndSetters) {
  using Scalar = TypeParam;
  using RxSO3Type = RxSO3<Scalar>;
  using Point = typename RxSO3Type::Point;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();

  RxSO3Type rxso3;
  Scalar const scale = Scalar(1.2);
  rxso3.setScale(scale);
  EXPECT_THAT(rxso3.scale(), IsNear(scale, kEpsilon)) << "setScale";
  SO3<Scalar> const so3 =
      LieGroupTestData<RxSO3Type>::groupElements().front().so3();
  rxso3.setSO3(so3);
  EXPECT_THAT(rxso3.scale(), IsNear(scale, kEpsilon))
      << "setSO3 leaves the scale as is";
  EXPECT_THAT(RxSO3Type(scale, so3), IsNear(rxso3, kEpsilon))
      << "RxSO3(scale, SO3)";
  EXPECT_THAT(RxSO3Type(scale, so3.matrix()), IsNear(rxso3, kEpsilon))
      << "RxSO3(scale, rotation matrix)";
  Eigen::Quaternion<Scalar> const q = rxso3.quaternion();
  EXPECT_THAT(RxSO3Type(scale, q),
              IsNear(RxSO3Type(scale, SO3<Scalar>(q)), kEpsilon))
      << "RxSO3(scale, unit quaternion)";

  Matrix3<Scalar> const R =
      SO3<Scalar>::exp(Point(Scalar(0.2), Scalar(0.5), Scalar(-1.0))).matrix();
  Matrix3<Scalar> const sR = R * Scalar(1.3);
  EXPECT_THAT(RxSO3Type(sR).matrix(), IsNear(sR, kEpsilon)) << "RxSO3(sR)";
  rxso3.setScaledRotationMatrix(sR);
  EXPECT_THAT(rxso3.matrix(), IsNear(sR, kEpsilon))
      << "setScaledRotationMatrix";
  rxso3.setScale(scale);
  rxso3.setRotationMatrix(R);
  EXPECT_THAT(rxso3.rotationMatrix(), IsNear(R, kEpsilon))
      << "setRotationMatrix";
  EXPECT_THAT(rxso3.scale(), IsNear(scale, kEpsilon))
      << "setRotationMatrix leaves the scale as is";
}

template <class Scalar>
class RxSO3FloatingPointTest : public ::testing::Test {};

TYPED_TEST_SUITE(RxSO3FloatingPointTest, FloatingPointScalars,
                 ScalarNameGenerator);

TYPED_TEST(RxSO3FloatingPointTest, ScaledRotationsAreScaledOrthogonal) {
  using Scalar = TypeParam;
  using ::testing::Each;
  using ::testing::Not;
  int constexpr kNumSamples = 10;

  StdVector<Matrix3<Scalar>> scaled_rotations;
  for (Matrix3<Scalar> const& M :
       randomMatrices<Matrix3<Scalar>>(kNumSamples)) {
    Matrix3<Scalar> const R = makeRotationMatrix(M);
    for (Scalar const scale :
         {Scalar(0.01), Scalar(0.99), Scalar(1), Scalar(10)}) {
      scaled_rotations.push_back(scale * R);
    }
  }
  auto const reflections =
      mapped(scaled_rotations, [](Matrix3<Scalar> const& sR) {
        Matrix3<Scalar> sR_cols_swapped;
        sR_cols_swapped << sR.col(1), sR.col(0), sR.col(2);
        return sR_cols_swapped;
      });

  EXPECT_THAT(scaled_rotations, Each(IsScaledOrthogonalAndPositive()));
  EXPECT_THAT(reflections, Each(Not(IsScaledOrthogonalAndPositive())));
}

TYPED_TEST(RxSO3FloatingPointTest, ExpMatchesReference) {
  reference::expectAccurateExp<RxSO3<TypeParam>>(reference::kRxSO3Exp);
}

TYPED_TEST(RxSO3FloatingPointTest, DxExpXMatchesReference) {
  reference::expectAccurateDxExpX<RxSO3<TypeParam>>(reference::kRxSO3DxExpX);
}

TEST(RxSO3DeathTest, RejectsScaledReflections) {
  // A reflection scaled by 1/2 is scaled orthogonal, but has a negative
  // determinant.
  Matrix3<double> const scaled_reflection =
      Vector3<double>(0.5, 0.5, -0.5).asDiagonal();
  EXPECT_DEATH(std::ignore = RxSO3d{scaled_reflection}, "scaled orthogonal");
}

TYPED_TEST(RxSO3FloatingPointTest, LogMatchesReference) {
  reference::expectAccurateLog<RxSO3<TypeParam>>(reference::kRxSO3Log);
}

TEST(RxSO3DeathTest, SetQuaternionRejectsScalesBelowEpsilon) {
  RxSO3d T;
  // The scale |q|^2 = 1e-15 is below the epsilon of the class invariant.
  Eigen::Quaterniond const tiny(std::sqrt(1e-15), 0, 0, 0);
  EXPECT_DEATH(T.setQuaternion(tiny), "Scale factor");
}

TYPED_TEST(RxSO3FloatingPointTest, DxExpXFollowsTheScaleSaturation) {
  using Scalar = TypeParam;
  using RxSO3Type = RxSO3<Scalar>;
  using Tangent = typename RxSO3Type::Tangent;
  using Jacobian = Matrix<Scalar, RxSO3Type::num_parameters, RxSO3Type::DoF>;
  // exp saturates the scale beyond these log scales, where its derivative
  // with respect to the log scale vanishes.
  for (double const sigma : {-30.0, 30.0}) {
    Tangent const x(Scalar(0.3), Scalar(-0.2), Scalar(0.1), Scalar(sigma));
    Jacobian const J_num =
        vectorFieldNumDiff<Scalar, RxSO3Type::num_parameters, RxSO3Type::DoF>(
            [](Tangent const& t) { return RxSO3Type::exp(t).params(); }, x);
    Jacobian const J = RxSO3Type::Dx_exp_x(x);
    // The Jacobian is proportional to the square root of the saturated scale,
    // which is far from one, so the tolerance is relative to its largest
    // entry.
    EXPECT_THAT(J, IsNear(J_num, Constants<Scalar>::epsilonSqrt() *
                                     J_num.cwiseAbs().maxCoeff()))
        << "sigma = " << sigma;
    EXPECT_THAT(J.col(RxSO3Type::DoF - 1),
                IsNear(Vector4<Scalar>::Zero().eval(), Scalar(0)))
        << "sigma = " << sigma;
  }
}

TEST(RxSO3DeathTest, SetScaleRejectsScalesOutOfRange) {
  RxSO3d T;
  EXPECT_DEATH(T.setScale(1e-15), "Scale factor 1e-15");
  EXPECT_DEATH(T.setScale(1e15), "Scale factor 1e\\+15");
}

}  // namespace
}  // namespace Sophus
