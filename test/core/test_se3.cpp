// Copyright 2022, 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#include <sophus/se3.hpp>

#include "reference_se3.hpp"
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::SE3<double>>;
template class Map<Sophus::SE3<double> const>;
}  // namespace Eigen

namespace Sophus {

template class SE3<double, Eigen::AutoAlign>;
template class SE3<float, Eigen::DontAlign>;
#if SOPHUS_CERES
template class SE3<ceres::Jet<double, 3>>;
#endif

template <class Scalar>
struct LieGroupTestData<SE3<Scalar>> {
  using SE3Type = SE3<Scalar>;
  using Point = typename SE3Type::Point;
  using Tangent = typename SE3Type::Tangent;

  static StdVector<SE3Type> groupElements() { return getTestSE3s<Scalar>(); }

  static StdVector<Tangent> tangents() {
    StdVector<Tangent> tangent_vec;
    Tangent tmp;
    tmp << Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0);
    tangent_vec.push_back(tmp);
    tmp << Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0);
    tangent_vec.push_back(tmp);
    tmp << Scalar(0), Scalar(1), Scalar(0), Scalar(1), Scalar(0), Scalar(0);
    tangent_vec.push_back(tmp);
    tmp << Scalar(0), Scalar(-5), Scalar(10), Scalar(0), Scalar(0), Scalar(0);
    tangent_vec.push_back(tmp);
    tmp << Scalar(-1), Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(1);
    tangent_vec.push_back(tmp);
    tmp << Scalar(20), Scalar(-1), Scalar(0), Scalar(-1), Scalar(1), Scalar(0);
    tangent_vec.push_back(tmp);
    tmp << Scalar(30), Scalar(5), Scalar(-1), Scalar(20), Scalar(-1), Scalar(0);
    tangent_vec.push_back(tmp);
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
    registerLieGroupTests<SE3<double>, SE3<float>
#if SOPHUS_CERES
                          ,
                          SE3<ceres::Jet<double, 3>>
#endif
                          >("SE3") &&
    LieGroupTests<SE3<double>>::registerLeftJacobian("SE3") &&
    LieGroupTests<SE3<float>>::registerLeftJacobian("SE3");

template <class Scalar>
class SE3Test : public ::testing::Test {};

using SE3Scalars = ::testing::Types<double, float
#if SOPHUS_CERES
                                    ,
                                    ceres::Jet<double, 3>
#endif
                                    >;
TYPED_TEST_SUITE(SE3Test, SE3Scalars, ScalarNameGenerator);

TYPED_TEST(SE3Test, MapsAccessTheUnderlyingData) {
  using Scalar = TypeParam;
  using SE3Type = SE3<Scalar>;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();

  Vector7<Scalar> raw;
  raw << Scalar(0), Scalar(1), Scalar(0), Scalar(0), Scalar(1), Scalar(3),
      Scalar(2);
  Eigen::Map<SE3Type const> const map_of_const_se3(raw.data());
  EXPECT_THAT(map_of_const_se3.unit_quaternion().coeffs(),
              IsNear(raw.template head<4>(), kEpsilon));
  EXPECT_THAT(map_of_const_se3.translation(),
              IsNear(raw.template tail<3>(), kEpsilon));
  EXPECT_EQ(map_of_const_se3.unit_quaternion().coeffs().data(), raw.data());
  EXPECT_EQ(map_of_const_se3.translation().data(), raw.data() + 4);
  Eigen::Map<SE3Type const> const const_shallow_copy = map_of_const_se3;
  EXPECT_EQ(const_shallow_copy.unit_quaternion().coeffs().eval(),
            map_of_const_se3.unit_quaternion().coeffs().eval());
  EXPECT_EQ(const_shallow_copy.translation().eval(),
            map_of_const_se3.translation().eval());

  Vector7<Scalar> raw2;
  raw2 << Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(3), Scalar(2),
      Scalar(1);
  Eigen::Map<SE3Type> map_of_se3(raw.data());
  Eigen::Quaternion<Scalar> quat;
  quat.coeffs() = raw2.template head<4>();
  map_of_se3.setQuaternion(quat);
  map_of_se3.translation() = raw2.template tail<3>();
  EXPECT_THAT(map_of_se3.unit_quaternion().coeffs(),
              IsNear(raw2.template head<4>(), kEpsilon));
  EXPECT_THAT(map_of_se3.translation(),
              IsNear(raw2.template tail<3>(), kEpsilon));
  EXPECT_EQ(map_of_se3.unit_quaternion().coeffs().data(), raw.data());
  EXPECT_EQ(map_of_se3.translation().data(), raw.data() + 4);
  EXPECT_NE(map_of_se3.unit_quaternion().coeffs().data(), quat.coeffs().data());
  Eigen::Map<SE3Type> const shallow_copy = map_of_se3;
  EXPECT_EQ(shallow_copy.unit_quaternion().coeffs().eval(),
            map_of_se3.unit_quaternion().coeffs().eval());
  EXPECT_EQ(shallow_copy.translation().eval(), map_of_se3.translation().eval());
  Eigen::Map<SE3Type> const const_map_of_se3 = map_of_se3;
  EXPECT_EQ(const_map_of_se3.unit_quaternion().coeffs().eval(),
            map_of_se3.unit_quaternion().coeffs().eval());
  EXPECT_EQ(const_map_of_se3.translation().eval(),
            map_of_se3.translation().eval());
  // Writing through the map changed the underlying data.
  EXPECT_EQ(raw, raw2);
}

TYPED_TEST(SE3Test, DataAccessesTheParameters) {
  using Scalar = TypeParam;
  using SE3Type = SE3<Scalar>;

  Vector7<Scalar> raw;
  raw << Scalar(0), Scalar(1), Scalar(0), Scalar(0), Scalar(1), Scalar(3),
      Scalar(2);
  Vector7<Scalar> raw2;
  raw2 << Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(3), Scalar(2),
      Scalar(1);
  Eigen::Quaternion<Scalar> quat;
  quat.coeffs() = raw2.template head<4>();

  SE3Type const const_se3(quat, raw2.template tail<3>().eval());
  EXPECT_EQ(rawParams(const_se3), raw2);

  SE3Type se3(quat, raw2.template tail<3>().eval());
  EXPECT_EQ(rawParams(se3), raw2);
  Eigen::Map<Vector<Scalar, 7>>(se3.data()) = raw;
  EXPECT_EQ(rawParams(se3), raw);
}

TYPED_TEST(SE3Test, ElementaryTransformations) {
  using Scalar = TypeParam;
  using SE3Type = SE3<Scalar>;
  using Tangent = typename SE3Type::Tangent;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();

  EXPECT_THAT(
      SE3Type::transX(Scalar(0.2)).translation(),
      IsNear(Vector3<Scalar>(Scalar(0.2), Scalar(0), Scalar(0)), kEpsilon));
  EXPECT_THAT(
      SE3Type::transY(Scalar(0.7)).translation(),
      IsNear(Vector3<Scalar>(Scalar(0), Scalar(0.7), Scalar(0)), kEpsilon));
  EXPECT_THAT(
      SE3Type::transZ(Scalar(-0.2)).translation(),
      IsNear(Vector3<Scalar>(Scalar(0), Scalar(0), Scalar(-0.2)), kEpsilon));

  Tangent t;
  t << Scalar(0), Scalar(0), Scalar(0), Scalar(0.2), Scalar(0), Scalar(0);
  EXPECT_EQ(SE3Type::rotX(Scalar(0.2)).matrix(), SE3Type::exp(t).matrix());
  t << Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(-0.2), Scalar(0);
  EXPECT_EQ(SE3Type::rotY(Scalar(-0.2)).matrix(), SE3Type::exp(t).matrix());
  t << Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(1.1);
  EXPECT_EQ(SE3Type::rotZ(Scalar(1.1)).matrix(), SE3Type::exp(t).matrix());
}

TYPED_TEST(SE3Test, AssignsBetweenMapsAndElements) {
  using Scalar = TypeParam;
  using SE3Type = SE3<Scalar>;

  Vector7<Scalar> data1;
  data1 << Scalar(0), Scalar(1), Scalar(0), Scalar(0), Scalar(1), Scalar(2),
      Scalar(3);
  Vector7<Scalar> data2;
  data2 << Scalar(0), Scalar(0), Scalar(1), Scalar(0), Scalar(3), Scalar(2),
      Scalar(1);
  Eigen::Map<SE3Type> map1(data1.data());
  Eigen::Map<SE3Type> map2(data2.data());

  map2 = map1;
  EXPECT_EQ(map2.matrix(), map1.matrix()) << "map -> map assignment";

  SE3Type copy;
  copy = map1;
  EXPECT_EQ(copy.matrix(), map1.matrix()) << "map -> type assignment";

  copy = SE3Type::trans(Scalar(4), Scalar(5), Scalar(6)) *
         SE3Type::rotZ(Scalar(0.5));
  map1 = copy;
  EXPECT_EQ(map1.matrix(), copy.matrix()) << "type -> map assignment";
}

TYPED_TEST(SE3Test, SetRotationMatrix) {
  using Scalar = TypeParam;
  using SE3Type = SE3<Scalar>;
  using Point = typename SE3Type::Point;

  SO3<Scalar> const R(
      SO3<Scalar>::exp(Point(Scalar(0.2), Scalar(0.5), Scalar(0.0))));
  SE3Type se3;
  se3.setRotationMatrix(R.matrix());
  EXPECT_THAT(se3.rotationMatrix(),
              IsNear(R.matrix(), Constants<Scalar>::epsilon()));
}

TYPED_TEST(SE3Test, Constructors) {
  using Scalar = TypeParam;
  using SE3Type = SE3<Scalar>;
  using Point = typename SE3Type::Point;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();

  EXPECT_EQ(SE3Type().matrix(), Matrix4<Scalar>::Identity().eval());

  SE3Type const se3 = LieGroupTestData<SE3Type>::groupElements().front();
  Point const translation = se3.translation();
  SO3<Scalar> const so3 = se3.so3();

  EXPECT_THAT(SE3Type(so3, translation), IsNear(se3, kEpsilon));
  EXPECT_THAT(SE3Type(so3.matrix(), translation), IsNear(se3, kEpsilon));
  EXPECT_THAT(SE3Type(so3.unit_quaternion(), translation),
              IsNear(se3, kEpsilon));
  EXPECT_THAT(SE3Type(se3.matrix()), IsNear(se3, kEpsilon));
}

template <class Scalar>
class SE3FloatingPointTest : public ::testing::Test {};

TYPED_TEST_SUITE(SE3FloatingPointTest, FloatingPointScalars,
                 ScalarNameGenerator);

TYPED_TEST(SE3FloatingPointTest, FitToSE3IsIdempotent) {
  using Scalar = TypeParam;
  using SE3Type = SE3<Scalar>;
  int constexpr kNumSamples = 100;

  auto const fitted =
      mapped(randomMatrices<Matrix4<Scalar>>(kNumSamples),
             [](Matrix4<Scalar> const& M) { return SE3Type::fitToSE3(M); });
  auto const refitted = mapped(
      fitted, [](SE3Type const& g) { return SE3Type::fitToSE3(g.matrix()); });
  EXPECT_THAT(refitted, ::testing::Pointwise(
                            IsNear(Constants<Scalar>::epsilon()), fitted));
}

TYPED_TEST(SE3FloatingPointTest, AnglesOfElementaryRotations) {
  using Scalar = TypeParam;
  using SE3Type = SE3<Scalar>;
  using ::testing::Pointwise;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();
  std::vector<Scalar> const angles = {Scalar(0.0), Scalar(0.1), Scalar(0.3),
                                      Scalar(-0.7)};

  EXPECT_THAT(
      mapped(angles,
             [](Scalar angle) { return SE3Type::rotX(angle).angleX(); }),
      Pointwise(IsNear(kEpsilon), angles));
  EXPECT_THAT(
      mapped(angles,
             [](Scalar angle) { return SE3Type::rotY(angle).angleY(); }),
      Pointwise(IsNear(kEpsilon), angles));
  EXPECT_THAT(
      mapped(angles,
             [](Scalar angle) { return SE3Type::rotZ(angle).angleZ(); }),
      Pointwise(IsNear(kEpsilon), angles));
}

TYPED_TEST(SE3FloatingPointTest, LogMatchesReference) {
  reference::expectAccurateLog<SE3<TypeParam>>(reference::kSE3Log);
}

TYPED_TEST(SE3FloatingPointTest, ExpMatchesReference) {
  reference::expectAccurateExp<SE3<TypeParam>>(reference::kSE3Exp);
}

TYPED_TEST(SE3FloatingPointTest, LeftJacobianMatchesReference) {
  using SE3Type = SE3<TypeParam>;
  reference::expectAccurate<TypeParam>(
      reference::kSE3LeftJacobian, [](reference::Case const& c) {
        return SE3Type::leftJacobian(
            reference::input<typename SE3Type::Tangent>(c));
      });
}

TYPED_TEST(SE3FloatingPointTest, LeftJacobianInverseMatchesReference) {
  using SE3Type = SE3<TypeParam>;
  reference::expectAccurate<TypeParam>(
      reference::kSE3LeftJacobianInverse, [](reference::Case const& c) {
        return SE3Type::leftJacobianInverse(
            reference::input<typename SE3Type::Tangent>(c));
      });
}

TYPED_TEST(SE3FloatingPointTest, DxExpXMatchesReference) {
  reference::expectAccurateDxExpX<SE3<TypeParam>>(reference::kSE3DxExpX);
}

}  // namespace
}  // namespace Sophus
