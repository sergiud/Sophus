// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#include <tuple>

#include <sophus/sim3.hpp>

#include "reference_sim3.hpp"
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::Sim3<double>>;
template class Map<Sophus::Sim3<double> const>;
}  // namespace Eigen

namespace Sophus {

template class Sim3<double, Eigen::AutoAlign>;
template class Sim3<float, Eigen::DontAlign>;
#if SOPHUS_CERES
template class Sim3<ceres::Jet<double, 3>>;
#endif

template <class Scalar>
struct LieGroupTestData<Sim3<Scalar>> {
  using Sim3Type = Sim3<Scalar>;
  using RxSO3Type = RxSO3<Scalar>;
  using Point = typename Sim3Type::Point;
  using Tangent = typename Sim3Type::Tangent;
  using Vector4Type = Vector4<Scalar>;

  static StdVector<Sim3Type> groupElements() {
    Scalar const kPi = Constants<Scalar>::pi();
    StdVector<Sim3Type> sim3_vec;
    sim3_vec.push_back(
        Sim3Type(RxSO3Type::exp(Vector4Type(Scalar(0.2), Scalar(0.5),
                                            Scalar(0.0), Scalar(1.))),
                 Point(Scalar(0), Scalar(0), Scalar(0))));
    sim3_vec.push_back(
        Sim3Type(RxSO3Type::exp(Vector4Type(Scalar(0.2), Scalar(0.5),
                                            Scalar(-1.0), Scalar(1.1))),
                 Point(Scalar(10), Scalar(0), Scalar(0))));
    sim3_vec.push_back(
        Sim3Type(RxSO3Type::exp(Vector4Type(Scalar(0.), Scalar(0.), Scalar(0.),
                                            Scalar(0.))),
                 Point(Scalar(0), Scalar(10), Scalar(5))));
    sim3_vec.push_back(
        Sim3Type(RxSO3Type::exp(Vector4Type(Scalar(0.), Scalar(0.), Scalar(0.),
                                            Scalar(1.1))),
                 Point(Scalar(0), Scalar(10), Scalar(5))));
    sim3_vec.push_back(
        Sim3Type(RxSO3Type::exp(Vector4Type(Scalar(0.), Scalar(0.),
                                            Scalar(0.00001), Scalar(0.))),
                 Point(Scalar(0), Scalar(0), Scalar(0))));
    sim3_vec.push_back(Sim3Type(
        RxSO3Type::exp(Vector4Type(Scalar(0.), Scalar(0.), Scalar(0.00001),
                                   Scalar(0.0000001))),
        Point(Scalar(1), Scalar(-1.00000001), Scalar(2.0000000001))));
    sim3_vec.push_back(
        Sim3Type(RxSO3Type::exp(Vector4Type(Scalar(0.), Scalar(0.),
                                            Scalar(0.00001), Scalar(0))),
                 Point(Scalar(0.01), Scalar(0), Scalar(0))));
    sim3_vec.push_back(Sim3Type(
        RxSO3Type::exp(Vector4Type(kPi, Scalar(0), Scalar(0), Scalar(0.9))),
        Point(Scalar(4), Scalar(-5), Scalar(0))));
    sim3_vec.push_back(
        Sim3Type(RxSO3Type::exp(Vector4Type(Scalar(0.2), Scalar(0.5),
                                            Scalar(0.0), Scalar(0))),
                 Point(Scalar(0), Scalar(0), Scalar(0))) *
        Sim3Type(
            RxSO3Type::exp(Vector4Type(kPi, Scalar(0), Scalar(0), Scalar(0))),
            Point(Scalar(0), Scalar(0), Scalar(0))) *
        Sim3Type(RxSO3Type::exp(Vector4Type(Scalar(-0.2), Scalar(-0.5),
                                            Scalar(-0.0), Scalar(0))),
                 Point(Scalar(0), Scalar(0), Scalar(0))));
    sim3_vec.push_back(
        Sim3Type(RxSO3Type::exp(Vector4Type(Scalar(0.3), Scalar(0.5),
                                            Scalar(0.1), Scalar(0))),
                 Point(Scalar(2), Scalar(0), Scalar(-7))) *
        Sim3Type(
            RxSO3Type::exp(Vector4Type(kPi, Scalar(0), Scalar(0), Scalar(0))),
            Point(Scalar(0), Scalar(0), Scalar(0))) *
        Sim3Type(RxSO3Type::exp(Vector4Type(Scalar(-0.3), Scalar(-0.5),
                                            Scalar(-0.1), Scalar(0))),
                 Point(Scalar(0), Scalar(6), Scalar(0))));
    return sim3_vec;
  }

  static StdVector<Tangent> tangents() {
    StdVector<Tangent> tangent_vec;
    Tangent tmp;
    tmp << Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
        Scalar(0);
    tangent_vec.push_back(tmp);
    tmp << Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(0), Scalar(0),
        Scalar(0);
    tangent_vec.push_back(tmp);
    tmp << Scalar(0), Scalar(1), Scalar(0), Scalar(1), Scalar(0), Scalar(0),
        Scalar(0.1);
    tangent_vec.push_back(tmp);
    tmp << Scalar(0), Scalar(0), Scalar(1), Scalar(0), Scalar(1), Scalar(0),
        Scalar(0.1);
    tangent_vec.push_back(tmp);
    tmp << Scalar(-1), Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(1),
        Scalar(-0.1);
    tangent_vec.push_back(tmp);
    tmp << Scalar(20), Scalar(-1), Scalar(0), Scalar(-1), Scalar(1), Scalar(0),
        Scalar(-0.1);
    tangent_vec.push_back(tmp);
    tmp << Scalar(30), Scalar(5), Scalar(-1), Scalar(20), Scalar(-1), Scalar(0),
        Scalar(1.5);
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
    registerLieGroupTests<Sim3<double>, Sim3<float>
#if SOPHUS_CERES
                          ,
                          Sim3<ceres::Jet<double, 3>>
#endif
                          >("Sim3");

template <class Scalar>
class Sim3Test : public ::testing::Test {};

using Sim3Scalars = ::testing::Types<double, float
#if SOPHUS_CERES
                                     ,
                                     ceres::Jet<double, 3>
#endif
                                     >;
TYPED_TEST_SUITE(Sim3Test, Sim3Scalars, ScalarNameGenerator);

// For an angle around 1e-4 and a vanishing log scale, W and its inverse come
// from their Taylor series, since their closed forms cancel catastrophically
// there. These elements are separate from LieGroupTestData, so that their
// tolerance is not diluted by the pairwise interpolation and mean checks
// against the much larger elements there.
TYPED_TEST(Sim3Test, ExpInvertsLogForSmallAngles) {
  using Scalar = TypeParam;
  using Sim3Type = Sim3<Scalar>;
  using Point = typename Sim3Type::Point;

  Sim3Type const T(RxSO3<Scalar>::exp(Vector4<Scalar>(
                       Scalar(0.), Scalar(0.), Scalar(0.0001), Scalar(0.))),
                   Point(Scalar(1), Scalar(-0.5), Scalar(2)));
  EXPECT_THAT(Sim3Type::exp(T.log()), IsNear(T, Constants<Scalar>::epsilon()));
}

// Where the angle and the log scale are both small, W and its inverse also
// come from their Taylor series, since terms shared by the numerators and
// denominators of their closed forms nearly cancel.
TYPED_TEST(Sim3Test, ExpInvertsLogForSmallAnglesAndScales) {
  using Scalar = TypeParam;
  using Sim3Type = Sim3<Scalar>;
  using Point = typename Sim3Type::Point;

  Sim3Type const T(
      RxSO3<Scalar>::exp(Vector4<Scalar>(Scalar(0.), Scalar(0.),
                                         Scalar(0.00003), Scalar(0.00002))),
      Point(Scalar(1), Scalar(-0.5), Scalar(2)));
  EXPECT_THAT(Sim3Type::exp(T.log()), IsNear(T, Constants<Scalar>::epsilon()));
}

TYPED_TEST(Sim3Test, MapsAccessTheUnderlyingData) {
  using Scalar = TypeParam;
  using Sim3Type = Sim3<Scalar>;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();

  Vector7<Scalar> raw;
  raw << Scalar(0), Scalar(1), Scalar(0), Scalar(0), Scalar(1), Scalar(3),
      Scalar(2);
  Eigen::Map<Sim3Type const> const map_of_const_sim3(raw.data());
  EXPECT_THAT(map_of_const_sim3.quaternion().coeffs(),
              IsNear(raw.template head<4>(), kEpsilon));
  EXPECT_THAT(map_of_const_sim3.translation(),
              IsNear(raw.template tail<3>(), kEpsilon));
  EXPECT_EQ(map_of_const_sim3.quaternion().coeffs().data(), raw.data());
  EXPECT_EQ(map_of_const_sim3.translation().data(), raw.data() + 4);
  Eigen::Map<Sim3Type const> const const_shallow_copy = map_of_const_sim3;
  EXPECT_EQ(const_shallow_copy.quaternion().coeffs().eval(),
            map_of_const_sim3.quaternion().coeffs().eval());
  EXPECT_EQ(const_shallow_copy.translation().eval(),
            map_of_const_sim3.translation().eval());

  Vector7<Scalar> raw2;
  raw2 << Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(3), Scalar(2),
      Scalar(1);
  Eigen::Map<Sim3Type> map_of_sim3(raw.data());
  Eigen::Quaternion<Scalar> quat;
  quat.coeffs() = raw2.template head<4>();
  map_of_sim3.setQuaternion(quat);
  map_of_sim3.translation() = raw2.template tail<3>();
  EXPECT_THAT(map_of_sim3.quaternion().coeffs(),
              IsNear(raw2.template head<4>(), kEpsilon));
  EXPECT_THAT(map_of_sim3.translation(),
              IsNear(raw2.template tail<3>(), kEpsilon));
  EXPECT_EQ(map_of_sim3.quaternion().coeffs().data(), raw.data());
  EXPECT_EQ(map_of_sim3.translation().data(), raw.data() + 4);
  Eigen::Map<Sim3Type> const shallow_copy = map_of_sim3;
  EXPECT_EQ(shallow_copy.quaternion().coeffs().eval(),
            map_of_sim3.quaternion().coeffs().eval());
  EXPECT_EQ(shallow_copy.translation().eval(),
            map_of_sim3.translation().eval());
  Eigen::Map<Sim3Type> const const_map_of_sim3 = map_of_sim3;
  EXPECT_EQ(const_map_of_sim3.quaternion().coeffs().eval(),
            map_of_sim3.quaternion().coeffs().eval());
  EXPECT_EQ(const_map_of_sim3.translation().eval(),
            map_of_sim3.translation().eval());
  // Writing through the map changed the underlying data.
  EXPECT_EQ(raw, raw2);
}

TYPED_TEST(Sim3Test, DataAccessesTheParameters) {
  using Scalar = TypeParam;
  using Sim3Type = Sim3<Scalar>;

  Vector7<Scalar> raw;
  raw << Scalar(0), Scalar(1), Scalar(0), Scalar(0), Scalar(1), Scalar(3),
      Scalar(2);
  Vector7<Scalar> raw2;
  raw2 << Scalar(1), Scalar(0), Scalar(0), Scalar(0), Scalar(3), Scalar(2),
      Scalar(1);
  Eigen::Quaternion<Scalar> quat;
  quat.coeffs() = raw2.template head<4>();

  Sim3Type const const_sim3(quat, raw2.template tail<3>().eval());
  EXPECT_EQ(rawParams(const_sim3), raw2);

  Sim3Type sim3(quat, raw2.template tail<3>().eval());
  EXPECT_EQ(rawParams(sim3), raw2);
  Eigen::Map<Vector7<Scalar>>(sim3.data()) = raw;
  EXPECT_EQ(rawParams(sim3), raw);
}

TYPED_TEST(Sim3Test, AssignsBetweenMapsAndElements) {
  using Scalar = TypeParam;
  using Sim3Type = Sim3<Scalar>;
  using Point = typename Sim3Type::Point;

  Vector7<Scalar> data1;
  data1 << Scalar(0), Scalar(2), Scalar(0), Scalar(0), Scalar(1), Scalar(2),
      Scalar(3);
  Vector7<Scalar> data2;
  data2 << Scalar(0), Scalar(0), Scalar(2), Scalar(0), Scalar(3), Scalar(2),
      Scalar(1);
  Eigen::Map<Sim3Type> map1(data1.data());
  Eigen::Map<Sim3Type> map2(data2.data());

  map2 = map1;
  EXPECT_EQ(map2.matrix(), map1.matrix()) << "map -> map assignment";

  Sim3Type copy;
  copy = map1;
  EXPECT_EQ(copy.matrix(), map1.matrix()) << "map -> type assignment";

  copy = Sim3Type(RxSO3<Scalar>::exp(Vector4<Scalar>(
                      Scalar(0.2), Scalar(0.5), Scalar(-1.0), Scalar(1.1))),
                  Point(Scalar(10), Scalar(0), Scalar(0)));
  map1 = copy;
  EXPECT_EQ(map1.matrix(), copy.matrix()) << "type -> map assignment";
}

TYPED_TEST(Sim3Test, ConstructorsAndSetters) {
  using Scalar = TypeParam;
  using Sim3Type = Sim3<Scalar>;
  using Point = typename Sim3Type::Point;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();

  EXPECT_EQ(Sim3Type().matrix(), Matrix4<Scalar>::Identity().eval());

  StdVector<Sim3Type> const sim3_vec =
      LieGroupTestData<Sim3Type>::groupElements();
  // An element with a nonzero translation, so that each constructor is checked
  // to keep it.
  Sim3Type sim3 = sim3_vec[1];
  Point const translation = sim3.translation();
  RxSO3<Scalar> const rxso3 = sim3.rxso3();

  EXPECT_THAT(Sim3Type(rxso3, translation), IsNear(sim3, kEpsilon))
      << "Sim3(RxSO3, translation)";
  EXPECT_THAT(Sim3Type(rxso3.quaternion(), translation), IsNear(sim3, kEpsilon))
      << "Sim3(quaternion, translation)";
  EXPECT_THAT(
      Sim3Type(rxso3.scale(), rxso3.quaternion().normalized(), translation),
      IsNear(sim3, kEpsilon))
      << "Sim3(scale, unit quaternion, translation)";
  EXPECT_THAT(Sim3Type(sim3.matrix()), IsNear(sim3, kEpsilon))
      << "Sim3(matrix)";

  Scalar const scale = Scalar(1.2);
  sim3.setScale(scale);
  EXPECT_THAT(sim3.scale(), IsNear(scale, kEpsilon)) << "setScale";

  sim3.setQuaternion(sim3_vec.front().rxso3().quaternion());
  EXPECT_THAT(sim3.rxso3().quaternion().coeffs(),
              IsNear(sim3_vec.front().rxso3().quaternion().coeffs(), kEpsilon))
      << "setQuaternion";
}

template <class Scalar>
class Sim3FloatingPointTest : public ::testing::Test {};

TYPED_TEST_SUITE(Sim3FloatingPointTest, FloatingPointScalars,
                 ScalarNameGenerator);

TYPED_TEST(Sim3FloatingPointTest, ExpMatchesReference) {
  reference::expectAccurateExp<Sim3<TypeParam>>(reference::kSim3Exp);
}

TYPED_TEST(Sim3FloatingPointTest, LogMatchesReference) {
  reference::expectAccurateLog<Sim3<TypeParam>>(reference::kSim3Log);
}

TYPED_TEST(Sim3FloatingPointTest, DxExpXMatchesReference) {
  reference::expectAccurateDxExpX<Sim3<TypeParam>>(reference::kSim3DxExpX);
}

TEST(Sim3DeathTest, RejectsMatricesWithoutHomogeneousLastRow) {
  constexpr Eigen::Index kLastRow = 4 - 1;
  constexpr double kPerturbation = 5;
  for (Eigen::Index col = 0; col <= kLastRow; ++col) {
    SCOPED_TRACE(::testing::Message() << "column " << col);
    Matrix4<double> T = Matrix4<double>::Identity();
    T(kLastRow, col) += kPerturbation;
    EXPECT_DEATH(std::ignore = Sim3<double>{T}, "Last row");
  }
}

TYPED_TEST(Sim3FloatingPointTest, SetsRotationFromTemporaryMatrix) {
  using Scalar = TypeParam;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();
  Scalar const kLogScale(0.3);
  Vector3<Scalar> const omega(Scalar(0.2), Scalar(0.5), Scalar(-1));
  Vector3<Scalar> const new_omega(Scalar(-1.2), Scalar(0.4), Scalar(0.1));
  Sim3<Scalar> T(
      RxSO3<Scalar>::exp((Vector4<Scalar>() << omega, kLogScale).finished()),
      Vector3<Scalar>(Scalar(1), Scalar(2), Scalar(3)));
  Scalar const scale = T.scale();
  T.setRotationMatrix(SO3<Scalar>::exp(new_omega).matrix());
  EXPECT_THAT(T.rotationMatrix(),
              IsNear(SO3<Scalar>::exp(new_omega).matrix(), kEpsilon));
  EXPECT_THAT(T.scale(), IsNear(scale, kEpsilon))
      << "setRotationMatrix leaves the scale as is";
}

}  // namespace
}  // namespace Sophus
