// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#include <tuple>

#include <sophus/sim2.hpp>

#include "reference_sim2.hpp"
#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::Sim2<double>>;
template class Map<Sophus::Sim2<double> const>;
}  // namespace Eigen

namespace Sophus {

template class Sim2<double, Eigen::AutoAlign>;
template class Sim2<float, Eigen::DontAlign>;
#if SOPHUS_CERES
template class Sim2<ceres::Jet<double, 3>>;
#endif

template <class Scalar>
struct LieGroupTestData<Sim2<Scalar>> {
  using Sim2Type = Sim2<Scalar>;
  using RxSO2Type = RxSO2<Scalar>;
  using Point = typename Sim2Type::Point;
  using Tangent = typename Sim2Type::Tangent;
  using Vector2Type = Vector2<Scalar>;

  static StdVector<Sim2Type> groupElements() {
    Scalar const kPi = Constants<Scalar>::pi();
    StdVector<Sim2Type> sim2_vec;
    sim2_vec.push_back(
        Sim2Type(RxSO2Type::exp(Vector2Type(Scalar(0.2), Scalar(1.))),
                 Point(Scalar(0), Scalar(0))));
    sim2_vec.push_back(
        Sim2Type(RxSO2Type::exp(Vector2Type(Scalar(0.2), Scalar(1.1))),
                 Point(Scalar(10), Scalar(0))));
    sim2_vec.push_back(
        Sim2Type(RxSO2Type::exp(Vector2Type(Scalar(0.), Scalar(0.))),
                 Point(Scalar(0), Scalar(10))));
    sim2_vec.push_back(
        Sim2Type(RxSO2Type::exp(Vector2Type(Scalar(0.00001), Scalar(0.))),
                 Point(Scalar(0), Scalar(0))));
    sim2_vec.push_back(Sim2Type(
        RxSO2Type::exp(Vector2Type(Scalar(0.00001), Scalar(0.0000001))),
        Point(Scalar(1), Scalar(-1.00000001))));
    sim2_vec.push_back(
        Sim2Type(RxSO2Type::exp(Vector2Type(Scalar(0.), Scalar(0.))),
                 Point(Scalar(0.01), Scalar(0))));
    sim2_vec.push_back(Sim2Type(RxSO2Type::exp(Vector2Type(kPi, Scalar(0.9))),
                                Point(Scalar(4), Scalar(0))));
    sim2_vec.push_back(
        Sim2Type(RxSO2Type::exp(Vector2Type(Scalar(0.2), Scalar(0))),
                 Point(Scalar(0), Scalar(0))) *
        Sim2Type(RxSO2Type::exp(Vector2Type(kPi, Scalar(0))),
                 Point(Scalar(0), Scalar(0))) *
        Sim2Type(RxSO2Type::exp(Vector2Type(Scalar(-0.2), Scalar(0))),
                 Point(Scalar(0), Scalar(0))));
    sim2_vec.push_back(
        Sim2Type(RxSO2Type::exp(Vector2Type(Scalar(0.3), Scalar(0))),
                 Point(Scalar(2), Scalar(-7))) *
        Sim2Type(RxSO2Type::exp(Vector2Type(kPi, Scalar(0))),
                 Point(Scalar(0), Scalar(0))) *
        Sim2Type(RxSO2Type::exp(Vector2Type(Scalar(-0.3), Scalar(0))),
                 Point(Scalar(0), Scalar(6))));
    return sim2_vec;
  }

  static StdVector<Tangent> tangents() {
    StdVector<Tangent> tangent_vec;
    tangent_vec.push_back(Tangent(Scalar(0), Scalar(0), Scalar(0), Scalar(0)));
    tangent_vec.push_back(Tangent(Scalar(1), Scalar(0), Scalar(0), Scalar(0)));
    tangent_vec.push_back(
        Tangent(Scalar(0), Scalar(1), Scalar(0), Scalar(0.1)));
    tangent_vec.push_back(
        Tangent(Scalar(-1), Scalar(1), Scalar(1), Scalar(-0.1)));
    tangent_vec.push_back(
        Tangent(Scalar(20), Scalar(-1), Scalar(0), Scalar(-0.1)));
    tangent_vec.push_back(
        Tangent(Scalar(30), Scalar(5), Scalar(-1), Scalar(1.5)));
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
    registerLieGroupTests<Sim2<double>, Sim2<float>
#if SOPHUS_CERES
                          ,
                          Sim2<ceres::Jet<double, 3>>
#endif
                          >("Sim2");

template <class Scalar>
class Sim2Test : public ::testing::Test {};

using Sim2Scalars = ::testing::Types<double, float
#if SOPHUS_CERES
                                     ,
                                     ceres::Jet<double, 3>
#endif
                                     >;
TYPED_TEST_SUITE(Sim2Test, Sim2Scalars, ScalarNameGenerator);

// For an angle around 1e-4 and a vanishing log scale, W and its inverse come
// from their Taylor series, since their closed forms cancel catastrophically
// there. These elements are separate from LieGroupTestData, so that their
// tolerance is not diluted by the pairwise interpolation and mean checks
// against the much larger elements there.
TYPED_TEST(Sim2Test, ExpInvertsLogForSmallAngles) {
  using Scalar = TypeParam;
  using Sim2Type = Sim2<Scalar>;
  using Point = typename Sim2Type::Point;

  Sim2Type const T(
      RxSO2<Scalar>::exp(Vector2<Scalar>(Scalar(0.0001), Scalar(0))),
      Point(Scalar(1), Scalar(-0.5)));
  EXPECT_THAT(Sim2Type::exp(T.log()), IsNear(T, Constants<Scalar>::epsilon()));
}

// Where the angle and the log scale are both small, W and its inverse also
// come from their Taylor series, since terms shared by the numerators and
// denominators of their closed forms nearly cancel.
TYPED_TEST(Sim2Test, ExpInvertsLogForSmallAnglesAndScales) {
  using Scalar = TypeParam;
  using Sim2Type = Sim2<Scalar>;
  using Point = typename Sim2Type::Point;

  Sim2Type const T(
      RxSO2<Scalar>::exp(Vector2<Scalar>(Scalar(0.00003), Scalar(0.00002))),
      Point(Scalar(1), Scalar(-0.5)));
  EXPECT_THAT(Sim2Type::exp(T.log()), IsNear(T, Constants<Scalar>::epsilon()));
}

TYPED_TEST(Sim2Test, MapsAccessTheUnderlyingData) {
  using Scalar = TypeParam;
  using Sim2Type = Sim2<Scalar>;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();

  Vector4<Scalar> raw(Scalar(0), Scalar(1), Scalar(3), Scalar(2));
  Eigen::Map<Sim2Type const> const map_of_const_sim2(raw.data());
  EXPECT_THAT(map_of_const_sim2.complex(),
              IsNear(raw.template head<2>(), kEpsilon));
  EXPECT_THAT(map_of_const_sim2.translation(),
              IsNear(raw.template tail<2>(), kEpsilon));
  EXPECT_EQ(map_of_const_sim2.complex().data(), raw.data());
  EXPECT_EQ(map_of_const_sim2.translation().data(), raw.data() + 2);
  Eigen::Map<Sim2Type const> const const_shallow_copy = map_of_const_sim2;
  EXPECT_EQ(const_shallow_copy.complex().eval(),
            map_of_const_sim2.complex().eval());
  EXPECT_EQ(const_shallow_copy.translation().eval(),
            map_of_const_sim2.translation().eval());

  Vector4<Scalar> const raw2(Scalar(1), Scalar(0), Scalar(2), Scalar(1));
  Eigen::Map<Sim2Type> map_of_sim2(raw.data());
  Vector2<Scalar> const z = raw2.template head<2>();
  map_of_sim2.setComplex(z);
  map_of_sim2.translation() = raw2.template tail<2>();
  EXPECT_THAT(map_of_sim2.complex(), IsNear(raw2.template head<2>(), kEpsilon));
  EXPECT_THAT(map_of_sim2.translation(),
              IsNear(raw2.template tail<2>(), kEpsilon));
  EXPECT_EQ(map_of_sim2.complex().data(), raw.data());
  EXPECT_EQ(map_of_sim2.translation().data(), raw.data() + 2);
  Eigen::Map<Sim2Type> const shallow_copy = map_of_sim2;
  EXPECT_EQ(shallow_copy.complex().eval(), map_of_sim2.complex().eval());
  EXPECT_EQ(shallow_copy.translation().eval(),
            map_of_sim2.translation().eval());
  Eigen::Map<Sim2Type> const const_map_of_sim2 = map_of_sim2;
  EXPECT_EQ(const_map_of_sim2.complex().eval(), map_of_sim2.complex().eval());
  EXPECT_EQ(const_map_of_sim2.translation().eval(),
            map_of_sim2.translation().eval());
  // Writing through the map changed the underlying data.
  EXPECT_EQ(raw, raw2);
}

TYPED_TEST(Sim2Test, DataAccessesTheParameters) {
  using Scalar = TypeParam;
  using Sim2Type = Sim2<Scalar>;

  Vector4<Scalar> const raw(Scalar(0), Scalar(1), Scalar(3), Scalar(2));
  Vector4<Scalar> const raw2(Scalar(1), Scalar(0), Scalar(2), Scalar(1));
  Vector2<Scalar> const z = raw2.template head<2>();

  Sim2Type const const_sim2(z, raw2.template tail<2>().eval());
  EXPECT_EQ(rawParams(const_sim2), raw2);

  Sim2Type sim2(z, raw2.template tail<2>().eval());
  EXPECT_EQ(rawParams(sim2), raw2);
  Eigen::Map<Vector4<Scalar>>(sim2.data()) = raw;
  EXPECT_EQ(rawParams(sim2), raw);
}

TYPED_TEST(Sim2Test, AssignsBetweenMapsAndElements) {
  using Scalar = TypeParam;
  using Sim2Type = Sim2<Scalar>;
  using Point = typename Sim2Type::Point;

  Vector4<Scalar> data1(Scalar(0), Scalar(2), Scalar(1), Scalar(2));
  Vector4<Scalar> data2(Scalar(2), Scalar(0), Scalar(2), Scalar(1));
  Eigen::Map<Sim2Type> map1(data1.data());
  Eigen::Map<Sim2Type> map2(data2.data());

  map2 = map1;
  EXPECT_EQ(map2.matrix(), map1.matrix()) << "map -> map assignment";

  Sim2Type copy;
  copy = map1;
  EXPECT_EQ(copy.matrix(), map1.matrix()) << "map -> type assignment";

  copy = Sim2Type(RxSO2<Scalar>::exp(Vector2<Scalar>(Scalar(-1), Scalar(1))),
                  Point(Scalar(10), Scalar(0)));
  map1 = copy;
  EXPECT_EQ(map1.matrix(), copy.matrix()) << "type -> map assignment";
}

TYPED_TEST(Sim2Test, ConstructorsAndSetters) {
  using Scalar = TypeParam;
  using Sim2Type = Sim2<Scalar>;
  using Point = typename Sim2Type::Point;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();

  EXPECT_EQ(Sim2Type().matrix(), Matrix3<Scalar>::Identity().eval());

  StdVector<Sim2Type> const sim2_vec =
      LieGroupTestData<Sim2Type>::groupElements();
  // An element with a nonzero translation, so that each constructor is checked
  // to keep it.
  Sim2Type sim2 = sim2_vec[1];
  Point const translation = sim2.translation();
  RxSO2<Scalar> const rxso2 = sim2.rxso2();

  EXPECT_THAT(Sim2Type(rxso2, translation), IsNear(sim2, kEpsilon));
  EXPECT_THAT(Sim2Type(rxso2.complex(), translation), IsNear(sim2, kEpsilon));
  EXPECT_THAT(Sim2Type(sim2.matrix()), IsNear(sim2, kEpsilon));

  Scalar const scale = Scalar(1.2);
  sim2.setScale(scale);
  EXPECT_THAT(sim2.scale(), IsNear(scale, kEpsilon)) << "setScale";

  sim2.setComplex(sim2_vec.front().rxso2().complex());
  EXPECT_THAT(sim2.rxso2().complex(),
              IsNear(sim2_vec.front().rxso2().complex(), kEpsilon))
      << "setComplex";
}

template <class Scalar>
class Sim2FloatingPointTest : public ::testing::Test {};

TYPED_TEST_SUITE(Sim2FloatingPointTest, FloatingPointScalars,
                 ScalarNameGenerator);

TYPED_TEST(Sim2FloatingPointTest, ExpMatchesReference) {
  reference::expectAccurateExp<Sim2<TypeParam>>(reference::kSim2Exp);
}

TYPED_TEST(Sim2FloatingPointTest, LogMatchesReference) {
  reference::expectAccurateLog<Sim2<TypeParam>>(reference::kSim2Log);
}

TYPED_TEST(Sim2FloatingPointTest, DxExpXMatchesReference) {
  reference::expectAccurateDxExpX<Sim2<TypeParam>>(reference::kSim2DxExpX);
}

TEST(Sim2DeathTest, RejectsMatricesWithoutHomogeneousLastRow) {
  constexpr Eigen::Index kLastRow = 3 - 1;
  constexpr double kPerturbation = 5;
  for (Eigen::Index col = 0; col <= kLastRow; ++col) {
    SCOPED_TRACE(::testing::Message() << "column " << col);
    Matrix3<double> T = Matrix3<double>::Identity();
    T(kLastRow, col) += kPerturbation;
    EXPECT_DEATH(std::ignore = Sim2<double>{T}, "Last row");
  }
}

TYPED_TEST(Sim2FloatingPointTest, SetsRotationFromTemporaryMatrix) {
  using Scalar = TypeParam;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();
  Scalar const kAngle(0.5);
  Scalar const kLogScale(0.3);
  Scalar const kNewAngle(-1.2);
  Sim2<Scalar> T(RxSO2<Scalar>::exp(Vector2<Scalar>(kAngle, kLogScale)),
                 Vector2<Scalar>(Scalar(1), Scalar(2)));
  Scalar const scale = T.scale();
  T.setRotationMatrix(SO2<Scalar>::exp(kNewAngle).matrix());
  EXPECT_THAT(T.rotationMatrix(),
              IsNear(SO2<Scalar>::exp(kNewAngle).matrix(), kEpsilon));
  EXPECT_THAT(T.scale(), IsNear(scale, kEpsilon))
      << "setRotationMatrix leaves the scale as is";
}

}  // namespace
}  // namespace Sophus
