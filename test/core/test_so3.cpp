// Copyright 2022, 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#include <cmath>
#include <cstddef>
#include <random>

#include <sophus/interpolate.hpp>
#include <sophus/so3.hpp>

#include "tests.hpp"

// Explicit instantiate all class templates so that all member methods
// get compiled and for code coverage analysis.
namespace Eigen {
template class Map<Sophus::SO3<double>>;
template class Map<Sophus::SO3<double> const>;
}  // namespace Eigen

namespace Sophus {

template class SO3<double, Eigen::AutoAlign>;
template class SO3<float, Eigen::DontAlign>;
#if SOPHUS_CERES
template class SO3<ceres::Jet<double, 3>>;
#endif

template <class Scalar>
struct LieGroupTestData<SO3<Scalar>> {
  using SO3Type = SO3<Scalar>;
  using Point = typename SO3Type::Point;
  using Tangent = typename SO3Type::Tangent;

  static StdVector<SO3Type> groupElements() {
    Scalar const kPi = Constants<Scalar>::pi();
    StdVector<SO3Type> so3_vec;
    so3_vec.push_back(SO3Type(Eigen::Quaternion<Scalar>(
        Scalar(0.1e-11), Scalar(0.), Scalar(1.), Scalar(0.))));
    so3_vec.push_back(SO3Type(Eigen::Quaternion<Scalar>(
        Scalar(-1), Scalar(0.00001), Scalar(0.0), Scalar(0.0))));
    so3_vec.push_back(
        SO3Type::exp(Point(Scalar(0.2), Scalar(0.5), Scalar(0.0))));
    so3_vec.push_back(
        SO3Type::exp(Point(Scalar(0.2), Scalar(0.5), Scalar(-1.0))));
    so3_vec.push_back(SO3Type::exp(Point(Scalar(0.), Scalar(0.), Scalar(0.))));
    so3_vec.push_back(
        SO3Type::exp(Point(Scalar(0.), Scalar(0.), Scalar(0.00001))));
    so3_vec.push_back(SO3Type::exp(Point(kPi, Scalar(0), Scalar(0))));
    so3_vec.push_back(
        SO3Type::exp(Point(Scalar(0.2), Scalar(0.5), Scalar(0.0))) *
        SO3Type::exp(Point(kPi, Scalar(0), Scalar(0))) *
        SO3Type::exp(Point(Scalar(-0.2), Scalar(-0.5), Scalar(-0.0))));
    so3_vec.push_back(
        SO3Type::exp(Point(Scalar(0.3), Scalar(0.5), Scalar(0.1))) *
        SO3Type::exp(Point(kPi, Scalar(0), Scalar(0))) *
        SO3Type::exp(Point(Scalar(-0.3), Scalar(-0.5), Scalar(-0.1))));
    return so3_vec;
  }

  static StdVector<Tangent> tangents() {
    Scalar const kPi = Constants<Scalar>::pi();
    StdVector<Tangent> tangent_vec;
    tangent_vec.push_back(Tangent(Scalar(0), Scalar(0), Scalar(0)));
    tangent_vec.push_back(Tangent(Scalar(1), Scalar(0), Scalar(0)));
    tangent_vec.push_back(Tangent(Scalar(0), Scalar(1), Scalar(0)));
    tangent_vec.push_back(
        Tangent(Scalar(kPi / 2.), Scalar(kPi / 2.), Scalar(0)));
    tangent_vec.push_back(Tangent(Scalar(-1), Scalar(1), Scalar(0)));
    tangent_vec.push_back(Tangent(Scalar(20), Scalar(-1), Scalar(0)));
    tangent_vec.push_back(Tangent(Scalar(30), Scalar(5), Scalar(-1)));
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
    registerLieGroupTests<SO3<double>, SO3<float>
#if SOPHUS_CERES
                          ,
                          SO3<ceres::Jet<double, 3>>
#endif
                          >("SO3") &&
    LieGroupTests<SO3<double>>::registerLeftJacobian("SO3") &&
    LieGroupTests<SO3<float>>::registerLeftJacobian("SO3");

template <class Scalar>
class SO3Test : public ::testing::Test {};

using SO3Scalars = ::testing::Types<double, float
#if SOPHUS_CERES
                                    ,
                                    ceres::Jet<double, 3>
#endif
                                    >;
TYPED_TEST_SUITE(SO3Test, SO3Scalars, ScalarNameGenerator);

TYPED_TEST(SO3Test, RepeatedProductsKeepUnitMagnitude) {
  using Scalar = TypeParam;
  using SO3Type = SO3<Scalar>;
  int constexpr kNumRepetitions = 1000;

  StdVector<SO3Type> const so3_vec = LieGroupTestData<SO3Type>::groupElements();
  SO3Type current_q;
  for (int i = 0; i < kNumRepetitions; ++i) {
    for (SO3Type const& q : so3_vec) {
      current_q *= q;
    }
  }
  EXPECT_THAT(current_q.unit_quaternion().norm(),
              IsNear(Scalar(1), Constants<Scalar>::epsilon()));
}

TYPED_TEST(SO3Test, MapsAccessTheUnderlyingData) {
  using Scalar = TypeParam;
  using SO3Type = SO3<Scalar>;

  Vector4<Scalar> raw(Scalar(0), Scalar(1), Scalar(0), Scalar(0));
  Eigen::Map<SO3Type const> const map_of_const_so3(raw.data());
  EXPECT_THAT(map_of_const_so3.unit_quaternion().coeffs(),
              IsNear(raw, Constants<Scalar>::epsilon()));
  EXPECT_EQ(map_of_const_so3.unit_quaternion().coeffs().data(), raw.data());
  Eigen::Map<SO3Type const> const const_shallow_copy = map_of_const_so3;
  EXPECT_EQ(const_shallow_copy.unit_quaternion().coeffs().eval(),
            map_of_const_so3.unit_quaternion().coeffs().eval());

  Vector4<Scalar> const raw2(Scalar(1), Scalar(0), Scalar(0), Scalar(0));
  Eigen::Map<SO3Type> map_of_so3(raw.data());
  Eigen::Quaternion<Scalar> quat;
  quat.coeffs() = raw2;
  map_of_so3.setQuaternion(quat);
  EXPECT_THAT(map_of_so3.unit_quaternion().coeffs(),
              IsNear(raw2, Constants<Scalar>::epsilon()));
  EXPECT_EQ(map_of_so3.unit_quaternion().coeffs().data(), raw.data());
  EXPECT_NE(map_of_so3.unit_quaternion().coeffs().data(), quat.coeffs().data());
  Eigen::Map<SO3Type> const shallow_copy = map_of_so3;
  EXPECT_EQ(shallow_copy.unit_quaternion().coeffs().eval(),
            map_of_so3.unit_quaternion().coeffs().eval());
}

TYPED_TEST(SO3Test, DataAccessesTheParameters) {
  using Scalar = TypeParam;
  using SO3Type = SO3<Scalar>;

  Vector4<Scalar> const raw(Scalar(0), Scalar(1), Scalar(0), Scalar(0));
  Vector4<Scalar> const raw2(Scalar(1), Scalar(0), Scalar(0), Scalar(0));
  Eigen::Quaternion<Scalar> quat;
  quat.coeffs() = raw2;

  SO3Type const const_so3(quat);
  EXPECT_EQ(rawParams(const_so3), raw2);

  SO3Type so3(quat);
  Eigen::Map<Vector<Scalar, 4>>(so3.data()) = raw;
  EXPECT_EQ(rawParams(so3), raw);
}

TYPED_TEST(SO3Test, ElementaryRotationsMatchExp) {
  using Scalar = TypeParam;
  using SO3Type = SO3<Scalar>;
  using Point = typename SO3Type::Point;

  EXPECT_EQ(SO3Type::rotX(Scalar(0.2)).matrix(),
            SO3Type::exp(Point(Scalar(0.2), Scalar(0), Scalar(0))).matrix());
  EXPECT_EQ(SO3Type::rotY(Scalar(-0.2)).matrix(),
            SO3Type::exp(Point(Scalar(0), Scalar(-0.2), Scalar(0))).matrix());
  EXPECT_EQ(SO3Type::rotZ(Scalar(1.1)).matrix(),
            SO3Type::exp(Point(Scalar(0), Scalar(0), Scalar(1.1))).matrix());
}

TYPED_TEST(SO3Test, AssignsBetweenMapsAndElements) {
  using Scalar = TypeParam;
  using SO3Type = SO3<Scalar>;

  Vector4<Scalar> data1(Scalar{1}, Scalar{0}, Scalar{0}, Scalar{0});
  Vector4<Scalar> data2(Scalar{0}, Scalar{1}, Scalar{0}, Scalar{0});
  Eigen::Map<SO3Type> map1(data1.data());
  Eigen::Map<SO3Type> map2(data2.data());

  map2 = map1;
  EXPECT_EQ(map2.matrix(), map1.matrix()) << "map -> map assignment";

  SO3Type copy;
  copy = map1;
  EXPECT_EQ(copy.matrix(), map1.matrix()) << "map -> type assignment";

  copy = SO3Type::rotZ(Scalar(0.5));
  map1 = copy;
  EXPECT_EQ(map1.matrix(), copy.matrix()) << "type -> map assignment";
}

TYPED_TEST(SO3Test, ConstructsFromRotationMatrix) {
  using Scalar = TypeParam;
  using SO3Type = SO3<Scalar>;

  Matrix3<Scalar> const R =
      LieGroupTestData<SO3Type>::groupElements().front().matrix();
  EXPECT_THAT(SO3Type(R).matrix(), IsNear(R, Constants<Scalar>::epsilon()));
}

template <class Scalar>
class SO3FloatingPointTest : public ::testing::Test {};

TYPED_TEST_SUITE(SO3FloatingPointTest, FloatingPointScalars,
                 ScalarNameGenerator);

TYPED_TEST(SO3FloatingPointTest, SampleUniformIsSymmetric) {
  using Scalar = TypeParam;
  using SO3Type = SO3<Scalar>;
  using Point = typename SO3Type::Point;
  using ::testing::Lt;
  int constexpr kNumTrials = 5;
  std::size_t constexpr kNumSamples = 5000;
  // Critical value for chi-square(df=5), p=0.01.
  double constexpr kChiSquareCriticalValue = 15.086;

  std::default_random_engine generator(0);

  // A non-rigorous test for checking that our sampleUniform() function is
  // giving us symmetric results
  //
  // We (a) split the output space in half, (b) apply a series of random
  // rotations to a point, (c) check which half of the output space each
  // transformed point ends up, and then (d) apply a standard "coin toss"
  // chi-square test

  // Each of the trials below contributes one chi-square(df=1) statistic.
  // Summed, they follow a chi-square(df=5) distribution under the null
  // hypothesis of uniform and symmetric sampling. Comparing that sum against
  // a single chi-square(df=5) critical value, rather than comparing each
  // trial individually against the df=1 critical value, avoids inflating the
  // overall false-failure rate through multiple comparisons: with 5
  // independent per-trial checks each at p=0.01, the chance that at least
  // one spuriously exceeds its threshold is about 5%, not 1%. That rate
  // showed up in practice as a std::normal_distribution implementation
  // difference between libstdc++ and libc++ producing a different but
  // equally valid sample sequence for the same fixed seed.
  double chi_square_sum = 0;
  for (int trial = 0; trial < kNumTrials; ++trial) {
    std::normal_distribution<Scalar> normal(0, 10);

    // Pick a random plane to split the output space by
    Point plane_normal(normal(generator), normal(generator), normal(generator));
    plane_normal /= plane_normal.norm();

    // Pick a random point to be rotated
    Point input_point(normal(generator), normal(generator), normal(generator));
    input_point /= input_point.norm();

    // Randomly rotate points and track # that land on each side of plane
    std::size_t positive_count = 0;
    std::size_t negative_count = 0;
    for (std::size_t i = 0; i < kNumSamples; ++i) {
      SO3Type const R = SO3Type::sampleUniform(generator);
      if (plane_normal.dot(R * input_point) > 0) {
        ++positive_count;
      } else {
        ++negative_count;
      }
    }

    double const expected_count = static_cast<double>(kNumSamples) / 2.0;
    chi_square_sum +=
        std::pow(positive_count - expected_count, 2.0) / expected_count +
        std::pow(negative_count - expected_count, 2.0) / expected_count;
  }

  EXPECT_THAT(chi_square_sum, Lt(kChiSquareCriticalValue));
}

TYPED_TEST(SO3FloatingPointTest, FitToSO3IsIdempotent) {
  using Scalar = TypeParam;
  using SO3Type = SO3<Scalar>;
  int constexpr kNumSamples = 100;

  auto const fitted =
      mapped(randomMatrices<Matrix3<Scalar>>(kNumSamples),
             [](Matrix3<Scalar> const& M) { return SO3Type::fitToSO3(M); });
  auto const refitted = mapped(
      fitted, [](SO3Type const& g) { return SO3Type::fitToSO3(g.matrix()); });
  EXPECT_THAT(refitted, ::testing::Pointwise(
                            IsNear(Constants<Scalar>::epsilon()), fitted));
}

TYPED_TEST(SO3FloatingPointTest, AnglesOfElementaryRotations) {
  using Scalar = TypeParam;
  using SO3Type = SO3<Scalar>;
  using ::testing::Pointwise;
  Scalar const kEpsilon = Constants<Scalar>::epsilon();
  std::vector<Scalar> const angles = {Scalar(0.0), Scalar(0.1), Scalar(0.3),
                                      Scalar(-0.7)};

  EXPECT_THAT(
      mapped(angles,
             [](Scalar angle) { return SO3Type::rotX(angle).angleX(); }),
      Pointwise(IsNear(kEpsilon), angles));
  EXPECT_THAT(
      mapped(angles,
             [](Scalar angle) { return SO3Type::rotY(angle).angleY(); }),
      Pointwise(IsNear(kEpsilon), angles));
  EXPECT_THAT(
      mapped(angles,
             [](Scalar angle) { return SO3Type::rotZ(angle).angleZ(); }),
      Pointwise(IsNear(kEpsilon), angles));
}

}  // namespace
}  // namespace Sophus
