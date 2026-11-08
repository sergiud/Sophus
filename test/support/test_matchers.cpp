// SPDX-FileCopyrightText: 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>
// SPDX-License-Identifier: MIT

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <sophus/so2.hpp>

#include "matchers.hpp"
#include "registration.hpp"
#include "type_names.hpp"

namespace Sophus {
namespace {

using ::testing::DescribeMatcher;
using ::testing::ExplainMatchResult;
using ::testing::HasSubstr;
using ::testing::Lt;
using ::testing::Not;
using ::testing::PrintToString;
using ::testing::StringMatchResultListener;

[[maybe_unused]] bool const kRegistered =
    (registerTest(
         {"RegisteredTest", "Name/0", "float", "value", __FILE__, __LINE__},
         [] {}),
     true);

/// Returns the test registered by kRegistered, or null if there is none.
::testing::TestInfo const* findRegisteredTest() {
  ::testing::UnitTest const& unit_test = *::testing::UnitTest::GetInstance();
  for (int i = 0; i < unit_test.total_test_suite_count(); ++i) {
    ::testing::TestSuite const& suite = *unit_test.GetTestSuite(i);
    if (std::string(suite.name()) != "RegisteredTest") {
      continue;
    }
    for (int j = 0; j < suite.total_test_count(); ++j) {
      if (std::string(suite.GetTestInfo(j)->name()) == "Name/0") {
        return suite.GetTestInfo(j);
      }
    }
  }
  return nullptr;
}

template <class Scalar>
class MatchersTest : public ::testing::Test {};

using FloatingPointScalars = ::testing::Types<double, float>;
TYPED_TEST_SUITE(MatchersTest, FloatingPointScalars, ScalarNameGenerator);

TYPED_TEST(MatchersTest, IsNearAcceptsScalarsWithinInclusiveTolerance) {
  using Scalar = TypeParam;
  EXPECT_THAT(Scalar(1.25), IsNear(Scalar(1), Scalar(0.25)));
  EXPECT_THAT(Scalar(0.75), IsNear(Scalar(1), Scalar(0.25)));
  EXPECT_THAT(Scalar(1.5), Not(IsNear(Scalar(1), Scalar(0.25))));
  EXPECT_THAT(Scalar(0.5), Not(IsNear(Scalar(1), Scalar(0.25))));
}

TYPED_TEST(MatchersTest, IsNearRejectsNonFiniteValues) {
  using Scalar = TypeParam;
  Scalar const kNaN = std::numeric_limits<Scalar>::quiet_NaN();
  Scalar const kInfinity = std::numeric_limits<Scalar>::infinity();
  Scalar const kLargeTolerance = std::numeric_limits<Scalar>::max();
  EXPECT_THAT(kNaN, Not(IsNear(Scalar(0), kLargeTolerance)));
  EXPECT_THAT(kInfinity, Not(IsNear(Scalar(0), kLargeTolerance)));
}

TYPED_TEST(MatchersTest, IsNearComparesMatricesByLargestAbsoluteDifference) {
  using Scalar = TypeParam;
  Matrix2<Scalar> expected;
  expected << Scalar(1), Scalar(2), Scalar(3), Scalar(4);
  Matrix2<Scalar> offset;
  offset << Scalar(0.5), Scalar(-0.5), Scalar(0.5), Scalar(-0.5);

  // The Frobenius norm of the offset is 1, but each entry differs by 0.5.
  EXPECT_THAT((expected + offset).eval(), IsNear(expected, Scalar(0.5)));
  EXPECT_THAT((expected + offset).eval(), Not(IsNear(expected, Scalar(0.25))));

  Matrix2<Scalar> one_entry_off = expected;
  one_entry_off(1, 0) += Scalar(0.5);
  EXPECT_THAT(one_entry_off, Not(IsNear(expected, Scalar(0.25))));
}

TYPED_TEST(MatchersTest, IsNearAcceptsEigenExpressions) {
  using Scalar = TypeParam;
  Vector3<Scalar> const v(Scalar(1), Scalar(2), Scalar(3));
  EXPECT_THAT(v.template head<2>(),
              IsNear(Vector2<Scalar>(Scalar(1), Scalar(2)), Scalar(0)));
  EXPECT_THAT(v, IsNear(Scalar(2) * v - v, Scalar(0)));
  EXPECT_THAT(v.template tail<2>(),
              Not(IsNear(v.template head<2>(), Scalar(0.5))));
}

TYPED_TEST(MatchersTest, IsNearComparesLieGroupsByTheirMatrices) {
  using Scalar = TypeParam;
  SO2<Scalar> const rotation(Scalar(0.5));
  EXPECT_THAT(SO2<Scalar>(Scalar(0.5)), IsNear(rotation, Scalar(0)));
  EXPECT_THAT(SO2<Scalar>(Scalar(0.7)), Not(IsNear(rotation, Scalar(0.1))));
  // Both rotations differ by less than 0.1 in every entry of the matrix.
  EXPECT_THAT(SO2<Scalar>(Scalar(0.55)), IsNear(rotation, Scalar(0.1)));
}

TYPED_TEST(MatchersTest, IsNearExplainsTheDistanceAndDescribesTheTolerance) {
  using Scalar = TypeParam;
  Vector2<Scalar> const expected(Scalar(1), Scalar(2));
  Vector2<Scalar> const actual(Scalar(1), Scalar(2.5));
  auto const matcher = IsNear(expected, Scalar(0.125));

  StringMatchResultListener listener;
  EXPECT_FALSE(ExplainMatchResult(matcher, actual, &listener));
  EXPECT_THAT(listener.str(), HasSubstr(PrintToString(Scalar(0.5))));

  std::string const description =
      DescribeMatcher<Vector2<Scalar>>(matcher, false);
  EXPECT_THAT(description, HasSubstr(PrintToString(Scalar(0.125))));
  EXPECT_THAT(description, HasSubstr(PrintToString(expected)));
  EXPECT_THAT(DescribeMatcher<Vector2<Scalar>>(matcher, true),
              HasSubstr("isn't"));
}

TYPED_TEST(MatchersTest, IsFiniteRejectsInfinityAndNaN) {
  using Scalar = TypeParam;
  EXPECT_THAT(Scalar(1), IsFinite());
  EXPECT_THAT(std::numeric_limits<Scalar>::max(), IsFinite());
  EXPECT_THAT(std::numeric_limits<Scalar>::infinity(), Not(IsFinite()));
  EXPECT_THAT(-std::numeric_limits<Scalar>::infinity(), Not(IsFinite()));
  EXPECT_THAT(std::numeric_limits<Scalar>::quiet_NaN(), Not(IsFinite()));
}

TYPED_TEST(MatchersTest, HasSquaredNormAppliesTheMatcherToTheSquaredNorm) {
  using Scalar = TypeParam;
  Vector2<Scalar> const v(Scalar(3), Scalar(4));
  EXPECT_THAT(v, HasSquaredNorm(Lt(Scalar(26))));
  EXPECT_THAT(v, Not(HasSquaredNorm(Lt(Scalar(25)))));
  EXPECT_THAT(Scalar(-3), HasSquaredNorm(Lt(Scalar(10))));
  EXPECT_THAT(Scalar(-3), Not(HasSquaredNorm(Lt(Scalar(9)))));

  StringMatchResultListener listener;
  EXPECT_FALSE(ExplainMatchResult(HasSquaredNorm(Lt(Scalar(1))), v, &listener));
  EXPECT_THAT(listener.str(), HasSubstr(PrintToString(Scalar(25))));
}

TYPED_TEST(MatchersTest, IsScaledOrthogonalAndPositiveRejectsReflections) {
  using Scalar = TypeParam;
  Matrix2<Scalar> const scaled_rotation =
      Scalar(2.5) * SO2<Scalar>(Scalar(0.3)).matrix();
  EXPECT_THAT(scaled_rotation, IsScaledOrthogonalAndPositive());

  Matrix2<Scalar> reflection;
  reflection << scaled_rotation.col(1), scaled_rotation.col(0);
  EXPECT_THAT(reflection, Not(IsScaledOrthogonalAndPositive()));

  Matrix2<Scalar> sheared = scaled_rotation;
  sheared(0, 1) += Scalar(1);
  EXPECT_THAT(sheared, Not(IsScaledOrthogonalAndPositive()));
}

TYPED_TEST(MatchersTest, PrintsMatricesRowByRowOnTheirOwnLines) {
  using Scalar = TypeParam;
  Matrix2<Scalar> m;
  m << Scalar(1), Scalar(2), Scalar(3), Scalar(4);
  EXPECT_EQ(PrintToString(m), "\n[[1, 2],\n [3, 4]]");
}

TYPED_TEST(MatchersTest, PrintsVectorsOnOneLine) {
  using Scalar = TypeParam;
  EXPECT_EQ(PrintToString(Vector3<Scalar>(Scalar(1), Scalar(2), Scalar(3))),
            "[1, 2, 3]");
}

TYPED_TEST(MatchersTest, PrintsLieGroupsAsTheirMatrices) {
  using Scalar = TypeParam;
  SO2<Scalar> const identity;
  EXPECT_EQ(PrintToString(identity), PrintToString(identity.matrix()));
}

TYPED_TEST(MatchersTest, IsNearWithOnlyAToleranceComparesPairsPointwise) {
  using Scalar = TypeParam;
  using ::testing::Pointwise;
  std::vector<Vector2<Scalar>> const expected = {
      Vector2<Scalar>(Scalar(1), Scalar(2)),
      Vector2<Scalar>(Scalar(3), Scalar(4))};
  std::vector<Vector2<Scalar>> actual = expected;
  actual[1].y() += Scalar(0.25);

  EXPECT_THAT(actual, Pointwise(IsNear(Scalar(0.25)), expected));
  EXPECT_THAT(actual, Not(Pointwise(IsNear(Scalar(0.125)), expected)));

  StringMatchResultListener listener;
  EXPECT_FALSE(ExplainMatchResult(Pointwise(IsNear(Scalar(0.125)), expected),
                                  actual, &listener));
  EXPECT_THAT(listener.str(), HasSubstr("#1"));
  EXPECT_THAT(listener.str(), HasSubstr(PrintToString(Scalar(0.25))));
}

TYPED_TEST(MatchersTest, IsNearQuaternionAcceptsBothCoversOfARotation) {
  using Scalar = TypeParam;
  using Quaternion = Eigen::Quaternion<Scalar>;
  Quaternion const q(Scalar(0.5), Scalar(0.5), Scalar(-0.5), Scalar(0.5));
  Quaternion const minus_q(-q.coeffs());
  Quaternion near_q = q;
  near_q.x() += Scalar(0.125);

  EXPECT_THAT(q, IsNearQuaternion(q, Scalar(0)));
  EXPECT_THAT(minus_q, IsNearQuaternion(q, Scalar(0)));
  EXPECT_THAT(near_q, IsNearQuaternion(q, Scalar(0.125)));
  EXPECT_THAT(near_q, Not(IsNearQuaternion(q, Scalar(0.0625))));
  EXPECT_THAT(Quaternion(-near_q.coeffs()), IsNearQuaternion(q, Scalar(0.125)));
  EXPECT_THAT(Quaternion(-near_q.coeffs()),
              Not(IsNearQuaternion(q, Scalar(0.0625))));
}

TYPED_TEST(MatchersTest, IsNearQuaternionRejectsADifferentRotation) {
  using Scalar = TypeParam;
  using Quaternion = Eigen::Quaternion<Scalar>;
  Quaternion const q(Scalar(1), Scalar(0), Scalar(0), Scalar(0));
  Quaternion const other(Scalar(0), Scalar(1), Scalar(0), Scalar(0));
  EXPECT_THAT(other, Not(IsNearQuaternion(q, Scalar(0.5))));

  StringMatchResultListener listener;
  EXPECT_FALSE(
      ExplainMatchResult(IsNearQuaternion(q, Scalar(0.5)), other, &listener));
  EXPECT_THAT(listener.str(), HasSubstr(PrintToString(Scalar(1))));
  EXPECT_THAT(DescribeMatcher<Quaternion>(IsNearQuaternion(q, Scalar(0.5))),
              HasSubstr("sign"));
}

TEST(RegisterTest, RegistersTheTestWithItsNamesAndParameters) {
  ::testing::TestInfo const* const info = findRegisteredTest();
  ASSERT_THAT(info, ::testing::NotNull());
  EXPECT_STREQ(info->test_suite_name(), "RegisteredTest");
  EXPECT_STREQ(info->type_param(), "float");
  EXPECT_STREQ(info->value_param(), "value");
  EXPECT_STREQ(info->file(), __FILE__);
}

TEST(RegisterTest, FunctionTestRunsItsBodyOnce) {
  int runs = 0;
  details::FunctionTest test([&runs] { ++runs; });
  test.TestBody();
  EXPECT_EQ(runs, 1);
}

TEST(RegisterTest, FactoryCreatesTestsThatRunTheBody) {
  int runs = 0;
  auto const factory = details::makeFunctionTestFactory([&runs] { ++runs; });
  std::unique_ptr<details::FunctionTest> const test(factory());
  test->TestBody();
  EXPECT_EQ(runs, 1);
}

}  // namespace
}  // namespace Sophus
