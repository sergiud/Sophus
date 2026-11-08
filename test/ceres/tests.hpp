// Copyright 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>

#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <ceres/ceres.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <Eigen/StdVector>

#include <sophus/ceres_manifold.hpp>
#include <sophus/types.hpp>

#include "matchers.hpp"
#include "registration.hpp"
#include "type_names.hpp"

/// The norm of the rotational part of a tangent vector. Each test of a Lie
/// group specializes it, since the layout of the tangent differs per group.
template <typename LieGroup>
struct RotationalPart;

namespace Sophus {

template <class T>
using StdVector = std::vector<T, Eigen::aligned_allocator<T>>;

/// The group elements and points the Ceres tests of a Lie group run on. Each
/// test of a Lie group specializes it with the static member functions
/// `groupElements` and `points`.
template <class LieGroup>
struct LieGroupCeresTestData;

/// Recovers the class template of a Lie group, such as `SO3` from `SO3d`, which
/// `Sophus::Manifold` and the cost functors instantiate for other scalars.
template <class LieGroup>
struct GroupTemplate;

template <template <typename, int> class LieGroup, class Scalar, int Options>
struct GroupTemplate<LieGroup<Scalar, Options>> {
  template <typename T, int Options_ = 0>
  using Type = LieGroup<T, Options_>;
};

/// The Lie group `LieGroupd` with its scalar replaced by `T`.
template <class LieGroupd, class T>
using Rebind = typename GroupTemplate<LieGroupd>::template Type<T>;

/// The Ceres manifold of the Lie group `LieGroupd`.
template <class LieGroupd>
using ManifoldOf = Manifold<GroupTemplate<LieGroupd>::template Type>;

template <typename T>
typename T::Scalar* data(T& t) {
  return t.data();
}

inline double* data(double& d) { return &d; }

template <typename T>
const typename T::Scalar* data(const T& t) {
  return t.data();
}

inline const double* data(const double& d) { return &d; }

template <typename T>
struct Random {
  template <typename R>
  static T sample(R& rng) {
    std::normal_distribution<double> rnorm;
    static_assert(T::RowsAtCompileTime >= 0,
                  "Matrix should have known size at compile-time");
    static_assert(T::ColsAtCompileTime >= 0,
                  "Matrix should have known size at compile-time");
    T res;
    for (Eigen::Index i = 0; i < res.size(); ++i) {
      res.data()[i] = rnorm(rng);
    }
    return res;
  }
};

template <>
struct Random<double> {
  template <typename R>
  static double sample(R& rng) {
    std::normal_distribution<double> rnorm;
    return rnorm(rng);
  }
};

template <class LieGroupd>
struct TestLieGroupCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit TestLieGroupCostFunctor(const LieGroupd& T_aw) : T_aw(T_aw) {}

  template <class T>
  bool operator()(T const* const sT_wa, T* sResiduals) const {
    using LieGroupT = Rebind<LieGroupd, T>;
    Eigen::Map<LieGroupT const> const T_wa(sT_wa);
    // The Mapper class makes the functor work both for groups whose tangent
    // is a scalar, such as SO2, and for groups whose tangent is a vector.
    using Mapper = Mapper<typename LieGroupT::Tangent>;
    typename Mapper::Map residuals = Mapper::map(sResiduals);

    // Sophus types mix with doubles and Jet types without a cast to T, which
    // mixedProductsMatchJetProducts checks.
    residuals = (T_aw * T_wa).log();
    return true;
  }

  LieGroupd T_aw;
};

template <class LieGroupd>
struct TestPointCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Pointd = typename LieGroupd::Point;

  TestPointCostFunctor(const LieGroupd& T_aw, const Pointd& point_a)
      : T_aw(T_aw), point_a(point_a) {}

  template <class T>
  bool operator()(T const* const sT_wa, T const* const spoint_b,
                  T* sResiduals) const {
    using LieGroupT = Rebind<LieGroupd, T>;
    using PointT = typename LieGroupT::Point;
    Eigen::Map<LieGroupT const> const T_wa(sT_wa);
    Eigen::Map<PointT const> point_b(spoint_b);
    Eigen::Map<PointT> residuals(sResiduals);

    // The double group acts on a Jet point and the Jet group on a double
    // point, which mixedProductsMatchJetProducts checks.
    PointT const point_b_prime = T_aw * point_b;
    PointT const point_a_prime = T_wa * point_a;
    residuals = point_b_prime - point_a_prime;
    return true;
  }

  LieGroupd T_aw;
  Pointd point_a;
};

template <class LieGroupd>
struct TestGraphFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit TestGraphFunctor(const LieGroupd& diff) : diff(diff) {}

  template <typename T>
  bool operator()(const T* a, const T* b, T* residuals) const {
    using LieGroupT = Rebind<LieGroupd, T>;
    Eigen::Map<const LieGroupT> A(a);
    Eigen::Map<const LieGroupT> B(b);
    using Mapper = Mapper<typename LieGroupT::Tangent>;
    typename Mapper::Map diff_log = Mapper::map(residuals);

    // The double group multiplied by a Jet group.
    diff_log = (diff * (B.inverse() * A)).log();
    return true;
  }

  const LieGroupd diff;
};

/// Registers the tests of optimization with Ceres over the parameters of a
/// Lie group, and of the invariants of its Ceres manifold that relate Plus
/// and Minus. The derivative of Minus is checked against a numerical
/// derivative by the generic tests of the group. Ceres optimizes `double`
/// parameter blocks only, so the tests of a group named `SO3` form the suite
/// `SO3/LieGroupCeresTest/double`, with one test per entry of its
/// `LieGroupCeresTestData`, such as `XMinusXIsZero/3`.
template <class LieGroupd>
class LieGroupCeresTests {
 public:
  using Pointd = typename LieGroupd::Point;
  using Tangentd = typename LieGroupd::Tangent;
  using Manifoldd = ManifoldOf<LieGroupd>;
  static int constexpr kNumParameters = LieGroupd::num_parameters;
  static int constexpr kDoF = LieGroupd::DoF;

  static bool registerAll(std::string const& group_name);

 private:
  static double epsilon() { return Constants<double>::epsilon(); }

  static Tangentd zeroTangent() {
    Tangentd zero;
    setToZero(zero);
    return zero;
  }

  static std::string scalarName() {
    return details::scalarName(details::TypeTag<LieGroupd>{});
  }

  static std::string describe(LieGroupd const& g) {
    return ::testing::PrintToString(g.params());
  }

  /// Joins indices of test data to the name of a test, such as `3_5`.
  template <class... Indices>
  static std::string indexName(Indices... indices) {
    std::string name;
    ((name += (name.empty() ? "" : "_") + std::to_string(indices)), ...);
    return name;
  }

  static void add(std::string const& suite, std::string const& name,
                  std::string const& value_param, int line,
                  std::function<void()> body) {
    registerTest({suite, name, scalarName(), value_param, __FILE__, line},
                 std::move(body));
  }

  /// Optimizes a group element initialized with `T_w_init` towards
  /// `T_w_targ` and returns the result.
  static LieGroupd optimizePose(LieGroupd const& T_w_targ,
                                LieGroupd const& T_w_init,
                                Pointd const& point_a_init,
                                Pointd const& point_b) {
    int constexpr kNumPointParameters = Pointd::RowsAtCompileTime;
    double constexpr kToleranceFactor = 0.01;
    int constexpr kMaxNumIterations = 100;

    LieGroupd T_wr = T_w_init;
    Pointd point_a = point_a_init;

    ceres::Problem problem;
    problem.AddParameterBlock(T_wr.data(), kNumParameters, new Manifoldd);

    // Derivatives are evaluated by automatic differentiation.
    ceres::CostFunction* cost_function1 =
        new ceres::AutoDiffCostFunction<TestLieGroupCostFunctor<LieGroupd>,
                                        kDoF, kNumParameters>(
            new TestLieGroupCostFunctor<LieGroupd>(T_w_targ.inverse()));
    problem.AddResidualBlock(cost_function1, nullptr, T_wr.data());
    ceres::CostFunction* cost_function2 =
        new ceres::AutoDiffCostFunction<TestPointCostFunctor<LieGroupd>,
                                        kNumPointParameters, kNumParameters,
                                        kNumPointParameters>(
            new TestPointCostFunctor<LieGroupd>(T_w_targ.inverse(), point_b));
    problem.AddResidualBlock(cost_function2, nullptr, T_wr.data(),
                             point_a.data());

    ceres::Solver::Options options;
    options.gradient_tolerance = kToleranceFactor * epsilon();
    options.function_tolerance = kToleranceFactor * epsilon();
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = kMaxNumIterations;

    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);
    return T_wr;
  }

  static void poseOptimizationConvergesToTarget(
      LieGroupd const& T_w_targ, LieGroupd const& T_w_init,
      StdVector<Pointd> const& points) {
    using ::testing::Each;
    using ::testing::Lt;
    std::vector<double> squared_errors;
    for (std::size_t k = 0; k < points.size(); ++k) {
      for (std::size_t l = 0; l < points.size(); ++l) {
        if (k != l) {
          LieGroupd const T_wr =
              optimizePose(T_w_targ, T_w_init, points[k], points[l]);
          squared_errors.push_back(
              squaredNorm((T_w_targ.inverse() * T_wr).log()));
        }
      }
    }
    EXPECT_THAT(squared_errors, Each(Lt(10. * epsilon())))
        << "for each ordered pair of distinct points";
  }

  static void xPlusZeroIsX(LieGroupd const& x) {
    using ::testing::Lt;
    Manifoldd const manifold;
    Tangentd const zero = zeroTangent();
    LieGroupd test_group;
    ASSERT_TRUE(manifold.Plus(x.data(), data(zero), test_group.data()));
    EXPECT_THAT((x.inverse() * test_group).log(),
                HasSquaredNorm(Lt(epsilon())));
  }

  static void xMinusXIsZero(LieGroupd const& x) {
    using ::testing::Lt;
    Manifoldd const manifold;
    Tangentd test_tangent;
    ASSERT_TRUE(manifold.Minus(x.data(), x.data(), data(test_tangent)));
    EXPECT_THAT(test_tangent, HasSquaredNorm(Lt(epsilon())));
  }

  static void minusPlusIsIdentity(LieGroupd const& x, Tangentd const& delta) {
    using ::testing::Lt;
    Manifoldd const manifold;
    LieGroupd test_group;
    Tangentd test_tangent;
    ASSERT_TRUE(manifold.Plus(x.data(), data(delta), test_group.data()));
    ASSERT_TRUE(
        manifold.Minus(test_group.data(), x.data(), data(test_tangent)));
    Tangentd const diff = test_tangent - delta;
    EXPECT_THAT(diff, HasSquaredNorm(Lt(epsilon())));
  }

  static void plusMinusIsIdentity(LieGroupd const& x, LieGroupd const& y) {
    using ::testing::Lt;
    Manifoldd const manifold;
    LieGroupd test_group;
    Tangentd test_tangent;
    ASSERT_TRUE(manifold.Minus(y.data(), x.data(), data(test_tangent)));
    ASSERT_TRUE(manifold.Plus(x.data(), data(test_tangent), test_group.data()));
    EXPECT_THAT((y.inverse() * test_group).log(),
                HasSquaredNorm(Lt(epsilon())));
  }

  static void minusJacobianTimesPlusJacobianIsIdentity(LieGroupd const& x) {
    Manifoldd const manifold;
    Eigen::Matrix<double, kNumParameters, kDoF,
                  kDoF == 1 ? Eigen::ColMajor : Eigen::RowMajor>
        Jplus;
    Eigen::Matrix<double, kDoF, kNumParameters, Eigen::RowMajor> Jminus;
    ASSERT_TRUE(manifold.PlusJacobian(x.data(), Jplus.data()));
    ASSERT_TRUE(manifold.MinusJacobian(x.data(), Jminus.data()));

    // Sophus::squaredNorm accepts vectors only, so the difference is
    // flattened to bound its squared Frobenius norm.
    Eigen::Matrix<double, kDoF, kDoF> const product = Jminus * Jplus;
    EXPECT_THAT(
        (product - Eigen::Matrix<double, kDoF, kDoF>::Identity()).reshaped(),
        HasSquaredNorm(::testing::Lt(epsilon())));
  }

  static void averagingOnCompleteGraphReducesErrors(std::size_t num_vertices) {
    using ::testing::Lt;
    double constexpr kSigmaInit = 0.5;
    double constexpr kSigmaObservation = 0.1;
    double constexpr kToleranceFactor = 1e-2;
    double constexpr kRequiredReduction = 0.25;
    // A fixed seed keeps the test repeatable.
    unsigned constexpr kSeed = 2021;

    // The noise is scaled per element such that the expected squared norm is
    // the same for all groups.
    double const sigma_init_elementwise = kSigmaInit / std::sqrt(kDoF);
    double const sigma_observation_elementwise =
        kSigmaObservation / std::sqrt(kDoF);

    // Lie group averaging on a complete graph with random initialization
    // noise and random noise in the observations.
    ceres::Problem problem;
    std::mt19937 rng(kSeed);
    StdVector<LieGroupd> V(num_vertices);
    StdVector<LieGroupd> V_estimate;
    V_estimate.reserve(num_vertices);
    double initial_error = 0.;
    auto* const manifold = new Manifoldd;

    // All vertices are initialized with i.i.d. normally distributed noise.
    for (std::size_t i = 0; i < num_vertices; ++i) {
      auto& v = V[i];
      v = LieGroupd::sampleUniform(rng);
      Tangentd const delta_log =
          Random<Tangentd>::sample(rng) * sigma_init_elementwise;
      LieGroupd const delta = LieGroupd::exp(delta_log);
      V_estimate.emplace_back(v * delta);
      initial_error += squaredNorm(delta_log);
      problem.AddParameterBlock(V_estimate.back().data(), kNumParameters,
                                manifold);
    }

    // Each edge, an observation of the complete undirected graph, has i.i.d.
    // multivariate normally distributed noise.
    for (std::size_t i = 0; i < num_vertices; ++i) {
      for (std::size_t j = i + 1; j < num_vertices; ++j) {
        LieGroupd const diff = V[i].inverse() * V[j];
        Tangentd const delta_log =
            Random<Tangentd>::sample(rng) * sigma_observation_elementwise;
        LieGroupd const delta = LieGroupd::exp(delta_log);
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<TestGraphFunctor<LieGroupd>, kDoF,
                                            kNumParameters, kNumParameters>(
                new TestGraphFunctor<LieGroupd>(diff * delta));
        problem.AddResidualBlock(cost, nullptr, V_estimate[i].data(),
                                 V_estimate[j].data());
      }
    }

    ceres::Solver::Options options;
    options.gradient_tolerance = kToleranceFactor * epsilon();
    options.function_tolerance = kToleranceFactor * epsilon();
    options.parameter_tolerance = kToleranceFactor * epsilon();
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    double final_error = 0.;
    for (std::size_t i = 0; i < num_vertices; ++i) {
      final_error += squaredNorm((V[i].inverse() * V_estimate[i]).log());
    }

    EXPECT_THAT(summary.final_cost,
                Lt(kRequiredReduction * summary.initial_cost));
    EXPECT_THAT(final_error, Lt(kRequiredReduction * initial_error));
  }

  /// Returns the values and the derivatives of Jet coefficients side by side.
  template <class Derived>
  static Eigen::MatrixXd valuesAndDerivatives(
      Eigen::MatrixBase<Derived> const& jets) {
    using Jet = typename Derived::Scalar;
    Eigen::MatrixXd result(jets.size(), 1 + Jet::DIMENSION);
    for (Eigen::Index i = 0; i < jets.size(); ++i) {
      result(i, 0) = jets(i).a;
      result.row(i).tail(Jet::DIMENSION) = jets(i).v.transpose();
    }
    return result;
  }

  /// Checks that products mixing double and Jet operands equal the products
  /// of Jet operands, in values and derivatives.
  static void mixedProductsMatchJetProducts(LieGroupd const& x,
                                            LieGroupd const& y,
                                            Pointd const& point) {
    using Jet = ceres::Jet<double, kNumParameters>;
    using LieGroupJet = Rebind<LieGroupd, Jet>;
    using PointJet = typename LieGroupJet::Point;
    Eigen::Matrix<Jet, kNumParameters, 1> y_params;
    for (int i = 0; i < kNumParameters; ++i) {
      y_params[i] = Jet(y.params()[i], i);
    }
    Eigen::Map<LieGroupJet const> const y_jet(y_params.data());
    LieGroupJet const x_jet = x.template cast<Jet>();
    PointJet const point_jet = y_jet * point.template cast<Jet>();

    EXPECT_THAT(
        valuesAndDerivatives(LieGroupJet(x * y_jet).params()),
        IsNear(valuesAndDerivatives(LieGroupJet(x_jet * y_jet).params()),
               epsilon()))
        << "(double, Jet) product";
    EXPECT_THAT(
        valuesAndDerivatives(LieGroupJet(y_jet * x).params()),
        IsNear(valuesAndDerivatives(LieGroupJet(y_jet * x_jet).params()),
               epsilon()))
        << "(Jet, double) product";
    EXPECT_THAT(
        valuesAndDerivatives(PointJet(x * point_jet)),
        IsNear(valuesAndDerivatives(PointJet(x_jet * point_jet)), epsilon()))
        << "double group acting on a Jet point";
    EXPECT_THAT(valuesAndDerivatives(PointJet(y_jet * point)),
                IsNear(valuesAndDerivatives(
                           PointJet(y_jet * point.template cast<Jet>())),
                       epsilon()))
        << "Jet group acting on a double point";
  }
};

template <class LieGroupd>
bool LieGroupCeresTests<LieGroupd>::registerAll(std::string const& group_name) {
  std::string const suite = group_name + "/LieGroupCeresTest/" + scalarName();
  StdVector<LieGroupd> const g =
      LieGroupCeresTestData<LieGroupd>::groupElements();
  StdVector<Pointd> const points = LieGroupCeresTestData<LieGroupd>::points();

  for (std::size_t i = 0; i < g.size(); ++i) {
    std::string const value = "x = " + describe(g[i]);
    add(suite, "XPlusZeroIsX/" + indexName(i), value, __LINE__,
        [x = g[i]] { xPlusZeroIsX(x); });
    add(suite, "XMinusXIsZero/" + indexName(i), value, __LINE__,
        [x = g[i]] { xMinusXIsZero(x); });
    add(suite, "MinusJacobianTimesPlusJacobianIsIdentity/" + indexName(i),
        value, __LINE__,
        [x = g[i]] { minusJacobianTimesPlusJacobianIsIdentity(x); });
  }

  for (std::size_t i = 0; i < g.size(); ++i) {
    for (std::size_t j = 0; j < g.size(); ++j) {
      std::string const pair_value =
          "x = " + describe(g[i]) + ", y = " + describe(g[j]);
      if (i != j) {
        add(suite, "PoseOptimizationConvergesToTarget/" + indexName(i, j),
            "target " + describe(g[i]) + ", initial " + describe(g[j]),
            __LINE__, [target = g[i], initial = g[j], points] {
              poseOptimizationConvergesToTarget(target, initial, points);
            });
      }
      add(suite, "PlusMinusIsIdentity/" + indexName(i, j), pair_value, __LINE__,
          [x = g[i], y = g[j]] { plusMinusIsIdentity(x, y); });
      add(suite, "MixedProductsMatchJetProducts/" + indexName(i, j), pair_value,
          __LINE__, [x = g[i], y = g[j], point = points.front()] {
            mixedProductsMatchJetProducts(x, y, point);
          });

      // log recovers delta = log(x^-1 * y) from x * exp(delta) only if the
      // rotational part of delta is below pi. Pairs rotated by more than
      // pi * (1 - epsilon) relative to each other therefore have no test.
      Tangentd const delta = (g[i].inverse() * g[j]).log();
      if (RotationalPart<LieGroupd>::Norm(delta) <=
          Constants<double>::pi() * (1. - epsilon())) {
        add(suite, "MinusPlusIsIdentity/" + indexName(i, j), pair_value,
            __LINE__, [x = g[i], delta] { minusPlusIsIdentity(x, delta); });
      }
    }
  }

  for (std::size_t const num_vertices : {20u, 40u, 80u, 160u}) {
    add(suite,
        "AveragingOnCompleteGraphReducesErrors/" + indexName(num_vertices),
        "number of vertices = " + std::to_string(num_vertices), __LINE__,
        [num_vertices] {
          averagingOnCompleteGraphReducesErrors(num_vertices);
        });
  }
  return true;
}

}  // namespace Sophus
