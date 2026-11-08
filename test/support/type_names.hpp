// SPDX-FileCopyrightText: 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#ifdef SOPHUS_CERES
#include <ceres/jet.h>
#endif

namespace Sophus {
namespace details {

template <class T>
struct TypeTag {};

inline std::string scalarName(TypeTag<double>) { return "double"; }
inline std::string scalarName(TypeTag<float>) { return "float"; }

#ifdef SOPHUS_CERES
template <class T, int N>
std::string scalarName(TypeTag<ceres::Jet<T, N>>) {
  return "Jet";
}
#endif

// Types other than scalars, such as Lie groups, are named after their scalar.
template <class T, class Scalar = typename T::Scalar>
std::string scalarName(TypeTag<T>) {
  return scalarName(TypeTag<Scalar>{});
}

}  // namespace details

/// Names typed test instantiations after their scalar type, such as
/// `SO3/LieGroupTest/float`, instead of after their index in the type list.
struct ScalarNameGenerator {
  template <class T>
  static std::string GetName(int /*index*/) {
    return details::scalarName(details::TypeTag<T>{});
  }
};

}  // namespace Sophus
