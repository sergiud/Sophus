// SPDX-FileCopyrightText: 2026 Sergiu Deitsch <sergiu.deitsch@gmail.com>
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <string>
#include <utility>

#include <gtest/gtest.h>

namespace Sophus {
namespace details {

/// The fixture of tests registered with `registerTest`, which runs a callable
/// as the test body.
class FunctionTest : public ::testing::Test {
 public:
  explicit FunctionTest(std::function<void()> body) : body_(std::move(body)) {}

  void TestBody() override { body_(); }

 private:
  std::function<void()> body_;
};

}  // namespace details

/// Names and locates a test registered with `registerTest`.
struct TestId {
  std::string suite;
  std::string name;
  std::string type_param;
  std::string value_param;
  char const* file;
  int line;
};

/// Registers a test at run time whose body is `body`. Must be called before
/// `RUN_ALL_TESTS`, for example from a static initializer.
inline void registerTest(TestId const& id, std::function<void()> body) {
  ::testing::RegisterTest(
      id.suite.c_str(), id.name.c_str(),
      id.type_param.empty() ? nullptr : id.type_param.c_str(),
      id.value_param.empty() ? nullptr : id.value_param.c_str(), id.file,
      id.line, [body = std::move(body)]() -> details::FunctionTest* {
        return new details::FunctionTest(body);
      });
}

}  // namespace Sophus
