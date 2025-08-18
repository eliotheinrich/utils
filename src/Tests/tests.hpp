#pragma once

#include <memory>
#include <random>
#include <complex>
#include <chrono>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <iostream>

#define GET_MACRO(_1, _2, NAME, ...) NAME
#define ASSERT(...) GET_MACRO(__VA_ARGS__, ASSERT_TWO_ARGS, ASSERT_ONE_ARG)(__VA_ARGS__)

#define ASSERT_ONE_ARG(x) if (!(x)) { return false; }

#define ASSERT_TWO_ARGS(x, y) \
  if (!(x)) {                 \
    std::cout << y << "\n";   \
    return false;             \
  }

template <>
struct fmt::formatter<std::complex<double>> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const std::complex<double>& c, FormatContext& ctx) {
    return format_to(ctx.out(), "{} + {}i", c.real(), c.imag());
  }
};

template <typename T, typename V>
bool is_close_eps(double eps, T first, V second) {
  return std::abs(first - second) < eps;
}

template <typename T, typename V>
bool is_close(T first, V second) {
  return is_close_eps(1e-8, first, second);
}

template <typename T, typename V, typename... Args>
bool is_close_eps(double eps, T first, V second, Args... args) {
  if (!is_close_eps(eps, first, second)) {
    return false;
  } else {
    return is_close_eps(eps, first, args...);
  }
}

template <typename T, typename V, typename... Args>
bool is_close(T first, V second, Args... args) {
  if (!is_close(first, second)) {
    return false;
  } else {
    return is_close(first, args...);
  }
}

using TestResult = std::tuple<bool, int>;

#define ADD_TEST(x)                                                               \
if (run_all || test_names.contains(#x)) {                                         \
  auto start = std::chrono::high_resolution_clock::now();                         \
  bool passed = x();                                                              \
  auto stop = std::chrono::high_resolution_clock::now();                          \
  int duration = duration_cast<std::chrono::microseconds>(stop - start).count();  \
  tests[#x "()"] = std::make_tuple(passed, duration);                             \
}                                                                                 \
