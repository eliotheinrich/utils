#pragma once

#include "QuantumState.h"
#include "QuantumCircuit.h"
#include "CliffordState.h"
#include "BinaryPolynomial.h"
#include "FreeFermion.h"
#include "Simulator.hpp"

#include <PyDataFrame.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/pair.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>

using namespace nanobind::literals;

#define EXPORT_SIMULATOR(A)                                                               \
  nanobind::class_<A>(m, #A)                                                              \
    .def(nanobind::init<dataframe::ExperimentParams&, uint32_t>())                        \
    .def_static("create_and_emplace", [](dataframe::ExperimentParams& params,             \
                                         uint32_t num_threads) {                          \
      A simulator(params, num_threads);                                                   \
      return std::make_pair(simulator, params);                                           \
    })                                                                                    \
    .def("init", [](                                                                      \
          A& self,                                                                        \
          const std::optional<nanobind::bytes>& data = std::nullopt) {                    \
          std::optional<std::vector<dataframe::byte_t>> _data;                            \
      if (data.has_value()) {                                                             \
        _data = convert_bytes(data.value());                                              \
      } else {                                                                            \
        _data = std::nullopt;                                                             \
      }                                                                                   \
      if (_data.has_value()) {                                                            \
        self.deserialize(_data.value());                                                  \
      }                                                                                   \
    }, "data"_a = nanobind::none())                                                       \
    .def("timesteps", &A::timesteps)                                                      \
    .def("get_texture", [](A& self) -> std::tuple<std::vector<float>, size_t , size_t> {  \
      Texture texture = self.get_texture();                                               \
      const float* data = texture.data();                                                 \
      return {std::vector<float>{data, data + texture.len()}, texture.n, texture.m};      \
    })                                                                                    \
    .def("key_callback", &A::key_callback)                                                \
    .def("equilibration_timesteps", [](A& self, uint32_t num_steps) {                     \
        self.equilibration_timesteps(num_steps);                                          \
    })                                                                                    \
    .def("take_samples", &A::take_samples)                                                \
    .def("serialize", [](A& self) {                                                       \
      return self.serialize();                                                            \
    });

#define EXPORT_CONFIG(A)                                              \
  nanobind::class_<A>(m, #A)                                          \
  .def(nanobind::init<dataframe::ExperimentParams&>())                \
  .def("compute", [](A& self, uint32_t num_threads) {                 \
      dataframe::DataSlide slide = self.compute(num_threads);         \
      std::vector<dataframe::byte_t> _bytes = slide.to_bytes();       \
      nanobind::bytes bytes = convert_bytes(_bytes);                  \
      return bytes;                                                   \
    })


