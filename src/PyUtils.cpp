#include <PyUtils.hpp>
#include <PyQutils.hpp>

NB_MODULE(utils_bindings, m) {
  nanobind::class_<EntropySampler>(m, "EntropySampler")
    .def(nanobind::init<dataframe::ExperimentParams&>())
    .def_static("create_and_emplace", [](dataframe::ExperimentParams& params) {
      auto sampler = std::make_shared<EntropySampler>(params);
      return std::make_pair(sampler, params);
    })
    .def("take_samples", [](EntropySampler& sampler, std::shared_ptr<EntanglementEntropyState> state) {
      dataframe::SampleMap samples;
      sampler.add_samples(samples, state);
      return samples;
    });

  nanobind::class_<InterfaceSampler>(m, "InterfaceSampler")
    .def(nanobind::init<dataframe::ExperimentParams&>())
    .def_static("create_and_emplace", [](dataframe::ExperimentParams& params) {
      auto sampler = std::make_shared<InterfaceSampler>(params);
      return std::make_pair(sampler, params);
    })
    .def("take_samples", [](InterfaceSampler& sampler, const std::vector<int>& surface) {
      dataframe::SampleMap samples;
      sampler.add_samples(samples, surface);
      return samples;
    });

  nanobind::class_<QuantumStateSampler>(m, "QuantumStateSampler")
    .def(nanobind::init<dataframe::ExperimentParams&>())
    .def_static("create_and_emplace", [](dataframe::ExperimentParams& params) {
      auto sampler = std::make_shared<QuantumStateSampler>(params);
      return std::make_pair(sampler, params);
    })
    .def("take_samples", [](QuantumStateSampler& sampler, const std::shared_ptr<MagicQuantumState>& state) {
      std::shared_ptr<QuantumState> qstate = std::dynamic_pointer_cast<QuantumState>(state);
      dataframe::SampleMap samples;
      sampler.add_samples(samples, qstate);
      return samples;
    });

  nanobind::class_<GenericParticipationSampler>(m, "GenericParticipationSampler")
    .def(nanobind::init<dataframe::ExperimentParams&>())
    .def_static("create_and_emplace", [](dataframe::ExperimentParams& params) {
      auto sampler = std::make_shared<GenericParticipationSampler>(params);
      return std::make_pair(sampler, params);
    })
    .def("take_samples", [](GenericParticipationSampler& sampler, const std::shared_ptr<MagicQuantumState>& state) {
      std::shared_ptr<QuantumState> qstate = std::dynamic_pointer_cast<QuantumState>(state);
      dataframe::SampleMap samples;
      sampler.add_samples(samples, qstate);
      return samples;
    });
    
  nanobind::class_<MPSParticipationSampler>(m, "MPSParticipationSampler")
    .def(nanobind::init<dataframe::ExperimentParams&>())
    .def_static("create_and_emplace", [](dataframe::ExperimentParams& params) {
      auto sampler = std::make_shared<MPSParticipationSampler>(params);
      return std::make_pair(sampler, params);
    })
    .def("take_samples", [](MPSParticipationSampler& sampler, const std::shared_ptr<MatrixProductState>& state) {
      dataframe::SampleMap samples;
      sampler.add_samples(samples, state);
      return samples;
    });

  nanobind::class_<CHPParticipationSampler>(m, "CHPParticipationSampler")
    .def(nanobind::init<dataframe::ExperimentParams&>())
    .def_static("create_and_emplace", [](dataframe::ExperimentParams& params) {
      auto sampler = std::make_shared<CHPParticipationSampler>(params);
      return std::make_pair(sampler, params);
    })
    .def("take_samples", [](CHPParticipationSampler& sampler, const std::shared_ptr<QuantumCHPState>& state) {
      dataframe::SampleMap samples;
      sampler.add_samples(samples, state);
      return samples;
    });

  nanobind::class_<GenericMagicSampler>(m, "GenericMagicSampler")
    .def(nanobind::init<dataframe::ExperimentParams&>())
    .def_static("create_and_emplace", [](dataframe::ExperimentParams& params) {
      auto sampler = std::make_shared<GenericMagicSampler>(params);
      return std::make_pair(sampler, params);
    })
    .def("set_sre_montecarlo_update", [](GenericMagicSampler& self, PyMutationFunc func) {
      auto mutation = convert_from_pyfunc(func);
      self.set_montecarlo_update(mutation);
    })
    .def("take_samples", [](GenericMagicSampler& sampler, const std::shared_ptr<MagicQuantumState>& state) {
      dataframe::SampleMap samples;
      sampler.add_samples(samples, state);
      return samples;
    });

  nanobind::class_<MPSMagicSampler>(m, "MPSMagicSampler")
    .def(nanobind::init<dataframe::ExperimentParams&>())
    .def_static("create_and_emplace", [](dataframe::ExperimentParams& params) {
      auto sampler = std::make_shared<MPSMagicSampler>(params);
      return std::make_pair(sampler, params);
    })
    .def("take_samples", [](MPSMagicSampler& sampler, const std::shared_ptr<MatrixProductState>& state) {
      dataframe::SampleMap samples;
      sampler.add_samples(samples, state);
      return samples;
    });

  nanobind::class_<BinaryMatrix>(m, "BinaryMatrix")
    .def(nanobind::init<size_t, size_t>())
    .def_ro("num_cols", &BinaryMatrix::num_cols)
    .def_ro("num_rows", &BinaryMatrix::num_rows)
    .def("set", &BinaryMatrix::set)
    .def("set", [](BinaryMatrix& self, size_t i, size_t j, size_t v) { self.set(i, j, static_cast<bool>(v)); })
    .def("get", &BinaryMatrix::get)
    .def("append_row", [](BinaryMatrix& self, const std::vector<int>& row) { 
        std::vector<bool> _row(row.size());
        std::transform(row.begin(), row.end(), _row.begin(),
            [](int value) { return static_cast<bool>(value); });
        self.append_row(_row);
      })
    .def("transpose", &BinaryMatrix::transpose)
    .def("__str__", [](BinaryMatrix& self) { return self.to_string(); })
    .def("rref", [](BinaryMatrix& self) { self.rref(); })
    .def("partial_rref", [](BinaryMatrix& self, const std::vector<int>& sites) { 
        std::vector<size_t> _sites(sites.size());
        std::transform(sites.begin(), sites.end(), _sites.begin(),
            [](int value) { return static_cast<size_t>(value); });
        self.partial_rref(_sites); 
      })
    .def("rank", [](BinaryMatrix& self, bool inplace) { return self.rank(inplace); }, "inplace"_a=false)
    .def("partial_rank", [](BinaryMatrix& self, const std::vector<int>& sites, bool inplace) { 
        std::vector<size_t> _sites(sites.size());
        std::transform(sites.begin(), sites.end(), _sites.begin(),
          [](int value) { return static_cast<size_t>(value); });
        return self.partial_rank(_sites, inplace); 
      }, "sites"_a, "inplace_"_a=false);

  nanobind::class_<ParityCheckMatrix, BinaryMatrix>(m, "ParityCheckMatrix")
    .def(nanobind::init<size_t, size_t>())
    .def("reduce", &ParityCheckMatrix::reduce)
    .def("to_generator_matrix", &ParityCheckMatrix::to_generator_matrix, "inplace"_a = false);

  nanobind::class_<GeneratorMatrix, BinaryMatrix>(m, "GeneratorMatrix")
    .def(nanobind::init<size_t, size_t>())
    .def("to_parity_check_matrix", &GeneratorMatrix::to_parity_check_matrix, "inplace"_a = false)
    .def("truncate", [](GeneratorMatrix& self, const std::vector<int>& sites) {
        std::vector<size_t> _sites(sites.size());
        std::transform(sites.begin(), sites.end(), _sites.begin(),
            [](int value) { return static_cast<size_t>(value); });
        return self.truncate(_sites);
      })
    .def("generator_locality", &GeneratorMatrix::generator_locality);

#ifdef BUILD_GLFW
#include "Display.h"
  constexpr int BUILT_WITH_GLFW = 1;
  nanobind::class_<FrameData>(m, "FrameData")
    .def(nanobind::init<int, std::vector<int>>())
    .def_ro("status_code", &FrameData::status_code)
    .def_ro("keys", &FrameData::keys);

  nanobind::class_<Animator>(m, "GLFWAnimator")
    .def("__init__", [](Animator *s, float r, float g, float b, float alpha) {
      new (s) Animator({r, g, b, alpha});
    })
    .def("is_paused", &Animator::is_paused)
    .def("new_frame", [](Animator& self, const std::vector<float>& texture, size_t n, size_t m) { return self.new_frame(texture, n, m); })
    .def("new_frame", [](Animator& self, const std::vector<std::vector<std::vector<float>>>& data) { return self.new_frame(Texture(data)); })
    .def("start", &Animator::start);
#else
  constexpr int BUILT_WITH_GLFW = 0;
#endif
  m.attr("QUTILS_BUILT_WITH_GLFW") = BUILT_WITH_GLFW;
}
