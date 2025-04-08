#include <PyQutils.hpp>
#include <memory>

using PyMutationFunc = std::function<PauliString(PauliString)>;
inline PauliMutationFunc convert_from_pyfunc(PyMutationFunc func) {
  return [func](PauliString& p) { p = func(p); };
}

inline std::vector<dataframe::byte_t> concat_bytes(const std::vector<dataframe::byte_t>& bytes) {
  return std::vector<dataframe::byte_t>(bytes.begin(), bytes.begin() + bytes.size() - 1);
}

NB_MODULE(qutils_bindings, m) {
  nanobind::class_<BitString>(m, "BitString")
    .def(nanobind::init<uint32_t>())
    .def_static("from_bits", [](uint32_t num_bits, uint32_t z) { return BitString::from_bits(num_bits, z); })
    .def_ro("bits", &BitString::bits)
    .def_ro("num_bits", &BitString::num_bits)
    .def("__str__", [](BitString& bs) { return fmt::format("{}", bs); })
    .def("hamming_weight", &BitString::hamming_weight)
    .def("get", &BitString::get)
    .def("set", &BitString::set)
    .def("size", &BitString::size);

  nanobind::class_<PauliString>(m, "PauliString")
    .def(nanobind::init<const std::string&>())
    .def_ro("num_qubits", &PauliString::num_qubits)
    .def("__str__", &PauliString::to_string_ops)
    .def("__mul__", &PauliString::operator*)
    .def("__rmul__", &PauliString::operator*)
    .def("__eq__", &PauliString::operator==)
    .def("__neq__", &PauliString::operator!=)
    .def("to_matrix", [](PauliString& self) { return self.to_matrix(); })
    .def("substring", [](const PauliString& self, const std::vector<uint32_t>& sites) { return self.substring(sites, true); })
    .def("substring_retain", [](const PauliString& self, const std::vector<uint32_t>& sites) { return self.substring(sites, false); })
    .def("x", &PauliString::x)
    .def("y", &PauliString::y)
    .def("z", &PauliString::z)
    .def("s", &PauliString::s)
    .def("sd", &PauliString::sd)
    .def("h", &PauliString::h)
    .def("cx", &PauliString::cx)
    .def("cy", &PauliString::cy)
    .def("cz", &PauliString::cz)
    .def("evolve", [](PauliString& self, const QuantumCircuit& qc) { self.evolve(qc); })
    .def("commutes", &PauliString::commutes)
    .def("set_x", &PauliString::set_x)
    .def("set_x", [](PauliString& self, size_t i, size_t v) { self.set_x(i, static_cast<bool>(v)); })
    .def("set_z", &PauliString::set_z)
    .def("set_z", [](PauliString& self, size_t i, size_t v) { self.set_z(i, static_cast<bool>(v)); })
    .def("get_x", &PauliString::get_x)
    .def("get_z", &PauliString::get_z)
    .def("reduce", [](const PauliString& self, bool z) { 
        QuantumCircuit qc(self.num_qubits);
        std::vector<uint32_t> qubits(self.num_qubits);
        std::iota(qubits.begin(), qubits.end(), 0);
        self.reduce(z, std::make_pair(&qc, qubits));
        return qc;
      }, "z"_a = true);

  m.def("random_paulistring", [](uint32_t num_qubits) {
    return PauliString::randh(num_qubits);
  });

  m.def("bitstring_paulistring", [](uint32_t num_qubits, uint32_t bitstring) {
    return PauliString::from_bitstring(num_qubits, bitstring);
  });

  nanobind::class_<QuantumCircuit>(m, "QuantumCircuit")
    .def(nanobind::init<uint32_t>())
    .def(nanobind::init<QuantumCircuit&>())
    .def("num_qubits", &QuantumCircuit::get_num_qubits)
    .def("__str__", &QuantumCircuit::to_string)
    .def("num_params", &QuantumCircuit::num_params)
    .def("length", &QuantumCircuit::length)
    .def("mzr", [](QuantumCircuit& self, uint32_t q, std::optional<bool> outcome) { 
      self.add_measurement(Measurement::computational_basis(q, outcome)); 
    }, "qubit"_a, "outcome"_a = std::nullopt)
    .def("add_measurement", [](QuantumCircuit& self, const Qubits& qubits, const PauliString& pauli, std::optional<bool> outcome) {
      self.add_measurement(qubits, pauli, outcome);
    }, "qubits"_a, "pauli"_a, "outcome"_a=std::nullopt)
    .def("add_weak_measurement", [](QuantumCircuit& self, const Qubits& qubits, double beta, const PauliString& pauli, std::optional<bool> outcome) {
      self.add_weak_measurement(qubits, beta, pauli, outcome);
    }, "qubits"_a, "beta"_a, "pauli"_a, "outcome"_a=std::nullopt)
    .def("add_gate", [](QuantumCircuit& self, const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) { self.add_gate(gate, qubits); })
    .def("add_gate", [](QuantumCircuit& self, const Eigen::MatrixXcd& gate, uint32_t q) { self.add_gate(gate, q); })
    .def("append", [](QuantumCircuit& self, const QuantumCircuit& other, const std::optional<std::vector<uint32_t>>& qubits) { 
      if (qubits) {
        self.append(other, qubits.value());
      } else {
        self.append(other); 
      }
    }, "circuit"_a, "qubits"_a = nanobind::none())
    .def("h", &QuantumCircuit::h)
    .def("x", &QuantumCircuit::x)
    .def("y", &QuantumCircuit::y)
    .def("z", &QuantumCircuit::z)
    .def("s", &QuantumCircuit::s)
    .def("sd", &QuantumCircuit::sd)
    .def("t", &QuantumCircuit::t)
    .def("td", &QuantumCircuit::td)
    .def("sqrtX", &QuantumCircuit::sqrtX)
    .def("sqrtY", &QuantumCircuit::sqrtY)
    .def("sqrtZ", &QuantumCircuit::sqrtZ)
    .def("sqrtXd", &QuantumCircuit::sqrtXd)
    .def("sqrtYd", &QuantumCircuit::sqrtYd)
    .def("sqrtZd", &QuantumCircuit::sqrtZd)
    .def("cx", &QuantumCircuit::cx)
    .def("cy", &QuantumCircuit::cy)
    .def("cz", &QuantumCircuit::cz)
    .def("swap", &QuantumCircuit::swap)
    .def("random_clifford", [](QuantumCircuit& self, const std::vector<uint32_t>& qubits) {
      self.random_clifford(qubits);
    })
    .def("adjoint", [](QuantumCircuit& self, const std::optional<std::vector<double>> params) { return self.adjoint(params); }, "params"_a = nanobind::none())
    .def("reverse", &QuantumCircuit::reverse)
    .def("conjugate", &QuantumCircuit::conjugate)
    .def("to_matrix", [](QuantumCircuit& self, const std::optional<std::vector<double>> params) { return self.to_matrix(params); }, "params"_a = nanobind::none());

  m.def("random_clifford", [](uint32_t num_qubits) { 
    return random_clifford(num_qubits);
  });
  m.def("single_qubit_random_clifford", [](uint32_t r) {
    QuantumCircuit qc(1);
    single_qubit_clifford_impl(qc, 0, r % 24);
    return qc;
  });
  m.def("generate_haar_circuit", &generate_haar_circuit);
  m.def("hardware_efficient_ansatz", &hardware_efficient_ansatz);
  m.def("haar_unitary", [](uint32_t num_qubits) { return haar_unitary(num_qubits); });

  nanobind::class_<MagicQuantumState>(m, "QuantumState")
    .def("num_qubits", &MagicQuantumState::get_num_qubits)
    .def("__str__", &MagicQuantumState::to_string)
    .def("__getstate__", [](const MagicQuantumState& self) { return convert_bytes(self.serialize()); })
    .def("h", &MagicQuantumState::h)
    .def("x", &MagicQuantumState::x)
    .def("y", &MagicQuantumState::y)
    .def("z", &MagicQuantumState::z)
    .def("s", &MagicQuantumState::s)
    .def("sd", &MagicQuantumState::sd)
    .def("t", &MagicQuantumState::t)
    .def("td", &MagicQuantumState::td)
    .def("sqrtX", &MagicQuantumState::sqrtX)
    .def("sqrtY", &MagicQuantumState::sqrtY)
    .def("sqrtZ", &MagicQuantumState::sqrtZ)
    .def("sqrtXd", &MagicQuantumState::sqrtXd)
    .def("sqrtYd", &MagicQuantumState::sqrtYd)
    .def("sqrtZd", &MagicQuantumState::sqrtZd)
    .def("cx", &MagicQuantumState::cx)
    .def("cy", &MagicQuantumState::cy)
    .def("cz", &MagicQuantumState::cz)
    .def("swap", &MagicQuantumState::swap)
    .def("random_clifford", &MagicQuantumState::random_clifford)
    .def("partial_trace", [](MagicQuantumState& self, const Qubits& qubits) { return std::dynamic_pointer_cast<MagicQuantumState>(self.partial_trace(qubits)); })
    .def("expectation", &MagicQuantumState::expectation)
    .def("probabilities", [](MagicQuantumState& self) { return to_nbarray(self.probabilities()); } )
    .def("purity", &MagicQuantumState::purity)
    .def("mzr", [](MagicQuantumState& self, uint32_t q, std::optional<bool> outcome) {
      return self.measure(Measurement::computational_basis(q, outcome));
    }, "qubit"_a, "outcome"_a=std::nullopt)
    .def("measure", [](MagicQuantumState& self, const Qubits& qubits, const PauliString& pauli, std::optional<bool> outcome) {
      return self.measure(Measurement(qubits, pauli, outcome));
    }, "qubits"_a, "pauli"_a, "outcome"_a=std::nullopt)
    .def("weak_measure", [](MagicQuantumState& self, const Qubits& qubits, double beta, const PauliString& pauli, std::optional<bool> outcome) {
      return self.weak_measure(WeakMeasurement(qubits, beta, pauli, outcome));
    }, "qubits"_a, "beta"_a, "pauli"_a, "outcome"_a=std::nullopt)
    .def("entropy", &MagicQuantumState::entropy, "qubits"_a, "index"_a)
    .def("sample_bitstrings", &MagicQuantumState::sample_bitstrings)
    .def("sample_paulis", &MagicQuantumState::sample_paulis)
    .def("sample_paulis_exact", &MagicQuantumState::sample_paulis_exact)
    .def("sample_paulis_exhaustive", &MagicQuantumState::sample_paulis_exhaustive)
    .def("sample_paulis_montecarlo", [](MagicQuantumState& self, size_t num_samples, size_t equilibration_timesteps, ProbabilityFunc prob, PyMutationFunc py_mutation) {
      auto mutation = convert_from_pyfunc(py_mutation);
      return self.sample_paulis_montecarlo({}, num_samples, equilibration_timesteps, prob, mutation);
    })
    .def("stabilizer_renyi_entropy", &MagicQuantumState::stabilizer_renyi_entropy)
    .def_static("calculate_magic_mutual_information_from_samples", [](MagicQuantumState& self, const MutualMagicAmplitudes& samples2, const MutualMagicAmplitudes& samples4) {
      return self.calculate_magic_mutual_information_from_samples(samples2, samples4);
    })
    .def_static("calculate_magic_mutual_information_from_samples2", &MagicQuantumState::calculate_magic_mutual_information_from_samples2)
    .def("magic_mutual_information_samples_exact", &MagicQuantumState::magic_mutual_information_samples_exact)
    .def("magic_mutual_information_samples", &MagicQuantumState::magic_mutual_information_samples_montecarlo)
    .def("magic_mutual_information_exhaustive", &MagicQuantumState::magic_mutual_information_exhaustive)
    .def("magic_mutual_information", &MagicQuantumState::magic_mutual_information)
    .def("bipartite_magic_mutual_information_samples_exact", &MagicQuantumState::bipartite_magic_mutual_information_samples_exact)
    .def("bipartite_magic_mutual_information_samples_montecarlo", &MagicQuantumState::bipartite_magic_mutual_information_samples_montecarlo)
    .def("bipartite_magic_mutual_information_exhaustive", &MagicQuantumState::bipartite_magic_mutual_information_exhaustive)
    .def("bipartite_magic_mutual_information", &MagicQuantumState::bipartite_magic_mutual_information);

  nanobind::class_<Statevector, MagicQuantumState>(m, "Statevector")
    .def(nanobind::init<uint32_t>())
    .def(nanobind::init<QuantumCircuit>())
    .def(nanobind::init<MatrixProductState>())
    .def("__setstate__", [](Statevector& self, const nanobind::bytes& bytes) { 
      new (&self) Statevector();
      self.deserialize(convert_bytes(bytes)); 
    })
    .def_ro("data", &Statevector::data)
    .def("normalize", &Statevector::normalize)
    .def("inner", &Statevector::inner)
    .def("expectation_matrix", [](Statevector& self, const Eigen::MatrixXcd& m, const std::vector<uint32_t>& qubits) { return self.expectation(m, qubits); })
    .def("evolve", [](Statevector& self, const QuantumCircuit& qc) { self.evolve(qc); })
    .def("evolve", [](Statevector& self, const Eigen::Matrix2cd& gate, uint32_t q) { self.evolve(gate, q); })
    .def("evolve", [](Statevector& self, const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) { self.evolve(gate, qubits); });

  nanobind::class_<DensityMatrix, MagicQuantumState>(m, "DensityMatrix")
    .def(nanobind::init<uint32_t>())
    .def(nanobind::init<QuantumCircuit>())
    .def("__setstate__", [](DensityMatrix& self, const nanobind::bytes& bytes) { 
      new (&self) DensityMatrix();
      self.deserialize(convert_bytes(bytes)); 
    })
    .def_ro("data", &DensityMatrix::data)
    .def("expectation_matrix", [](DensityMatrix& self, const Eigen::MatrixXcd& m, const std::vector<uint32_t>& qubits) { return self.expectation(m, qubits); })
    .def("evolve", [](DensityMatrix& self, const QuantumCircuit& qc) { self.evolve(qc); })
    .def("evolve", [](DensityMatrix& self, const Eigen::Matrix2cd& gate, uint32_t q) { self.evolve(gate, q); })
    .def("evolve", [](DensityMatrix& self, const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) { self.evolve(gate, qubits); });

  nanobind::class_<MatrixProductState, MagicQuantumState>(m, "MatrixProductState")
    .def(nanobind::init<uint32_t, uint32_t, double>(), "num_qubits"_a, "max_bond_dimension"_a, "sv_threshold"_a=1e-4)
    .def("__str__", &MatrixProductState::to_string)
    .def("__setstate__", [](MatrixProductState& self, const nanobind::bytes& bytes) { 
      new (&self) MatrixProductState(1, 2);
      self.deserialize(convert_bytes(bytes)); 
    })
    .def("print_mps", &MatrixProductState::print_mps)
    .def("set_debug_level", &MatrixProductState::set_debug_level)
    .def("bond_dimension_at_site", &MatrixProductState::bond_dimension)
    .def("singular_values", [](MatrixProductState& self, uint32_t q) { return to_nbarray(self.singular_values(q)); })
    .def("tensor", [](MatrixProductState& self, uint32_t q) { return to_nbarray(self.tensor(q)); })
    .def("get_logged_truncerr", [](MatrixProductState& self, uint32_t q) { return to_nbarray(self.get_logged_truncerr()); })
    .def("trace", &MatrixProductState::trace)
    .def("expectation_matrix", [](MatrixProductState& self, const Eigen::MatrixXcd& m, const std::vector<uint32_t>& qubits) { return self.expectation(m, qubits); })
    .def("magic_mutual_information", &MatrixProductState::magic_mutual_information)
    .def("evolve", [](MatrixProductState& self, const QuantumCircuit& qc) { self.evolve(qc); })
    .def("evolve", [](MatrixProductState& self, const Eigen::Matrix2cd& gate, uint32_t q) { self.evolve(gate, q); })
    .def("evolve", [](MatrixProductState& self, const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) { self.evolve(gate, qubits); });

  m.def("ising_ground_state", &MatrixProductState::ising_ground_state, "num_qubits"_a, "h"_a, "bond_dimension"_a=16, "sv_threshold"_a=1e-8, "num_sweeps"_a=10);

  nanobind::class_<EntropyState>(m, "EntropyState");

  nanobind::class_<EntropySampler>(m, "EntropySampler")
    .def(nanobind::init<dataframe::ExperimentParams&>())
    .def_static("create_and_emplace", [](dataframe::ExperimentParams& params) {
      EntropySampler sampler(params);
      return std::make_pair(sampler, params);
    })
    .def("take_samples", [](EntropySampler& sampler, std::shared_ptr<EntropyState> state) {
      dataframe::SampleMap samples;
      sampler.add_samples(samples, state);
      return samples;
    });

  nanobind::class_<InterfaceSampler>(m, "InterfaceSampler")
    .def(nanobind::init<dataframe::ExperimentParams&>())
    .def_static("create_and_emplace", [](dataframe::ExperimentParams& params) {
      InterfaceSampler sampler(params);
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
      QuantumStateSampler sampler(params);
      return std::make_pair(sampler, params);
    })
    .def("take_samples", [](QuantumStateSampler& sampler, const std::shared_ptr<MagicQuantumState>& state) {
      std::shared_ptr<QuantumState> qstate = std::dynamic_pointer_cast<QuantumState>(state);
      dataframe::SampleMap samples;
      sampler.add_samples(samples, qstate);
      return samples;
    });

  nanobind::class_<MagicStateSampler>(m, "MagicStateSampler")
    .def(nanobind::init<dataframe::ExperimentParams&>())
    .def_static("create_and_emplace", [](dataframe::ExperimentParams& params) {
      MagicStateSampler sampler(params);
      return std::make_pair(sampler, params);
    })
    .def("set_sre_montecarlo_update", [](MagicStateSampler& self, PyMutationFunc func) {
      auto mutation = convert_from_pyfunc(func);
      self.set_montecarlo_update(mutation);
    })
    .def("take_samples", [](MagicStateSampler& sampler, const std::shared_ptr<MagicQuantumState>& state) {
      dataframe::SampleMap samples;
      sampler.add_samples(samples, state);
      return samples;
    });

  nanobind::class_<QuantumCHPState, EntropyState>(m, "QuantumCHPState")
    .def(nanobind::init<uint32_t>())
    .def_ro("num_qubits", &QuantumCHPState::num_qubits)
    .def("__str__", &QuantumCHPState::to_string)
    .def("__getstate__", [](const QuantumCHPState& self) { return convert_bytes(self.serialize()); })
    .def("__setstate__", [](QuantumCHPState& self, const nanobind::bytes& bytes) { 
      new (&self) QuantumCHPState();
      self.deserialize(convert_bytes(bytes)); })
    .def("set_x", &QuantumCHPState::set_x)
    .def("set_x", [](QuantumCHPState& self, size_t i, size_t j, size_t v) { self.set_x(i, j, static_cast<bool>(v)); })
    .def("set_z", &QuantumCHPState::set_z)
    .def("set_z", [](QuantumCHPState& self, size_t i, size_t j, size_t v) { self.set_z(i, j, static_cast<bool>(v)); })
    .def("get_x", [](QuantumCHPState& self, size_t i, size_t j) { return self.tableau.get_x(i, j); })
    .def("get_z", [](QuantumCHPState& self, size_t i, size_t j) { return self.tableau.get_z(i, j); })
    .def("tableau", [](QuantumCHPState& self) { return self.tableau.to_matrix(); })
    .def("partial_rank", &QuantumCHPState::partial_rank)
    .def("evolve", [](QuantumCHPState& self, const QuantumCircuit& circuit) { circuit.apply(self); })
    .def("h", [](QuantumCHPState& self, uint32_t q) { self.h(q); })
    .def("s", [](QuantumCHPState& self, uint32_t q) { self.s(q); })
    .def("sd", [](QuantumCHPState& self, uint32_t q) { self.sd(q); })
    .def("x", [](QuantumCHPState& self, uint32_t q) { self.x(q); })
    .def("y", [](QuantumCHPState& self, uint32_t q) { self.y(q); })
    .def("z", [](QuantumCHPState& self, uint32_t q) { self.z(q); })
    .def("sqrtx", [](QuantumCHPState& self, uint32_t q) { self.sqrtx(q); })
    .def("sqrty", [](QuantumCHPState& self, uint32_t q) { self.sqrty(q); })
    .def("sqrtz", [](QuantumCHPState& self, uint32_t q) { self.sqrtz(q); })
    .def("sqrtxd", [](QuantumCHPState& self, uint32_t q) { self.sqrtxd(q); })
    .def("sqrtyd", [](QuantumCHPState& self, uint32_t q) { self.sqrtyd(q); })
    .def("sqrtzd", [](QuantumCHPState& self, uint32_t q) { self.sqrtzd(q); })
    .def("cx", [](QuantumCHPState& self, uint32_t q1, uint32_t q2) { self.cx(q1, q2); })
    .def("cy", [](QuantumCHPState& self, uint32_t q1, uint32_t q2) { self.cy(q1, q2); })
    .def("cz", [](QuantumCHPState& self, uint32_t q1, uint32_t q2) { self.cz(q1, q2); })
    .def("swap", [](QuantumCHPState& self, uint32_t q1, uint32_t q2) { self.swap(q1, q2); })
    .def("mxr", [](QuantumCHPState& self, uint32_t q) { return self.mxr(q); })
    .def("myr", [](QuantumCHPState& self, uint32_t q) { return self.myr(q); })
    .def("mzr", [](QuantumCHPState& self, uint32_t q) { return self.mzr(q); })
    .def("mxr_expectation", [](QuantumCHPState& self, uint32_t q) { return self.mxr_expectation(q); })
    .def("myr_expectation", [](QuantumCHPState& self, uint32_t q) { return self.myr_expectation(q); })
    .def("mzr_expectation", [](QuantumCHPState& self, uint32_t q) { return self.mzr_expectation(q); })
    .def("to_statevector", &QuantumCHPState::to_statevector)
    .def("entropy", &QuantumCHPState::entropy, "qubits"_a, "index"_a=2)
    .def("entropy_surface", [](QuantumCHPState& self) {
      return self.get_entropy_surface<int>(2u);
    })
    .def("random_clifford", &QuantumCHPState::random_clifford)
    .def("get_texture", [](QuantumCHPState& state, 
        const std::vector<float>& color_x, const std::vector<float>& color_z, const std::vector<float>& color_y
    ) -> std::tuple<std::vector<float>, size_t, size_t> {
      if ((color_x.size() != 3) || (color_z.size() != 3) || (color_y.size() != 3)) {
        throw std::runtime_error(fmt::format("Color must have size 3. Colors have sizes {}, {}, and {}.", color_x.size(), color_y.size(), color_z.size()));
      }

      Texture texture = state.get_texture({color_x[0], color_x[1], color_x[2], 1.0},
                                          {color_z[0], color_z[1], color_z[1], 1.0},
                                          {color_y[0], color_y[1], color_y[1], 1.0});

      const float* data = texture.data();
      return {std::vector<float>{data, data + texture.len()}, texture.n, texture.m};
    });

  nanobind::class_<CliffordTable>(m, "CliffordTable")
    .def("__init__", [](CliffordTable *t, const std::vector<PauliString>& p1, const std::vector<PauliString>& p2) {
        if (p1.size() != p2.size()) {
          throw std::runtime_error("Mismatched length of PauliStrings in filter function for CliffordTable.");
        }

        auto filter = [&p1, &p2](const QuantumCircuit& qc) -> bool {
          for (size_t i = 0; i < p1.size(); i++) {
            PauliString q1 = p1[i];
            PauliString q2 = p2[i];
            qc.apply(q1);
            if (q1 != q2) {
              return false;
            }
          }

          return true;
        };
        new (t) CliffordTable(filter); 
    })
    .def("circuits", &CliffordTable::get_circuits)
    .def("num_elements", &CliffordTable::num_elements)
    .def("random_circuit", [](CliffordTable& self) {
      QuantumCircuit qc(2);
      self.apply_random({0, 1}, qc);
      return qc;
    });

  nanobind::class_<FreeFermionState, EntropyState>(m, "FreeFermionState")
    .def(nanobind::init<size_t>())
    .def("system_size", &FreeFermionState::system_size)
    .def("__str__", &FreeFermionState::to_string)
    .def("particles_at", &FreeFermionState::particles_at)
    .def("swap", &FreeFermionState::swap)
    .def("entropy", &FreeFermionState::entropy)
    .def("prepare_hamiltonian", [](FreeFermionState& self, const Eigen::MatrixXcd& H) { return self.prepare_hamiltonian(H); })
    .def("prepare_hamiltonian_", [](FreeFermionState& self, const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B) { return self.prepare_hamiltonian(A, B); })
    .def("evolve_hamiltonian", [](FreeFermionState& self, const Eigen::MatrixXcd& H, double t) { self.evolve_hamiltonian(H, t); }, "H"_a, "t"_a = 1.0)
    .def("evolve_hamiltonian_", [](FreeFermionState& self, const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B, double t) { self.evolve_hamiltonian(A, B, t); }, "A"_a, "B"_a, "t"_a = 1.0)
    .def("evolve", [](FreeFermionState& self, const Eigen::MatrixXcd& U) { self.evolve(U); }, "U"_a)
    .def("weak_measure_hamiltonian", [](FreeFermionState& self, const Eigen::MatrixXcd& H, double beta) { self.weak_measurement_hamiltonian(H, beta); }, "H"_a, "beta"_a = 1.0)
    .def("weak_measure_hamiltonian", [](FreeFermionState& self, const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B, double beta) { self.weak_measurement_hamiltonian(A, B, beta); }, "A"_a, "B"_a, "beta"_a = 1.0)
    .def("weak_measure", [](FreeFermionState& self, const Eigen::MatrixXcd& U) { self.weak_measurement(U); }, "H"_a)
    .def("mzr", [](FreeFermionState& self, size_t i) {
      return self.projective_measurement(i, randf());
    })
    .def("num_particles", &FreeFermionState::num_particles)
    .def("correlation_matrix", &FreeFermionState::correlation_matrix)
    .def("occupation", [](FreeFermionState& self, size_t i) { return self.occupation(i); });


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
