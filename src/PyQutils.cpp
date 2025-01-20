#include <PyQutils.hpp>


using PyMutationFunc = std::function<PauliString(PauliString)>;
inline PauliMutationFunc convert_from_pyfunc(PyMutationFunc func) {
  return [func](PauliString& p, std::minstd_rand&) { p = func(p); };
}

NB_MODULE(qutils_bindings, m) {
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
    thread_local std::random_device gen;
    std::minstd_rand rng(gen());
    return PauliString::rand(num_qubits, rng);
  });

  m.def("bitstring_paulistring", [](uint32_t num_qubits, uint32_t bitstring) {
    return PauliString::from_bitstring(num_qubits, bitstring);
  });

  nanobind::class_<QuantumCircuit>(m, "QuantumCircuit")
    .def(nanobind::init<uint32_t>())
    .def(nanobind::init<QuantumCircuit&>())
    .def_ro("num_qubits", &QuantumCircuit::num_qubits)
    .def("__str__", &QuantumCircuit::to_string)
    .def("num_params", &QuantumCircuit::num_params)
    .def("length", &QuantumCircuit::length)
    .def("add_measurement", [](QuantumCircuit& self, uint32_t q) { self.add_measurement(q); })
    .def("add_measurement", [](QuantumCircuit& self, const std::vector<uint32_t>& qubits) { self.add_measurement(qubits); })
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
      thread_local std::random_device gen;
      std::minstd_rand rng(gen());
      self.random_clifford(qubits, rng);
    })
    .def("adjoint", [](QuantumCircuit& self, const std::optional<std::vector<double>> params) { return self.adjoint(params); }, "params"_a = nanobind::none())
    .def("reverse", &QuantumCircuit::reverse)
    .def("conjugate", &QuantumCircuit::conjugate)
    .def("to_matrix", [](QuantumCircuit& self, const std::optional<std::vector<double>> params) { return self.to_matrix(params); }, "params"_a = nanobind::none());

  m.def("random_clifford", [](uint32_t num_qubits) { 
    thread_local std::random_device gen;
    std::minstd_rand rng(gen());
    return random_clifford(num_qubits, rng);
  });
  m.def("single_qubit_random_clifford", [](uint32_t r) {
    QuantumCircuit qc(1);
    single_qubit_clifford_impl(qc, 0, r % 24);
    return qc;
  });
  m.def("generate_haar_circuit", &generate_haar_circuit);
  m.def("hardware_efficient_ansatz", &hardware_efficient_ansatz);
  m.def("haar_unitary", [](uint32_t num_qubits) { return haar_unitary(num_qubits); });

  nanobind::class_<QuantumState>(m, "QuantumState")
    .def("__str__", &QuantumState::to_string)
    .def("h", &QuantumState::h)
    .def("x", &QuantumState::x)
    .def("y", &QuantumState::y)
    .def("z", &QuantumState::z)
    .def("s", &QuantumState::s)
    .def("sd", &QuantumState::sd)
    .def("t", &QuantumState::t)
    .def("td", &QuantumState::td)
    .def("sqrtX", &QuantumState::sqrtX)
    .def("sqrtY", &QuantumState::sqrtY)
    .def("sqrtZ", &QuantumState::sqrtZ)
    .def("sqrtXd", &QuantumState::sqrtXd)
    .def("sqrtYd", &QuantumState::sqrtYd)
    .def("sqrtZd", &QuantumState::sqrtZd)
    .def("cx", &QuantumState::cx)
    .def("cy", &QuantumState::cy)
    .def("cz", &QuantumState::cz)
    .def("swap", &QuantumState::swap)
    .def("random_clifford", &QuantumState::random_clifford)
    .def("partial_trace", &QuantumState::partial_trace)
    .def("expectation", &QuantumState::expectation)
    .def("probabilities", &QuantumState::probabilities)
    .def("purity", &QuantumState::purity)
    .def("mzr", &QuantumState::mzr)
    .def("entropy", &QuantumState::entropy, "qubits"_a, "index"_a)
    .def("sample_paulis", &QuantumState::sample_paulis)
    .def("sample_paulis_exact", &QuantumState::sample_paulis_exact)
    .def("sample_paulis_exhaustive", &QuantumState::sample_paulis_exhaustive)
    .def("sample_paulis_montecarlo", [](QuantumState& self, size_t num_samples, size_t equilibration_timesteps, ProbabilityFunc prob, PyMutationFunc py_mutation) {
      auto mutation = convert_from_pyfunc(py_mutation);
      return self.sample_paulis_montecarlo(num_samples, equilibration_timesteps, prob, mutation);
    })
    .def("stabilizer_renyi_entropy", [](QuantumState& self, size_t index, const std::vector<PauliAmplitude>& samples) { return self.stabilizer_renyi_entropy(index, samples); })
    .def("stabilizer_renyi_entropy", [](QuantumState& self, size_t index, const std::vector<double>& samples) { return self.stabilizer_renyi_entropy(index, samples); })
    .def_static("calculate_magic_mutual_information_from_samples", &QuantumState::calculate_magic_mutual_information_from_samples)
    .def_static("calculate_magic_mutual_information_from_chi_samples", &QuantumState::calculate_magic_mutual_information_from_chi_samples)
    .def("magic_mutual_information_samples_exact", &QuantumState::magic_mutual_information_samples_exact)
    .def("magic_mutual_information_samples", &QuantumState::magic_mutual_information_samples_montecarlo)
    .def("magic_mutual_information_exhaustive", &QuantumState::magic_mutual_information_exhaustive)
    .def("magic_mutual_information", &QuantumState::magic_mutual_information)
    .def("bipartite_magic_mutual_information_samples_exact", &QuantumState::bipartite_magic_mutual_information_samples_exact)
    .def("bipartite_magic_mutual_information_samples_montecarlo", &QuantumState::bipartite_magic_mutual_information_samples_montecarlo)
    .def("bipartite_magic_mutual_information_exhaustive", &QuantumState::bipartite_magic_mutual_information_exhaustive)
    .def("bipartite_magic_mutual_information", &QuantumState::bipartite_magic_mutual_information);

  nanobind::class_<Statevector, QuantumState>(m, "Statevector")
    .def(nanobind::init<uint32_t>())
    .def(nanobind::init<QuantumCircuit>())
    .def(nanobind::init<MatrixProductState>())
    .def("normalize", &Statevector::normalize)
    .def("mzr_forced", [](Statevector& self, uint32_t q, bool outcome) { return self.mzr(q, outcome); })
    .def("measure", [](Statevector& self, const PauliString& p, const std::vector<uint32_t>& qubits) { return self.measure(p, qubits); })
    .def("weak_measure", [](Statevector& self, const PauliString& p, const std::vector<uint32_t>& qubits, double beta) { return self.weak_measure(p, qubits, beta); })
    .def("inner", &Statevector::inner)
    .def("expectation", [](Statevector& self, const Eigen::MatrixXcd& m, const std::vector<uint32_t>& qubits) { return self.expectation(m, qubits); })
    .def("evolve", [](Statevector& self, const QuantumCircuit& qc) { self.evolve(qc); })
    .def("evolve", [](Statevector& self, const Eigen::Matrix2cd& gate, uint32_t q) { self.evolve(gate, q); })
    .def("evolve", [](Statevector& self, const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) { self.evolve(gate, qubits); });

  nanobind::class_<DensityMatrix, QuantumState>(m, "DensityMatrix")
    .def(nanobind::init<uint32_t>())
    .def(nanobind::init<QuantumCircuit>())
    .def("expectation", [](DensityMatrix& self, const Eigen::MatrixXcd& m, const std::vector<uint32_t>& qubits) { return self.expectation(m, qubits); })
    .def("evolve", [](DensityMatrix& self, const QuantumCircuit& qc) { self.evolve(qc); })
    .def("evolve", [](DensityMatrix& self, const Eigen::Matrix2cd& gate, uint32_t q) { self.evolve(gate, q); })
    .def("evolve", [](DensityMatrix& self, const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) { self.evolve(gate, qubits); });

  nanobind::class_<MatrixProductState, QuantumState>(m, "MatrixProductState")
    .def(nanobind::init<uint32_t, uint32_t, double>(), "num_qubits"_a, "bond_dimension"_a, "sv_threshold"_a=1e-4)
    .def("__str__", &MatrixProductState::to_string)
    .def("print_mps", &MatrixProductState::print_mps)
    .def("measure", [](MatrixProductState& self, const PauliString& p, const std::vector<uint32_t>& qubits) { return self.measure(p, qubits); })
    .def("weak_measure", [](MatrixProductState& self, const PauliString& p, const std::vector<uint32_t>& qubits, double beta) { return self.weak_measure(p, qubits, beta); })
    //.def("expectation", [](MatrixProductState& self, const Eigen::MatrixXcd& m, const std::vector<uint32_t>& qubits) { return self.expectation(m, qubits); })
    .def("magic_mutual_information", &MatrixProductState::magic_mutual_information)
    .def("evolve", [](MatrixProductState& self, const QuantumCircuit& qc) { self.evolve(qc); })
    .def("evolve", [](MatrixProductState& self, const Eigen::Matrix2cd& gate, uint32_t q) { self.evolve(gate, q); })
    .def("evolve", [](MatrixProductState& self, const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) { self.evolve(gate, qubits); });

  nanobind::class_<MatrixProductOperator, QuantumState>(m, "MatrixProductOperator")
    .def(nanobind::init<const MatrixProductState&, const std::vector<uint32_t>&>());

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
    .def("set_sre_montecarlo_update", [](QuantumStateSampler& self, PyMutationFunc func) {
      auto mutation = convert_from_pyfunc(func);
      self.set_montecarlo_update(mutation);
    })
    .def("take_samples", [](QuantumStateSampler& sampler, const std::shared_ptr<QuantumState>& state) {
      dataframe::SampleMap samples;
      sampler.add_samples(samples, state);
      return samples;
    });

  nanobind::class_<QuantumCHPState, EntropyState>(m, "QuantumCHPState")
    .def(nanobind::init<uint32_t>())
    .def_ro("num_qubits", &QuantumCHPState::num_qubits)
    .def("__str__", &QuantumCHPState::to_string)
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
      thread_local std::random_device gen;
      std::minstd_rand rng(gen());
      QuantumCircuit qc(2);
      self.apply_random(rng, {0, 1}, qc);
      return qc;
    });

  nanobind::class_<FreeFermionState, EntropyState>(m, "FreeFermionState")
    .def("__init__", [](FreeFermionState *s, size_t system_size, bool particles_conserved) {
      new (s) FreeFermionState(system_size, particles_conserved);
    }, "system_size"_a, "particles_conserved"_a = true)
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
      thread_local std::random_device gen;
      std::minstd_rand rng(gen());
      double r = rng()/RAND_MAX;
      return self.projective_measurement(i, rng()/RAND_MAX);
    })
    .def("num_particles", &FreeFermionState::num_particles)
    .def("correlation_matrix", &FreeFermionState::correlation_matrix)
    .def("correlation_samples", [](FreeFermionState& self) {
      auto C = self.correlation_matrix();
      std::vector<std::vector<double>> correlations(self.system_size(), std::vector<double>(self.system_size()));

      for (size_t r = 0; r < self.system_size(); r++) {
        // Average over space
        for (size_t i = 0; i < self.system_size(); i++) {
          double c = std::abs(C(i, (i + r) % self.system_size()));
          correlations[r][i] = c*c;
        }
      }

      return correlations;
    })
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
