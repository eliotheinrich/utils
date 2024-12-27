#include <PyQutils.hpp>

using namespace nanobind::literals;

NB_MODULE(qutils_bindings, m) {
  std::complex<double> i(0, 1);

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

  nanobind::class_<Statevector>(m, "Statevector")
    .def(nanobind::init<uint32_t>())
    .def(nanobind::init<QuantumCircuit>())
    .def(nanobind::init<MatrixProductState>())
    .def_ro("num_qubits", &Statevector::num_qubits)
    .def("__str__", &Statevector::to_string)
    .def("normalize", &Statevector::normalize)
    .def("entropy", &Statevector::entropy, "qubits"_a, "index"_a)
    .def("h", &Statevector::h)
    .def("x", &Statevector::x)
    .def("y", &Statevector::y)
    .def("z", &Statevector::z)
    .def("s", &Statevector::s)
    .def("sd", &Statevector::sd)
    .def("t", &Statevector::t)
    .def("td", &Statevector::td)
    .def("sqrtX", &Statevector::sqrtX)
    .def("sqrtY", &Statevector::sqrtY)
    .def("sqrtZ", &Statevector::sqrtZ)
    .def("sqrtXd", &Statevector::sqrtXd)
    .def("sqrtYd", &Statevector::sqrtYd)
    .def("sqrtZd", &Statevector::sqrtZd)
    .def("cx", &Statevector::cx)
    .def("cy", &Statevector::cy)
    .def("cz", &Statevector::cz)
    .def("swap", &Statevector::swap)
    .def("random_clifford", &Statevector::random_clifford)
    .def("mzr", [](Statevector& self, uint32_t q) { self.mzr(q); })
    .def("mzr", [](Statevector& self, uint32_t q, bool outcome) { return self.mzr(q, outcome); })
    .def("measure", [](Statevector& self, const PauliString& p, const std::vector<uint32_t>& qubits) { return self.measure(p, qubits); })
    .def("weak_measure", [](Statevector& self, const PauliString& p, const std::vector<uint32_t>& qubits, double beta) { return self.weak_measure(p, qubits, beta); })
    .def("probabilities", [](Statevector& self) { return self.probabilities(); })
    .def("inner", &Statevector::inner)
    .def("expectation", [](Statevector& self, const PauliString& p) { return self.expectation(p); })
    .def("expectation", [](Statevector& self, const Eigen::MatrixXcd& m, const std::vector<uint32_t>& qubits) { return self.expectation(m, qubits); })
    .def("evolve", [](Statevector& self, const QuantumCircuit& qc) { self.evolve(qc); })
    .def("evolve", [](Statevector& self, const Eigen::Matrix2cd& gate, uint32_t q) { self.evolve(gate, q); })
    .def("evolve", [](Statevector& self, const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) { self.evolve(gate, qubits); });

  nanobind::class_<DensityMatrix>(m, "DensityMatrix")
    .def(nanobind::init<uint32_t>())
    .def(nanobind::init<QuantumCircuit>())
    .def_ro("num_qubits", &DensityMatrix::num_qubits)
    .def("__str__", &DensityMatrix::to_string)
    .def("entropy", &DensityMatrix::entropy, "qubits"_a, "index"_a)
    .def("h", &DensityMatrix::h)
    .def("x", &DensityMatrix::x)
    .def("y", &DensityMatrix::y)
    .def("z", &DensityMatrix::z)
    .def("s", &DensityMatrix::s)
    .def("sd", &DensityMatrix::sd)
    .def("t", &DensityMatrix::t)
    .def("td", &DensityMatrix::td)
    .def("sqrtX", &DensityMatrix::sqrtX)
    .def("sqrtY", &DensityMatrix::sqrtY)
    .def("sqrtZ", &DensityMatrix::sqrtZ)
    .def("sqrtXd", &DensityMatrix::sqrtXd)
    .def("sqrtYd", &DensityMatrix::sqrtYd)
    .def("sqrtZd", &DensityMatrix::sqrtZd)
    .def("cx", &DensityMatrix::cx)
    .def("cy", &DensityMatrix::cy)
    .def("cz", &DensityMatrix::cz)
    .def("swap", &DensityMatrix::swap)
    .def("random_clifford", &DensityMatrix::random_clifford)
    .def("mzr", &DensityMatrix::mzr)
    .def("probabilities", &DensityMatrix::probabilities)
    .def("expectation", [](DensityMatrix& self, const PauliString& p) { return self.expectation(p); })
    .def("expectation", [](DensityMatrix& self, const Eigen::MatrixXcd& m, const std::vector<uint32_t>& qubits) { return self.expectation(m, qubits); })
    .def("sample_paulis_exact", &DensityMatrix::sample_paulis_exact)
    .def("sample_paulis_exhaustive", &DensityMatrix::sample_paulis_exhaustive)
    .def("sample_paulis_montecarlo", &DensityMatrix::sample_paulis_montecarlo)
    .def("evolve", [](DensityMatrix& self, const QuantumCircuit& qc) { self.evolve(qc); })
    .def("evolve", [](DensityMatrix& self, const Eigen::Matrix2cd& gate, uint32_t q) { self.evolve(gate, q); })
    .def("evolve", [](DensityMatrix& self, const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) { self.evolve(gate, qubits); });

  nanobind::class_<MatrixProductState>(m, "MatrixProductState")
    .def(nanobind::init<uint32_t, uint32_t, double>(), "num_qubits"_a, "bond_dimension"_a, "sv_threshold"_a=1e-4)
    .def_ro("num_qubits", &MatrixProductState::num_qubits)
    .def("__str__", &MatrixProductState::to_string)
    .def("print_mps", &MatrixProductState::print_mps)
    .def("entropy", &MatrixProductState::entropy, "qubits"_a, "index"_a)
    .def("h", &MatrixProductState::h)
    .def("x", &MatrixProductState::x)
    .def("y", &MatrixProductState::y)
    .def("z", &MatrixProductState::z)
    .def("s", &MatrixProductState::s)
    .def("sd", &MatrixProductState::sd)
    .def("t", &MatrixProductState::t)
    .def("td", &MatrixProductState::td)
    .def("sqrtX", &MatrixProductState::sqrtX)
    .def("sqrtY", &MatrixProductState::sqrtY)
    .def("sqrtZ", &MatrixProductState::sqrtZ)
    .def("sqrtXd", &MatrixProductState::sqrtXd)
    .def("sqrtYd", &MatrixProductState::sqrtYd)
    .def("sqrtZd", &MatrixProductState::sqrtZd)
    .def("cx", &MatrixProductState::cx)
    .def("cy", &MatrixProductState::cy)
    .def("cz", &MatrixProductState::cz)
    .def("swap", &MatrixProductState::swap)
    .def("random_clifford", &MatrixProductState::random_clifford)
    .def("mzr", &MatrixProductState::mzr)
    .def("measure", [](MatrixProductState& self, const PauliString& p, const std::vector<uint32_t>& qubits) { return self.measure(p, qubits); })
    .def("weak_measure", [](MatrixProductState& self, const PauliString& p, const std::vector<uint32_t>& qubits, double beta) { return self.weak_measure(p, qubits, beta); })
    .def("probabilities", &MatrixProductState::probabilities)
    .def("expectation", [](MatrixProductState& self, const PauliString& p) { return self.expectation(p); })
    .def("expectation", [](MatrixProductState& self, const Eigen::MatrixXcd& m, const std::vector<uint32_t>& qubits) { return self.expectation(m, qubits); })
    .def("partial_trace", &MatrixProductState::partial_trace)
    .def("sample_paulis_exhaustive", &MatrixProductState::sample_paulis_exhaustive)
    .def("sample_paulis_montecarlo", &MatrixProductState::sample_paulis_montecarlo)
    .def("evolve", [](MatrixProductState& self, const QuantumCircuit& qc) { self.evolve(qc); })
    .def("evolve", [](MatrixProductState& self, const Eigen::Matrix2cd& gate, uint32_t q) { self.evolve(gate, q); })
    .def("evolve", [](MatrixProductState& self, const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) { self.evolve(gate, qubits); });

  nanobind::class_<MatrixProductOperator>(m, "MatrixProductOperator")
    .def(nanobind::init<const MatrixProductState&, const std::vector<uint32_t>&>())
    .def_ro("num_qubits", &MatrixProductOperator::num_qubits)
    .def("partial_trace", &MatrixProductOperator::partial_trace)
    .def("expectation", &MatrixProductOperator::expectation)
    .def("sample_paulis_exhaustive", &MatrixProductOperator::sample_paulis_exhaustive)
    .def("sample_paulis_montecarlo", &MatrixProductOperator::sample_paulis_montecarlo)
    .def("__str__", &MatrixProductOperator::to_string);

  m.def("ising_ground_state", &MatrixProductState::ising_ground_state, "num_qubits"_a, "h"_a, "bond_dimension"_a=16, "sv_threshold"_a=1e-8, "num_sweeps"_a=10);

  nanobind::class_<QuantumCHPState>(m, "QuantumCHPState")
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
    .def("random_clifford", &QuantumCHPState::random_clifford);

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

  nanobind::class_<FreeFermionState>(m, "FreeFermionState")
    .def("__init__", [](FreeFermionState *s, size_t system_size) {
      new (s) FreeFermionState(system_size, true);
    })
    .def("system_size", &FreeFermionState::system_size)
    .def("__str__", &FreeFermionState::to_string)
    .def("particles_at", &FreeFermionState::particles_at)
    .def("swap", &FreeFermionState::swap)
    .def("entropy", &FreeFermionState::entropy)
    .def("evolve", [](FreeFermionState& self, const Eigen::MatrixXcd& U, double t) { self.evolve(U, t); }, "U"_a, "t"_a = 1.0)
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
#endif
}
