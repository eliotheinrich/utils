#define FMT_HEADER_ONLY

#include "QuantumState.h"
#include "CliffordState.h"
#include "BinaryPolynomial.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/pair.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>

using namespace nanobind::literals;

NB_MODULE(pyqtools_bindings, m) {
  std::complex<double> i(0, 1);

  Eigen::Matrix2cd H; H << 1, 1, 1, -1; H /= std::sqrt(2);
  Eigen::Matrix2cd X; X << 0, 1, 1, 0;
  Eigen::Matrix2cd Y; Y << 0, -i, i, 0;
  Eigen::Matrix2cd Z; Z << 1, 0, 0, -1;
  Eigen::Matrix2cd sqrtX; sqrtX << 1, -i, -i, 1; sqrtX /= std::sqrt(2);
  Eigen::Matrix2cd sqrtY; sqrtY << 1, -1, 1, 1; sqrtY /= std::sqrt(2);
  Eigen::Matrix2cd sqrtZ; sqrtZ << 1, 0, 0, i;
  Eigen::Matrix2cd sqrtXd; sqrtXd << 1, i, i, 1; sqrtXd /= std::sqrt(2);
  Eigen::Matrix2cd sqrtYd; sqrtYd << 1, 1, -1, 1; sqrtYd /= std::sqrt(2);
  Eigen::Matrix2cd sqrtZd; sqrtZd << 1, 0, 0, -i;
  Eigen::Matrix2cd T; T << 1, 0, 0, std::complex<double>(1, 1)/std::sqrt(2);
  Eigen::Matrix2cd Td; Td << 1, 0, 0, std::complex<double>(1, -1)/std::sqrt(2);
  Eigen::Matrix4cd CX; CX << 1, 0, 0, 0,
                             0, 1, 0, 0,
                             0, 0, 0, 1,
                             0, 0, 1, 0;
  Eigen::Matrix4cd CY; CY << 1, 0, 0, 0,
                             0, 1, 0, 0,
                             0, 0, 0, -i,
                             0, 0, i, 0;
  Eigen::Matrix4cd CZ; CZ << 1, 0, 0, 0,
                             0, 1, 0, 0,
                             0, 0, 1, 0,
                             0, 0, 0, -1;
  Eigen::Matrix4cd SWAP; SWAP << 1, 0, 0, 0,
                                 0, 0, 1, 0,
                                 0, 1, 0, 0,
                                 0, 0, 0, 1;

  nanobind::class_<PauliString>(m, "PauliString")
    .def(nanobind::init<const std::string&>())
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
    .def("s", [](PauliString& self, uint32_t q) { self.s(q); })
    .def("sd", [](PauliString& self, uint32_t q) { self.sd(q); })
    .def("h", [](PauliString& self, uint32_t q) { self.h(q); })
    .def("cx", [](PauliString& self, uint32_t q1, uint32_t q2) { self.cx(q1, q2); })
    .def("cy", [](PauliString& self, uint32_t q1, uint32_t q2) { self.cy(q1, q2); })
    .def("cz", [](PauliString& self, uint32_t q1, uint32_t q2) { self.cz(q1, q2); })
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
    .def("__str__", &QuantumCircuit::to_string)
    .def("num_params", &QuantumCircuit::num_params)
    .def("length", &QuantumCircuit::length)
    .def("add_measurement", [](QuantumCircuit& self, uint32_t q) { self.add_measurement(q); })
    .def("add_measurement", [](QuantumCircuit& self, const std::vector<uint32_t>& qubits) { self.add_measurement(qubits); })
    .def("add_gate", [](QuantumCircuit& self, const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) { self.add_gate(gate, qubits); })
    .def("add_gate", [](QuantumCircuit& self, const Eigen::MatrixXcd& gate, uint32_t q) { self.add_gate(gate, q); })
    .def("append", [](QuantumCircuit& self, const QuantumCircuit& other) { self.append(other); })
    .def("h_gate", [H](QuantumCircuit& self, uint32_t q) { self.add_gate(H, q); })
    .def("x_gate", [X](QuantumCircuit& self, uint32_t q) { self.add_gate(X, q); })
    .def("y_gate", [Y](QuantumCircuit& self, uint32_t q) { self.add_gate(Y, q); })
    .def("z_gate", [Z](QuantumCircuit& self, uint32_t q) { self.add_gate(Z, q); })
    .def("s_gate", [sqrtZ](QuantumCircuit& self, uint32_t q) { self.add_gate(sqrtZ, q); })
    .def("sd_gate", [sqrtZd](QuantumCircuit& self, uint32_t q) { self.add_gate(sqrtZd, q); })
    .def("t_gate", [T](QuantumCircuit& self, uint32_t q) { self.add_gate(T, q); })
    .def("td_gate", [Td](QuantumCircuit& self, uint32_t q) { self.add_gate(Td, q); })
    .def("sqrtx_gate", [sqrtX](QuantumCircuit& self, uint32_t q) { self.add_gate(sqrtX, q); })
    .def("sqrty_gate", [sqrtY](QuantumCircuit& self, uint32_t q) { self.add_gate(sqrtY, q); })
    .def("sqrtz_gate", [sqrtZ](QuantumCircuit& self, uint32_t q) { self.add_gate(sqrtZ, q); })
    .def("sqrtxd_gate", [sqrtXd](QuantumCircuit& self, uint32_t q) { self.add_gate(sqrtXd, q); })
    .def("sqrtyd_gate", [sqrtYd](QuantumCircuit& self, uint32_t q) { self.add_gate(sqrtYd, q); })
    .def("sqrtzd_gate", [sqrtZd](QuantumCircuit& self, uint32_t q) { self.add_gate(sqrtZd, q); })
    .def("cx_gate", [CX](QuantumCircuit& self, uint32_t q1, uint32_t q2) { self.add_gate(CX, {q1, q2}); })
    .def("cy_gate", [CY](QuantumCircuit& self, uint32_t q1, uint32_t q2) { self.add_gate(CY, {q1, q2}); })
    .def("cz_gate", [CZ](QuantumCircuit& self, uint32_t q1, uint32_t q2) { self.add_gate(CZ, {q1, q2}); })
    .def("swap_gate", [SWAP](QuantumCircuit& self, uint32_t q1, uint32_t q2) { self.add_gate(SWAP, {q1, q2}); })
    .def("to_matrix", &QuantumCircuit::to_matrix);

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
    .def("h", [H](Statevector& self, uint32_t q) { self.evolve(H, q); })
    .def("x", [X](Statevector& self, uint32_t q) { self.evolve(X, q); })
    .def("y", [Y](Statevector& self, uint32_t q) { self.evolve(Y, q); })
    .def("z", [Z](Statevector& self, uint32_t q) { self.evolve(Z, q); })
    .def("s", [sqrtZ](Statevector& self, uint32_t q) { self.evolve(sqrtZ, q); })
    .def("sd", [sqrtZd](Statevector& self, uint32_t q) { self.evolve(sqrtZd, q); })
    .def("t", [T](Statevector& self, uint32_t q) { self.evolve(T, q); })
    .def("td", [Td](Statevector& self, uint32_t q) { self.evolve(Td, q); })
    .def("sqrtx", [sqrtX](Statevector& self, uint32_t q) { self.evolve(sqrtX, q); })
    .def("sqrty", [sqrtY](Statevector& self, uint32_t q) { self.evolve(sqrtY, q); })
    .def("sqrtz", [sqrtZ](Statevector& self, uint32_t q) { self.evolve(sqrtZ, q); })
    .def("sqrtxd", [sqrtXd](Statevector& self, uint32_t q) { self.evolve(sqrtXd, q); })
    .def("sqrtyd", [sqrtYd](Statevector& self, uint32_t q) { self.evolve(sqrtYd, q); })
    .def("sqrtzd", [sqrtZd](Statevector& self, uint32_t q) { self.evolve(sqrtZd, q); })
    .def("cx", [CX](Statevector& self, uint32_t q1, uint32_t q2) { self.evolve(CX, {q1, q2}); })
    .def("cy", [CY](Statevector& self, uint32_t q1, uint32_t q2) { self.evolve(CY, {q1, q2}); })
    .def("cz", [CZ](Statevector& self, uint32_t q1, uint32_t q2) { self.evolve(CZ, {q1, q2}); })
    .def("swap", [SWAP](Statevector& self, uint32_t q1, uint32_t q2) { self.evolve(SWAP, {q1, q2}); })
    .def("mzr", [](Statevector& self, uint32_t q) { return self.mzr(q); })
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
    .def("h", [H](DensityMatrix& self, uint32_t q) { self.evolve(H, q); })
    .def("x", [X](DensityMatrix& self, uint32_t q) { self.evolve(X, q); })
    .def("y", [Y](DensityMatrix& self, uint32_t q) { self.evolve(Y, q); })
    .def("z", [Z](DensityMatrix& self, uint32_t q) { self.evolve(Z, q); })
    .def("s", [sqrtZ](DensityMatrix& self, uint32_t q) { self.evolve(sqrtZ, q); })
    .def("sd", [sqrtZd](DensityMatrix& self, uint32_t q) { self.evolve(sqrtZd, q); })
    .def("t", [T](DensityMatrix& self, uint32_t q) { self.evolve(T, q); })
    .def("td", [Td](DensityMatrix& self, uint32_t q) { self.evolve(Td, q); })
    .def("sqrtx", [sqrtX](DensityMatrix& self, uint32_t q) { self.evolve(sqrtX, q); })
    .def("sqrty", [sqrtY](DensityMatrix& self, uint32_t q) { self.evolve(sqrtY, q); })
    .def("sqrtz", [sqrtZ](DensityMatrix& self, uint32_t q) { self.evolve(sqrtZ, q); })
    .def("sqrtxd", [sqrtXd](DensityMatrix& self, uint32_t q) { self.evolve(sqrtXd, q); })
    .def("sqrtyd", [sqrtYd](DensityMatrix& self, uint32_t q) { self.evolve(sqrtYd, q); })
    .def("sqrtzd", [sqrtZd](DensityMatrix& self, uint32_t q) { self.evolve(sqrtZd, q); })
    .def("cx", [CX](DensityMatrix& self, uint32_t q1, uint32_t q2) { self.evolve(CX, {q1, q2}); })
    .def("cy", [CY](DensityMatrix& self, uint32_t q1, uint32_t q2) { self.evolve(CY, {q1, q2}); })
    .def("cz", [CZ](DensityMatrix& self, uint32_t q1, uint32_t q2) { self.evolve(CZ, {q1, q2}); })
    .def("swap", [SWAP](DensityMatrix& self, uint32_t q1, uint32_t q2) { self.evolve(SWAP, {q1, q2}); })
    .def("mzr", &DensityMatrix::measure)
    .def("probabilities", &DensityMatrix::probabilities)
    .def("expectation", [](DensityMatrix& self, const PauliString& p) { return self.expectation(p); })
    .def("expectation", [](DensityMatrix& self, const Eigen::MatrixXcd& m, const std::vector<uint32_t>& qubits) { return self.expectation(m, qubits); })
    .def("sample_paulis_exact", &DensityMatrix::sample_paulis_exact)
    .def("sample_paulis_exhaustive", &DensityMatrix::sample_paulis_exhaustive)
    .def("sample_paulis_montecarlo", &DensityMatrix::sample_paulis_montecarlo)
    .def("magic_mutual_information_exact", &DensityMatrix::magic_mutual_information_exact)
    .def("magic_mutual_information_montecarlo", &DensityMatrix::magic_mutual_information_montecarlo)
    .def("magic_mutual_information_exhaustive", &DensityMatrix::magic_mutual_information_exhaustive)
    .def("evolve", [](DensityMatrix& self, const QuantumCircuit& qc) { self.evolve(qc); })
    .def("evolve", [](DensityMatrix& self, const Eigen::Matrix2cd& gate, uint32_t q) { self.evolve(gate, q); })
    .def("evolve", [](DensityMatrix& self, const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) { self.evolve(gate, qubits); });

  nanobind::class_<MatrixProductState>(m, "MatrixProductState")
    .def(nanobind::init<uint32_t, uint32_t, double>(), "num_qubits"_a, "bond_dimension"_a, "sv_threshold"_a=1e-4)
    .def_ro("num_qubits", &MatrixProductState::num_qubits)
    .def("__str__", &MatrixProductState::to_string)
    .def("print_mps", &MatrixProductState::print_mps)
    .def("entropy", &MatrixProductState::entropy, "qubits"_a, "index"_a)
    .def("h", [H](MatrixProductState& self, uint32_t q) { self.evolve(H, q); })
    .def("x", [X](MatrixProductState& self, uint32_t q) { self.evolve(X, q); })
    .def("y", [Y](MatrixProductState& self, uint32_t q) { self.evolve(Y, q); })
    .def("z", [Z](MatrixProductState& self, uint32_t q) { self.evolve(Z, q); })
    .def("s", [sqrtZ](MatrixProductState& self, uint32_t q) { self.evolve(sqrtZ, q); })
    .def("sd", [sqrtZd](MatrixProductState& self, uint32_t q) { self.evolve(sqrtZd, q); })
    .def("t", [T](MatrixProductState& self, uint32_t q) { self.evolve(T, q); })
    .def("td", [Td](MatrixProductState& self, uint32_t q) { self.evolve(Td, q); })
    .def("sqrtx", [sqrtX](MatrixProductState& self, uint32_t q) { self.evolve(sqrtX, q); })
    .def("sqrty", [sqrtY](MatrixProductState& self, uint32_t q) { self.evolve(sqrtY, q); })
    .def("sqrtz", [sqrtZ](MatrixProductState& self, uint32_t q) { self.evolve(sqrtZ, q); })
    .def("sqrtxd", [sqrtXd](MatrixProductState& self, uint32_t q) { self.evolve(sqrtXd, q); })
    .def("sqrtyd", [sqrtYd](MatrixProductState& self, uint32_t q) { self.evolve(sqrtYd, q); })
    .def("sqrtzd", [sqrtZd](MatrixProductState& self, uint32_t q) { self.evolve(sqrtZd, q); })
    .def("cx", [CX](MatrixProductState& self, uint32_t q1, uint32_t q2) { self.evolve(CX, {q1, q2}); })
    .def("cy", [CY](MatrixProductState& self, uint32_t q1, uint32_t q2) { self.evolve(CY, {q1, q2}); })
    .def("cz", [CZ](MatrixProductState& self, uint32_t q1, uint32_t q2) { self.evolve(CZ, {q1, q2}); })
    .def("swap", [SWAP](MatrixProductState& self, uint32_t q1, uint32_t q2) { self.evolve(SWAP, {q1, q2}); })
    .def("mzr", [](MatrixProductState& self, uint32_t q) { return self.mzr(q); })
    .def("measure", [](MatrixProductState& self, const PauliString& p, const std::vector<uint32_t>& qubits) { return self.measure(p, qubits); })
    .def("weak_measure", [](MatrixProductState& self, const PauliString& p, const std::vector<uint32_t>& qubits, double beta) { return self.weak_measure(p, qubits, beta); })
    .def("probabilities", &MatrixProductState::probabilities)
    .def("mzr_prob", &MatrixProductState::mzr_prob)
    .def("expectation", [](MatrixProductState& self, const PauliString& p) { return self.expectation(p); })
    .def("expectation", [](MatrixProductState& self, const Eigen::MatrixXcd& m, const std::vector<uint32_t>& qubits) { return self.expectation(m, qubits); })
    .def("partial_trace", &MatrixProductState::partial_trace)
    .def("sample_paulis_exhaustive", &MatrixProductState::sample_paulis_exhaustive)
    .def("sample_paulis_montecarlo", &MatrixProductState::sample_paulis_montecarlo)
    .def("magic_mutual_information", &MatrixProductState::magic_mutual_information)
    .def("magic_mutual_information_exact", &MatrixProductState::magic_mutual_information_exact)
    .def("magic_mutual_information_montecarlo", &MatrixProductState::magic_mutual_information_montecarlo)
    .def("magic_mutual_information_exhaustive", &MatrixProductState::magic_mutual_information_exhaustive)
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
    .def("magic_mutual_information_exact", &MatrixProductOperator::magic_mutual_information_exact)
    .def("magic_mutual_information_exact", &MatrixProductOperator::magic_mutual_information_exact)
    .def("magic_mutual_information_montecarlo", &MatrixProductOperator::magic_mutual_information_montecarlo)
    .def("magic_mutual_information_exhaustive", &MatrixProductOperator::magic_mutual_information_exhaustive)
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


  auto symm = [](const QuantumCircuit& q) {
    PauliString p("+ZZ");
    PauliString p_ = p;
    q.apply(p_);
    return p == p_;
  };

  static CliffordTable z2_table = CliffordTable(symm);

  m.def("random_clifford_z2", [](uint32_t num_qubits, uint32_t q1, uint32_t q2) {
    QuantumCircuit qc(num_qubits);
    thread_local std::random_device gen;
    std::minstd_rand rng(gen());
    z2_table.apply_random(rng, {q1, q2}, qc);
    return qc;
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
}
