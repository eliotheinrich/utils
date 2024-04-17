#include "QuantumState.h"
#include "CliffordState.h"
#include "BinaryPolynomial.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/map.h>
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
    .def_ro("num_qubits", &Statevector::num_qubits)
    .def("__str__", &Statevector::to_string)
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
    .def("measure", [](Statevector& self, uint32_t q) { return self.measure(q); })
    .def("measure", [](Statevector& self, uint32_t q, bool outcome) { return self.measure(q, outcome); })
    .def("probabilities", [](Statevector& self) { return self.probabilities(); })
    .def("inner", &Statevector::inner)
    .def("evolve", [](Statevector& self, const QuantumCircuit& qc) { self.evolve(qc); })
    .def("evolve", [](Statevector& self, const Eigen::Matrix2cd& gate, uint32_t q) { self.evolve(gate, q); })
    .def("evolve", [](Statevector& self, const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) { self.evolve(gate, qubits); });

  nanobind::class_<DensityMatrix>(m, "DensityMatrix")
    .def(nanobind::init<uint32_t>())
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
    .def("measure", &DensityMatrix::measure)
    .def("probabilities", &DensityMatrix::probabilities)
    .def("evolve", [](DensityMatrix& self, const QuantumCircuit& qc) { self.evolve(qc); })
    .def("evolve", [](DensityMatrix& self, const Eigen::Matrix2cd& gate, uint32_t q) { self.evolve(gate, q); })
    .def("evolve", [](DensityMatrix& self, const Eigen::MatrixXcd& gate, const std::vector<uint32_t>& qubits) { self.evolve(gate, qubits); });

  nanobind::class_<QuantumCHPState>(m, "QuantumCHPState")
    .def(nanobind::init<uint32_t>())
    .def("system_size", &QuantumCHPState::system_size)
    .def("__str__", &QuantumCHPState::to_string)
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
    .def("entropy", &QuantumCHPState::entropy, "qubits"_a, "index"_a=2.0)
    .def("random_clifford", &QuantumCHPState::random_clifford);

  nanobind::class_<BinaryMatrix>(m, "BinaryMatrix")
    .def(nanobind::init<size_t, size_t>())
    .def_ro("num_cols", &BinaryMatrix::num_cols)
    .def_ro("num_rows", &BinaryMatrix::num_rows)
    .def("set", &BinaryMatrix::set)
    .def("set", [](BinaryMatrix& self, size_t i, size_t j, size_t v) { self.set(i, j, static_cast<bool>(v)); })
    .def("get", &BinaryMatrix::get)
    .def("__str__", [](BinaryMatrix& self) { return self.to_string(); })
    .def("rref", [](BinaryMatrix& self) { self.rref(); })
    .def("rank", [](BinaryMatrix& self, bool inplace) { return self.rank(inplace); }, "inplace"_a=false);

  nanobind::class_<ParityCheckMatrix, BinaryMatrix>(m, "ParityCheckMatrix")
    .def(nanobind::init<size_t, size_t>())
    .def("reduce", &ParityCheckMatrix::reduce)
    .def("to_generator_matrix", &ParityCheckMatrix::to_generator_matrix, "inplace"_a = false);

  nanobind::class_<GeneratorMatrix, BinaryMatrix>(m, "GeneratorMatrix")
    .def(nanobind::init<size_t, size_t>())
    .def("to_parity_check_matrix", &GeneratorMatrix::to_parity_check_matrix, "inplace"_a = false)
    .def("generator_locality", &GeneratorMatrix::generator_locality);
}
