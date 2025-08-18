#include "tests.hpp"

#include "QuantumCircuit.h"

Qubits random_qubits(size_t num_qubits, size_t k) {
  std::minstd_rand rng(randi());
  Qubits qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  std::shuffle(qubits.begin(), qubits.end(), rng);

  Qubits r(qubits.begin(), qubits.begin() + k);
  return r;
}

bool test_circuit_dag() {
  constexpr size_t nqb = 8;
  QuantumCircuit qc(nqb);
  qc.add_gate(haar_unitary(2), {0, 1});
  qc.add_gate(haar_unitary(2), {2, 3});
  qc.add_gate(haar_unitary(2), {1, 2});
  qc.mzr(0);

  CircuitDAG dag = qc.to_dag();
  //std::cout << dag.to_string() << "\n";

  std::vector<std::set<size_t>> expected = {
    {2, 3},
    {2},
    {},
    {},
  };

  for (size_t i = 0; i < qc.length(); i++) {
    for (size_t j = 0; j < qc.length(); j++) {
      if (expected[i].contains(j)) {
        ASSERT(dag.contains_edge(i, j));
      } else {
        ASSERT(!dag.contains_edge(i, j));
      }
    }
  }

  return true;
}

QuantumCircuit random_unitary_circuit(size_t nqb, size_t depth, const std::vector<size_t>& gate_sizes) {
  QuantumCircuit qc(nqb);
  for (size_t i = 0; i < depth; i++) {
    size_t r = gate_sizes[randi(0, gate_sizes.size())];
    size_t q = randi(0, nqb - r + 1);
    Qubits qubits(r);
    std::iota(qubits.begin(), qubits.end(), q);
    qc.add_gate(haar_unitary(r), qubits);
  }

  return qc;
}

bool test_qc_canonical() {
  constexpr size_t nqb = 6;

  QuantumCircuit qc = random_unitary_circuit(nqb, 10, {2});
  QuantumCircuit canon = qc.to_canonical_form();

  ASSERT(qc.to_matrix().isApprox(canon.to_matrix()));

  return true;
}

bool test_pauli_reduce() {
  for (size_t i = 0; i < 100; i++) {
    size_t nqb =  randi() % 20 + 1;
    PauliString p1 = PauliString::randh(nqb);
    PauliString p2 = PauliString::randh(nqb);
    while (p2.commutes(p1)) {
      p2 = PauliString::randh(nqb);
    }

    std::vector<uint32_t> qubits(nqb);
    std::iota(qubits.begin(), qubits.end(), 0);
    QuantumCircuit qc(nqb);
    reduce_paulis(p1, p2, qubits, qc);

    PauliString p1_ = p1;
    PauliString p2_ = p2;
    qc.apply(p1_, p2_);

    ASSERT(p1_ == PauliString::basis(nqb, "X", 0, false) && p2_ == PauliString::basis(nqb, "Z", 0, false),
        fmt::format("p1 = {} and p2 = {}\nreduced to {} and {}.", p1, p2, p1_, p2_));
  }

  return true;
}

bool test_qc_simplify() {
  constexpr size_t nqb = 6;
  QuantumCircuit qc = random_unitary_circuit(nqb, 10, {1, 2});
  QuantumCircuit simple = qc.simplify();

  ASSERT(qc.to_matrix().isApprox(simple.to_matrix()));

  return true;
}

bool test_qc_reduce() {
  constexpr size_t nqb = 8;
  QuantumCircuit qc(nqb);
  auto qubits = random_qubits(nqb, 4);
  PauliString p = PauliString::randh(4);
  qc.h(0);
  qc.h(5);
  qc.add_measurement(qubits, p);
  qc.cx(0, 2);
  qc.swap(6, 2);
  qc.x(1);
  Qubits support = qc.get_support();
  std::set<uint32_t> exps(qubits.begin(), qubits.end());
  exps.insert(0);
  exps.insert(1);
  exps.insert(2);
  exps.insert(5);
  exps.insert(6);
  Qubits expected(exps.begin(), exps.end());
  std::sort(expected.begin(), expected.end());
  ASSERT(support == expected);

  auto [qc_, support_] = qc.reduce();
  QuantumCircuit qc_r(nqb);
  qc_r.append(qc_, support_);

  std::string s1 = qc.to_string();
  std::string s2 = qc_r.to_string();
  ASSERT(s1 == s2);
  // TODO check to_matrix equality?

  auto components = qc.split_into_unitary_components();
  for (const auto &q : components) {
    bool is_unitary = instruction_is_unitary(q.instructions[0]);
    for (size_t k = 1; k < q.length(); k++) {
      ASSERT(instruction_is_unitary(q.instructions[k]) == is_unitary);
    }
  }

  return true;
}

bool test_pauli() {
  Pauli id = Pauli::I;
  Pauli x = Pauli::X;
  Pauli y = Pauli::Y;
  Pauli z = Pauli::Z;

  auto validate_result = [](Pauli p1, Pauli p2, char g, uint8_t p) {
    auto [result, phase] = multiply_pauli(p1, p2);
    return (g == pauli_to_char(result)) && (p == phase);
  };

  ASSERT(pauli_to_char(id) == 'I');
  ASSERT(pauli_to_char(x) == 'X');
  ASSERT(pauli_to_char(y) == 'Y');
  ASSERT(pauli_to_char(z) == 'Z');

  ASSERT(validate_result(id, id, 'I', 0));
  ASSERT(validate_result(id, x, 'X', 0));
  ASSERT(validate_result(x, id, 'X', 0));
  ASSERT(validate_result(id, y, 'Y', 0));
  ASSERT(validate_result(y, id, 'Y', 0));
  ASSERT(validate_result(id, z, 'Z', 0));
  ASSERT(validate_result(z, id, 'Z', 0));
  ASSERT(validate_result(x, x, 'I', 0));
  ASSERT(validate_result(x, y, 'Z', 1));
  ASSERT(validate_result(y, x, 'Z', 3));
  ASSERT(validate_result(x, z, 'Y', 3));
  ASSERT(validate_result(z, x, 'Y', 1));
  ASSERT(validate_result(y, z, 'X', 1));
  ASSERT(validate_result(z, y, 'X', 3));

  return true;
}

int main(int argc, char *argv[]) {
  std::map<std::string, TestResult> tests;
  std::set<std::string> test_names;

  bool run_all = (argc == 1);

  if (!run_all) {
    for (size_t i = 1; i < argc; i++) {
      test_names.insert(argv[i]);
    }
  }
  Random::seed_rng(314);

  ADD_TEST(test_circuit_dag);
  ADD_TEST(test_qc_reduce);
  ADD_TEST(test_pauli_reduce);
  ADD_TEST(test_qc_canonical);
  ADD_TEST(test_qc_simplify);
  ADD_TEST(test_pauli);

  constexpr char green[] = "\033[1;32m";
  constexpr char black[] = "\033[0m";
  constexpr char red[] = "\033[1;31m";

  auto test_passed_str = [&](bool passed) {
    std::stringstream stream;
    if (passed) {
      stream << green << "PASSED" << black;
    } else {
      stream << red << "FAILED" << black;
    }
    
    return stream.str();
  };

  if (tests.size() == 0) {
    std::cout << "No tests to run.\n";
  } else {
    double total_duration = 0.0;
    for (const auto& [name, result] : tests) {
      auto [passed, duration] = result;
      std::cout << fmt::format("{:>40}: {} ({:.2f} seconds)\n", name, test_passed_str(passed), duration/1e6);
      total_duration += duration;
    }

    std::cout << fmt::format("Total duration: {:.2f} seconds\n", total_duration/1e6);
  }
}
