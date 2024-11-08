#include <random>
#define FMT_HEADER_ONLY
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "QuantumState.h"
#include "CliffordState.h"
#include "BinaryPolynomial.h"
#include "Graph.hpp"
#include <iostream>
#include <cassert>

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
bool is_close(T first, V second) {
  return std::abs(first - second) < 1e-8;
}

template <typename T, typename V, typename... Args>
bool is_close(T first, V second, Args... args) {
  if (!is_close(first, second)) {
    return false;
  } else {
    return is_close(first, args...);
  }
}

template <typename T, typename V>
bool states_close(const T& first, const V& second) {
  DensityMatrix d1(first);
  DensityMatrix d2(second);

  if (d1.num_qubits != d2.num_qubits) {
    return false;
  }

  return (d1.data - d2.data).cwiseAbs().maxCoeff() < 1e-2;
}

template <typename T, typename V, typename... Args>
bool states_close(const T& first, const V& second, const Args&... args) {
  if (!states_close(first, second)) {
    return false;
  } else {
    return states_close(first, args...);
  }
}

template <typename T, typename V>
bool states_close_pauli_fuzz(std::minstd_rand& rng, const T& first, const V& second) {
  ASSERT(first.num_qubits == second.num_qubits);

  for (size_t i = 0; i < 100; i++) {
    PauliString p = PauliString::rand(first.num_qubits, rng);
    ASSERT(is_close(first.expectation(p), second.expectation(p)));
  }

  return true;
}

template <typename T, typename V, typename... Args>
bool states_close_pauli_fuzz(std::minstd_rand& rng, const T& first, const V& second, const Args&... args) {
  if (!states_close_pauli_fuzz(first, second)) {
    return false;
  } else {
    return states_close(first, args...);
  }
}

template <typename T, typename... QuantumStates>
size_t get_num_qubits(const T& first, const QuantumStates&... args) {
  size_t num_qubits = first.num_qubits;
  if constexpr (sizeof...(args) == 0) {
    return num_qubits;
  } else {
    if (num_qubits != get_num_qubits(args...)) {
      throw std::runtime_error("Error; inappropriate states passed to get_num_qubits. Number of qubits do not match.");
    }

    return num_qubits;
  }
}

template <typename... QuantumStates>
void randomize_state_haar(std::minstd_rand& rng, QuantumStates&... states) {
  size_t num_qubits = get_num_qubits(states...);

  QuantumCircuit qc(num_qubits);
  size_t depth = num_qubits;

  qc.append(generate_haar_circuit(num_qubits, depth, false, rng()));
  qc.apply(states...);
}

template <typename... QuantumStates>
void randomize_state_clifford(std::minstd_rand& rng, QuantumStates&... states) {
  size_t num_qubits = get_num_qubits(states...);

  QuantumCircuit qc(num_qubits);
  size_t depth = num_qubits;

  for (size_t k = 0; k < depth; k++) {
    for (size_t i = 0; i < num_qubits/2 - 1; i++) {
      uint32_t q1 = (k % 2) ? 2*i : 2*i + 1;
      uint32_t q2 = q1 + 1;

      QuantumCircuit rc = random_clifford(2, rng);
      qc.append(random_clifford(2, rng), {q1, q2});
    }
  }

  qc.apply(states...);
}

std::minstd_rand seeded_rng() {
  thread_local std::random_device gen;
  int seed = gen();
  std::minstd_rand rng(seed);

  return rng;
}


bool test_solve_linear_system() {
  BinaryMatrix M(4, 4);
  M.set(0,0,1);
  M.set(0,1,1);
  M.set(0,2,1);
  M.set(0,3,1);

  M.set(1,0,1);
  M.set(1,1,1);

  M.set(2,0,1);
  M.set(2,2,1);

  M.set(3,0,1);
  std::vector<bool> v{0, 0, 0, 1};
  auto x = M.solve_linear_system(v);
  std::vector<bool> correct{1, 1, 1, 1};

  ASSERT(x != correct);

  return true;
}

bool test_binary_polynomial() {
  BinaryPolynomial poly(5);
  poly.add_term(1, 2);
  poly.add_term(1);
  auto inds = std::vector<size_t>{};
  poly.add_term(inds);

  return true;
}


void random_binary_matrix(std::shared_ptr<BinaryMatrixBase> A, int s) {
  thread_local std::minstd_rand rng;
  rng.seed(s);
  for (size_t i = 0; i < A->num_rows; i++) {
    for (size_t j = 0; j < A->num_cols; j++) {
      if (rng() % 2) {
        A->set(i, j, 1);
      }
    }
  }
}

bool test_binary_matrix() {
  size_t v = 3;
  std::shared_ptr<SparseBinaryMatrix> M1 = std::make_shared<SparseBinaryMatrix>(v, v);
  std::shared_ptr<BinaryMatrix> M2 = std::make_shared<BinaryMatrix>(v, v);

  for (int i = 0; i < 100; i++) {
    random_binary_matrix(M1, i);
    random_binary_matrix(M2, i);

    int r1 = M1->rank();
    int r2 = M2->rank();

    if (r1 != r2) {
      std::cout << "M1 = \n" << M1->to_string() << "\n\n";
      std::cout << "M2 = \n" << M2->to_string() << "\n\n";

      std::cout << "r1 = " << r1 << ", r2 = " << r2 << "\n\n";

      return false;
    }
  }

  return true;
}

bool test_generator_matrix() {
  std::vector<BinaryMatrix> test_cases;
  test_cases.push_back(BinaryMatrix(3, 5));
  //test_cases.push_back(std::make_shared<SparseBinaryMatrix>(3, 5));

  for (auto A : test_cases) {
    A.set(0, 0, 1);
    A.set(0, 1, 1);
    A.set(0, 2, 1);

    A.set(1, 1, 1);
    A.set(1, 3, 1);

    A.set(2, 0, 1);
    A.set(2, 4, 1);

  

    auto G = ParityCheckMatrix(A).to_generator_matrix();
    auto H = G.to_parity_check_matrix();

    if (!(G.congruent(H) && H.congruent(G))) {
      std::cout << "A = \n" << A.to_string() << std::endl;
      std::cout << "G = \n" << G.to_string() << std::endl;
      std::cout << "H = \n" << H.to_string() << std::endl;
      return false;
    }
  }

  return true;
}

bool test_random_regular_graph() {
  for (size_t n = 3; n < 10; n++) {
    for (size_t k = 1; k < n/2; k++) {
      if (n * k % 2) {
        // Algorithm not defined in this case
        continue;
      }

      Graph<int, int> g = Graph<int, int>::random_regular_graph(n, k);
      for (size_t j = 0; j < g.num_vertices; j++) {
        ASSERT(g.degree(j) != k);
        ASSERT(g.contains_edge(j, j));
      }
    }
  }

  return true;
}

bool test_parity_check_reduction() {
  size_t num_runs = 100;

  auto rng = seeded_rng();
  for (size_t i = 0; i < num_runs; i++) {
    size_t num_cols = rng() % 20;
    size_t num_rows = num_cols + 5;
    ParityCheckMatrix P(0, num_cols);

    for (size_t j = 0; j < num_rows; j++) {
      Bitstring b = Bitstring::random(num_cols, rng);
      P.append_row(b);
    }

    uint32_t rank = P.rank();
    P.reduce();

    ASSERT(rank != P.num_rows, fmt::format("{}\n", P.to_string()));
  }

  return true;
}

bool test_statevector() {
  size_t num_qubits = 3;

  Statevector s(num_qubits);
  s.x(0);

  for (size_t i = 0; i < num_qubits; i++) {
    PauliString Z(num_qubits);
    Z.set_z(i, 1);

    std::complex<double> d = s.expectation(Z);

    if (i == 0) {
      ASSERT(is_close(d, -1.0));
    } else {
      ASSERT(is_close(d, 1.0));
    }
  }

  return true;
}

bool test_mps() {
  size_t num_qubits = 6;
  size_t bond_dimension = 1u << num_qubits;

  auto rng = seeded_rng();

  Statevector s(num_qubits);
  MatrixProductState mps(num_qubits, bond_dimension);
  randomize_state_haar(rng, mps, s);

  for (size_t i = 0; i < 100; i++) {
    size_t r = rng() % 2;
    double d1, d2;
    size_t nqb;
    if (r == 0) {
      nqb = rng() % num_qubits + 1;
    } else if (r == 1) {
      nqb = rng() % 2 + 1;
    }

    size_t q = rng() % (num_qubits - nqb + 1);
    std::vector<uint32_t> qubits(nqb);
    std::iota(qubits.begin(), qubits.end(), q);

    if (r == 0) {
      PauliString P = PauliString::rand(nqb, rng);
      PauliString Pp = P.superstring(qubits, num_qubits);
      d1 = std::abs(s.expectation(Pp));
      d2 = std::abs(mps.expectation(Pp));

      ASSERT(is_close(d1, d2), fmt::format("<{}> = {}, {}\n", P.to_string_ops(), d1, d2));
    } else if (r == 1) {
      Eigen::MatrixXcd M = haar_unitary(nqb, rng);

      d1 = std::abs(s.expectation(M, qubits));
      d2 = std::abs(mps.expectation(M, qubits));

      auto mat_to_str = [](const Eigen::MatrixXcd& mat) {
        std::stringstream ss;
        ss << mat;
        return ss.str();
      };
      ASSERT(is_close(d1, d2), fmt::format("<{}> = {}, {}\n", mat_to_str(M), d1, d2));
    }
  }

  return mps.debug_tests() && states_close(s, mps);
}

bool test_clifford_states_unitary() {
  size_t nqb = 2;
  QuantumCHPState chp(nqb);
  QuantumGraphState graph(nqb);
  Statevector sv(nqb);

  auto rng = seeded_rng();

  for (size_t i = 0; i < 10; i++) {
    QuantumCircuit qc(nqb);
    qc.append(random_clifford(nqb, rng));

    // TODO include measurements

    for (size_t j = 0; j < 3; j++) {
      size_t q = rng() % nqb;
      //qc.mzr(q);
    }

    qc.apply(sv, chp, graph);

    Statevector sv_chp = chp.to_statevector();
    Statevector sv_graph = graph.to_statevector();

    ASSERT(states_close(sv, sv_chp), fmt::format("Clifford simulators disagree."));
  }

  return true;
}

bool test_pauli_reduce() {
  auto rng = seeded_rng();

  for (size_t i = 0; i < 100; i++) {
    size_t nqb =  rng() % 20 + 1;
    PauliString p1 = PauliString::rand(nqb, rng);
    PauliString p2 = PauliString::rand(nqb, rng);
    while (p2.commutes(p1)) {
      p2 = PauliString::rand(nqb, rng);
    }

    std::vector<uint32_t> qubits(nqb);
    std::iota(qubits.begin(), qubits.end(), 0);
    QuantumCircuit qc(nqb);
    reduce_paulis(p1, p2, qubits, qc);

    PauliString p1_ = p1;
    PauliString p2_ = p2;
    qc.apply(p1_, p2_);

    ASSERT(p1_ == PauliString::basis(nqb, "X", 0, false) && p2_ == PauliString::basis(nqb, "Z", 0, false),
        fmt::format("p1 = {} and p2 = {}\nreduced to {} and {}.", p1.to_string_ops(), p2.to_string_ops(), p1_.to_string_ops(), p2_.to_string_ops()));
  }

  return true;
}

bool test_nonlocal_mps() {
  size_t nqb = 6;

  auto rng = seeded_rng();

  for (size_t i = 0; i < 4; i++) {
    QuantumCircuit qc = generate_haar_circuit(nqb, nqb, true, rng());

    std::vector<uint32_t> qubit_map(nqb);
    std::iota(qubit_map.begin(), qubit_map.end(), 0);
    std::shuffle(qubit_map.begin(), qubit_map.end(), rng);
    qc.apply_qubit_map(qubit_map);

    MatrixProductState mps(nqb, 20);
    Statevector sv(nqb);

    qc.apply(sv, mps);

    ASSERT(states_close(sv, mps), fmt::format("States not close after nonlocal circuit: \n{}\n{}", sv.to_string(), mps.to_string()));
  }

  return true;
}

bool test_partial_trace() {
  constexpr size_t nqb = 6;
  static_assert(nqb % 2 == 0);

  auto rng = seeded_rng();

  for (size_t i = 0; i < 10; i++) {
    MatrixProductState mps(nqb, 1u << nqb);
    DensityMatrix rho(nqb);
    randomize_state_haar(rng, mps, rho);

    size_t qA = rng() % (nqb / 2);
    std::vector<uint32_t> qubitsA(nqb);
    std::iota(qubitsA.begin(), qubitsA.end(), 0);
    std::shuffle(qubitsA.begin(), qubitsA.end(), rng);
    qubitsA = std::vector<uint32_t>(qubitsA.begin(), qubitsA.begin() + qA);

    size_t qB = rng() % (nqb / 2);
    std::vector<uint32_t> qubitsB(nqb - qA);
    std::iota(qubitsB.begin(), qubitsB.end(), 0);
    std::shuffle(qubitsB.begin(), qubitsB.end(), rng);
    qubitsB = std::vector<uint32_t>(qubitsB.begin(), qubitsB.begin() + qB);

    auto complement = [](const std::vector<uint32_t>& qubits, size_t num_qubits) {
      std::vector<bool> mask(num_qubits, true);
      for (const auto q : qubits) {
        mask[q] = false;
      } 

      std::vector<uint32_t> qubits_;
      for (size_t i = 0; i < num_qubits; i++) {
        if (mask[i]) {
          qubits_.push_back(i);
        }
      }

      return qubits_;
    };

    auto rho0 = rho.partial_trace({});
    auto rhoA = rho.partial_trace(qubitsA);
    auto rhoB = rhoA.partial_trace(qubitsB);

    auto mpo0 = mps.partial_trace({});
    auto mpoA = mps.partial_trace(qubitsA);
    auto mpoB = mpoA.partial_trace(qubitsB);

    PauliString P = PauliString::rand(nqb, rng);

    std::vector<uint32_t> qubitsA_ = complement(qubitsA, mps.num_qubits);
    std::vector<uint32_t> qubitsB_ = complement(qubitsB, mpoA.num_qubits);

    PauliString PA = P.substring(qubitsA_, false);

    auto a0_rho = std::abs(rho.expectation(P));
    auto aA_rho = std::abs(rho.expectation(PA));
    
    auto a0_mps = std::abs(mps.expectation(P));
    auto aA_mps = std::abs(mps.expectation(PA));

    PA = P.substring(qubitsA_, true);
    PauliString PB = PA.substring(qubitsB_, true);

    auto b0_rho = std::abs(rho0.expectation(P));
    auto bA_rho = std::abs(rhoA.expectation(PA));
    auto bB_rho = std::abs(rhoB.expectation(PB));
    
    auto b0_mps = std::abs(mpo0.expectation(P));
    auto bA_mps = std::abs(mpoA.expectation(PA));
    auto bB_mps = std::abs(mpoB.expectation(PB));

    ASSERT(states_close(mpo0, rho0), fmt::format("States not equal after tracing []."));
    ASSERT(states_close(mpoA, rhoA), fmt::format("States not equal after tracing {}.", qubitsA));
    ASSERT(states_close(mpoB, rhoB), fmt::format("States not equal after tracing {} and {}.", qubitsA, qubitsB));

    ASSERT(is_close(a0_rho, a0_mps, b0_rho, b0_mps), fmt::format("Expectations not equal after tracing []. {}, {}, {}, {}", a0_rho, a0_mps, b0_rho, b0_mps));
    ASSERT(is_close(aA_rho, aA_mps, bA_rho, bA_mps), fmt::format("Expectations not equal after tracing {}. {}, {}, {}, {}", qubitsA, aA_rho, aA_mps, bA_rho, bA_mps));
    // Check that second round of trace is good
    ASSERT(is_close(bB_rho, bB_mps), fmt::format("Expectations not equal after tracing {} and {}. {}, {}", qubitsA, qubitsB, bB_rho, bB_mps));
  }

  return true;
}

bool test_mps_measure() {
  size_t nqb = 6;

  auto rng = seeded_rng();

  MatrixProductState mps(nqb, 1u << nqb);
  Statevector sv(nqb);
  int seed = rng();
  mps.seed(seed, seed);
  sv.seed(seed);

  for (size_t i = 0; i < 100; i++) {
    randomize_state_haar(rng, mps, sv);

    ASSERT(states_close(sv, mps), fmt::format("States do not agree before measurement.\n"));

    for (size_t j = 0; j < 5; j++) {
      PauliString P;
      std::vector<uint32_t> qubits;

      int r = rng() % 2;
      if (r == 0) {
        P = PauliString::rand(2, rng);
        uint32_t q = rng() % (nqb - 1);
        qubits = {q, q + 1};
      } else {
        P = PauliString::rand(1, rng);
        uint32_t q = rng() % nqb;
        qubits = {q};
      }

      bool b1 = mps.measure(P, qubits);
      bool b2 = sv.measure(P, qubits);

      ASSERT(mps.debug_tests(), fmt::format("MPS failed debug tests for P = {} on {}. seed = {}", P.to_string_ops(), qubits, seed));
      ASSERT(b1 == b2, fmt::format("Different measurement outcomes observed for {}.", P.to_string_ops()));
      ASSERT(states_close(sv, mps), fmt::format("States don't match after measurement of {} on {}.\n{}\n{}", P.to_string_ops(), qubits, sv.to_string(), mps.to_string()));
    }
  }

  return true;
}

bool test_weak_measure() {
  constexpr size_t nqb = 6;

  auto rng = seeded_rng();

  MatrixProductState mps(nqb, 1u << nqb);
  Statevector sv(nqb);
  int seed = rng();
  mps.seed(seed, seed);
  sv.seed(seed);

  for (size_t i = 0; i < 100; i++) {
    randomize_state_haar(rng, mps, sv);

    ASSERT(states_close(sv, mps), fmt::format("States do not agree before measurement.\n"));

    for (size_t j = 0; j < 5; j++) {
      PauliString P;
      std::vector<uint32_t> qubits;
      if (rng() % 2) {
        P = PauliString::rand(2, rng);
        uint32_t q = rng() % (nqb - 1);
        qubits = {q, q + 1};
      } else {
        P = PauliString::rand(1, rng);
        uint32_t q = rng() % nqb;
        qubits = {q};
      }


      constexpr double beta = 1.0;
      bool b1 = mps.weak_measure(P, qubits, beta);
      bool b2 = sv.weak_measure(P, qubits, beta);

      double d = std::abs(sv.inner(Statevector(mps)));

      ASSERT(mps.debug_tests(), fmt::format("MPS failed debug tests for P = {} on {}. seed = {}", P.to_string_ops(), qubits, seed));
      ASSERT(b1 == b2, "Different measurement outcomes observed.");
      ASSERT(states_close(sv, mps), fmt::format("States don't match after weak measurement of {} on {}. d = {} \n{}\n{}", P.to_string_ops(), qubits, d, sv.to_string(), mps.to_string()));
    }
  }

  return true;
}

bool test_z2_clifford() {
  auto rng = seeded_rng();

  constexpr size_t nqb = 8;

  auto Z2 = [](const QuantumCircuit& qc) {
    PauliString XX("XX");
    PauliString XX_ = XX;

    qc.apply(XX_);

    return XX == XX_;
  };

  CliffordTable t(Z2);
  MatrixProductState state(nqb, 1u << nqb);
  randomize_state_haar(rng, state);

  std::string s;
  for (size_t i = 0; i < nqb; i++) {
    s += "X";
  }

  PauliString T(s);
  double e = state.expectation(T);


  for (size_t i = 0; i < 1000; i++) {
    uint32_t q = rng() % (nqb - 1);
    t.apply_random(rng, {q, q+1}, state);

    double c = state.expectation(T);
    ASSERT(is_close(e, state.expectation(T)), fmt::format("Expectation of {} changed from {} to {}.", T.to_string_ops(), e, c));
  }
  
  return true;
}

#define ADD_TEST(x) tests[#x] = x;

int main() {
  std::map<std::string, bool> tests;

  //assert(test_binary_polynomial());
  //assert(test_binary_matrix());
  //assert(test_generator_matrix());
  //assert(test_random_regular_graph());
  //assert(test_parity_check_reduction());
  //assert(test_leaf_removal());
  ADD_TEST(test_nonlocal_mps());
  ADD_TEST(test_statevector());
  ADD_TEST(test_mps());
  ADD_TEST(test_partial_trace());
  ADD_TEST(test_clifford_states_unitary());
  ADD_TEST(test_pauli_reduce());
  ADD_TEST(test_mps_measure());  
  ADD_TEST(test_weak_measure());
  ADD_TEST(test_z2_clifford());

  for (const auto& [name, result] : tests) {
    std::cout << fmt::format("{:>30}: {}\n", name, result ? "\033[1;32m PASSED \033[0m" : "\033[1;31m FAILED\033[0m");
  }
}
