#include <random>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "QuantumState.h"
#include "CliffordState.h"
#include "BinaryPolynomial.h"
#include "Graph.hpp"
#include "Display.h"
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
bool is_close_eps(double eps, T first, V second) {
  return std::abs(first - second) < eps;
}

template <typename T, typename V>
bool is_close(T first, V second) {
  return is_close_eps(1e-8, first, second);
}

template <typename T, typename V, typename... Args>
bool is_close_eps(double eps, T first, V second, Args... args) {
  if (!is_close(first, second)) {
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
    auto c1 = std::abs(first.expectation(p));
    auto c2 = std::abs(second.expectation(p));
    ASSERT(is_close_eps(1e-4, c1, c2));
  }

  return true;
}

template <typename T, typename V, typename... Args>
bool states_close_pauli_fuzz(std::minstd_rand& rng, const T& first, const V& second, const Args&... args) {
  if (!states_close_pauli_fuzz(rng, first, second)) {
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
void randomize_state(QuantumStates&... states) {
  size_t num_qubits = get_num_qubits(states...);
  size_t depth = num_qubits;
  for (size_t i = 0; i < depth; i++) {
    uint32_t k = (i % 2) ? 0 : 1;
    for (uint32_t j = k; j < num_qubits - 1; j += 2) {
      Eigen::Matrix4cd random = Eigen::Matrix4cd::Random();
      std::vector<uint32_t> qubits = {j, j + 1};
      (states.evolve(random, qubits), ...);
    }
  }
}

template <typename... QuantumStates>
void randomize_state_haar(std::minstd_rand& rng, QuantumStates&... states) {
  size_t num_qubits = get_num_qubits(states...);

  QuantumCircuit qc(num_qubits);
  size_t depth = 2;

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
  seed = 314;
  std::minstd_rand rng(seed);

  return rng;
}

bool test_statevector() {
  size_t num_qubits = 3;

  Statevector s(num_qubits);
  s.x(0);

  for (size_t i = 0; i < num_qubits; i++) {
    PauliString Z(num_qubits);
    Z.set_z(i, 1);

    double d = s.expectation(Z);

    if (i == 0) {
      ASSERT(is_close(d, -1.0));
    } else {
      ASSERT(is_close(d, 1.0));
    }
  }

  return true;
}

bool test_mps_vs_statevector() {
  size_t num_qubits = 6;
  size_t bond_dimension = 1u << num_qubits;

  auto rng = seeded_rng();

  Statevector s(num_qubits);
  MatrixProductState mps(num_qubits, bond_dimension);

  for (size_t i = 0; i < 100; i++) {
    randomize_state_haar(rng, mps, s);
    size_t q = rng() % (num_qubits - 2) + 1;
    std::vector<uint32_t> qubits(q);
    std::iota(qubits.begin(), qubits.end(), 0);

    double s1 = s.entropy(qubits, 1);
    double s2 = mps.entropy(qubits, 1);

    Statevector s_(mps);
    double inner = std::abs(s_.inner(s));

    ASSERT(is_close(s1, s2), fmt::format("Entanglement does not match! s1 = {}, s2 = {}", s1, s2));
    ASSERT(is_close(inner, 1.0), "States do not match!");
  }

  return true;
}

bool test_mps_expectation() {
  size_t num_qubits = 6;
  size_t bond_dimension = 1u << num_qubits;

  auto rng = seeded_rng();

  Statevector s(num_qubits);
  MatrixProductState mps(num_qubits, bond_dimension);
  randomize_state_haar(rng, mps, s);
  ASSERT(states_close(s, mps), "States are not close.");

  auto mat_to_str = [](const Eigen::MatrixXcd& mat) {
    std::stringstream ss;
    ss << mat;
    return ss.str();
  };

  for (size_t i = 0; i < 100; i++) {
    size_t r = rng() % 2;
    size_t nqb;
    if (r == 0) {
      nqb = rng() % num_qubits + 1;
    } else if (r == 1) {
      nqb = rng() % 4 + 1;
    }

    size_t q = rng() % (num_qubits - nqb + 1);
    std::vector<uint32_t> qubits(nqb);
    std::iota(qubits.begin(), qubits.end(), q);

    if (r == 0) {
      MatrixProductMixedState mpo = mps.partial_trace_mpo({});

      PauliString P = PauliString::rand(nqb, rng);
      PauliString Pp = P.superstring(qubits, num_qubits);

      double d1 = s.expectation(Pp);
      double d2 = mps.expectation(Pp);
      double d3 = mpo.expectation(Pp);

      ASSERT(is_close(d1, d2, d3), fmt::format("<{}> on {} = {}, {}, {}\n", P, qubits, d1, d2, d3));
    } else if (r == 1) {
      Eigen::MatrixXcd M = haar_unitary(nqb, rng);

      std::complex<double> d1 = s.expectation(M, qubits);
      std::complex<double> d2 = mps.expectation(M, qubits);
      ASSERT(is_close(d1, d2), fmt::format("<{}> = {:.3f} + {:.3f}i, {:.3f} + {:.3f}i\n", mat_to_str(M), d1.real(), d1.imag(), d2.real(), d2.imag()));
    }
  }

  ASSERT(mps.debug_tests(), "MPS failed debug tests.");

  return true;
}

bool test_mpo_expectation() {
  size_t num_qubits = 6;
  size_t bond_dimension = 1u << num_qubits;

  auto rng = seeded_rng();

  Statevector s(num_qubits);
  MatrixProductState mps(num_qubits, bond_dimension);
  randomize_state_haar(rng, mps, s);
  ASSERT(states_close(s, mps), "States are not close.");

  std::vector<uint32_t> qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);

  for (size_t i = 0; i < 100; i++) {
    std::shuffle(qubits.begin(), qubits.end(), rng);
    size_t num_traced_qubits = 3;
    std::vector<uint32_t> traced_qubits = {0, 1, 2};


    //size_t num_traced_qubits = rng() % (num_qubits - 3);
    size_t num_remaining_qubits = num_qubits - num_traced_qubits;
    //std::vector<uint32_t> traced_qubits(qubits.begin(), qubits.begin() + num_traced_qubits);


    //size_t nqb = rng() % (num_remaining_qubits - 1) + 1;
    size_t nqb = 1;

    size_t q = rng() % (num_remaining_qubits - nqb + 1);
    std::vector<uint32_t> qubits_p(nqb);
    std::iota(qubits_p.begin(), qubits_p.end(), q);

    std::cout << fmt::format("Tracing over {}\n", traced_qubits);
    auto mpo = mps.partial_trace(traced_qubits);
    auto dm = s.partial_trace(traced_qubits);

    PauliString P = PauliString::rand(nqb, rng);
    PauliString Pp = P.superstring(qubits_p, num_remaining_qubits);

    std::cout << fmt::format("P = {}, Pp = {}, num_remaining_qubits = {}\n", P, Pp, num_remaining_qubits);

    double d1 = dm->expectation(Pp);
    double d2 = mpo->expectation(Pp);

    std::cout << fmt::format("<{}> -> {:.3f} and {:.3f}\n", Pp, d1, d2);

    ASSERT(is_close(d1, d2), fmt::format("<{}> on {} = {}, {}\n", P, qubits_p, d1, d2));
  }

  return true;
}

bool test_clifford_states_unitary() {
  constexpr size_t nqb = 4;
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
        fmt::format("p1 = {} and p2 = {}\nreduced to {} and {}.", p1, p2, p1_, p2_));
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

    auto rho0 = rho.partial_trace_density_matrix({});
    auto rhoA = rho.partial_trace_density_matrix(qubitsA);
    auto rhoB = rhoA.partial_trace_density_matrix(qubitsB);

    auto mpo0 = mps.partial_trace_mpo({});
    auto mpoA = mps.partial_trace_mpo(qubitsA);
    auto mpoB = mpoA.partial_trace_mpo(qubitsB);

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
  constexpr size_t nqb = 6;

  auto rng = seeded_rng();

  MatrixProductState mps(nqb, 1u << nqb);
  Statevector sv(nqb);

  for (size_t i = 0; i < 100; i++) {
    randomize_state_haar(rng, mps, sv);

    ASSERT(states_close(sv, mps), fmt::format("States do not agree before measurement.\n"));

    for (size_t j = 0; j < 5; j++) {
      uint32_t r = rng() % 2 + 1;
      PauliString P = PauliString::rand(r, rng);
      uint32_t q = rng() % (nqb + 1 - r);
      std::vector<uint32_t> qubits(r);
      std::iota(qubits.begin(), qubits.end(), q);

      int s = rng();
      QuantumState::seed(s);
      bool b1 = mps.measure(P, qubits);
      QuantumState::seed(s);
      bool b2 = sv.measure(P, qubits);

      ASSERT(mps.debug_tests(), fmt::format("MPS failed debug tests for P = {} on {}.", P, qubits));
      ASSERT(b1 == b2, fmt::format("Different measurement outcomes observed for {}.", P));
      ASSERT(states_close(sv, mps), fmt::format("States don't match after measurement of {} on {}.\n{}\n{}", P, qubits, sv.to_string(), mps.to_string()));
    }
  }

  return true;
}

bool test_weak_measure() {
  constexpr size_t nqb = 6;

  auto rng = seeded_rng();

  MatrixProductState mps(nqb, 1u << nqb);
  Statevector sv(nqb);

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
      int s = rng();
      QuantumState::seed(s);
      bool b1 = mps.weak_measure(P, qubits, beta);
      QuantumState::seed(s);
      bool b2 = sv.weak_measure(P, qubits, beta);

      double d = std::abs(sv.inner(Statevector(mps)));

      ASSERT(b1 == b2, "Different measurement outcomes observed.");
      ASSERT(states_close(sv, mps), fmt::format("States don't match after weak measurement of {} on {}. d = {} \n{}\n{}", P, qubits, d, sv.to_string(), mps.to_string()));
    }
  }

  return true;
}

bool test_z2_clifford() {
  auto rng = seeded_rng();

  constexpr size_t nqb = 8;

  auto XX_sym = [](const QuantumCircuit& qc) {
    PauliString XX("XX");
    PauliString XX_ = XX;

    qc.apply(XX_);

    return XX == XX_;
  };

  auto ZZ_sym = [](const QuantumCircuit& qc) {
    PauliString ZZ("ZZ");
    PauliString ZZ_ = ZZ;

    qc.apply(ZZ_);

    return ZZ == ZZ_;
  };

  CliffordTable tXX(XX_sym);
  CliffordTable tZZ(ZZ_sym);

  MatrixProductState state(nqb, 1u << nqb);
  randomize_state_haar(rng, state);

  std::string sx, sz;
  for (size_t i = 0; i < nqb; i++) {
    sx += "X";
    sz += "Z";
  }

  PauliString Tx(sx);
  PauliString Tz(sz);

  for (size_t i = 0; i < 1000; i++) {
    uint32_t q = rng() % (nqb - 1);
    double tz1 = state.expectation(Tz);
    tZZ.apply_random(rng, {q, q+1}, state);
    double tz2 = state.expectation(Tz);
    ASSERT(is_close(tz1, tz2), fmt::format("Expectation of {} changed from {} to {}", Tz, tz1, tz2));

    q = rng() % (nqb - 1);
    double tx1 = state.expectation(Tx);
    tXX.apply_random(rng, {q, q+1}, state);
    double tx2 = state.expectation(Tx);
    ASSERT(is_close(tx1, tx2), fmt::format("Expectation of {} changed from {} to {}", Tx, tx1, tx2));
  }
  
  return true;
}

// Check that trivial partial trace and MPS give same results from sample_pauli
bool test_mpo_sample_paulis() {
  auto rng = seeded_rng();

  constexpr size_t nqb = 8;
  
  MatrixProductState mps(nqb, 1u << nqb);
  randomize_state_haar(rng, mps);
  MatrixProductMixedState mpo = mps.partial_trace_mpo({});

  int s = rng();

  constexpr size_t num_samples = 100;

  QuantumState::seed(s);
  auto paulis1 = mps.sample_paulis({}, num_samples);
  QuantumState::seed(s);
  auto paulis2 = mpo.sample_paulis({}, num_samples);
  for (size_t i = 0; i < num_samples; i++) {
    auto [P1, t1_] = paulis1[i];
    auto [P2, t2_] = paulis2[i];

    double t1 = t1_[0];
    double t2 = t2_[0];
    ASSERT(P1 == P2 && is_close(t1, t2), fmt::format("Paulis <{}> = {:.3f} and <{}> = {:.3f} do not match in sample_paulis.", P1, t1, P2, t2));
  }

  mpo = mps.partial_trace_mpo({0, 1, 2});

  auto paulis = mpo.sample_paulis({}, 1);

  mpo = mps.partial_trace_mpo({0, 1, 5});
  bool found_error = false;
  try {
    paulis = mpo.sample_paulis({}, 1);
  } catch (const std::runtime_error& e) {
    found_error = true;
  }

  ASSERT(found_error);


  return true;
}

bool test_mps_inner() {
  constexpr size_t nqb = 6;
  auto rng = seeded_rng();

  MatrixProductState mps1(nqb, 64);
  MatrixProductState mps2(nqb, 64);
  Statevector s1(nqb);
  Statevector s2(nqb);

  for (size_t i = 0; i < 10; i++) {
    randomize_state_haar(rng, mps1, s1);
    randomize_state_haar(rng, mps2, s2);

    auto c1 = mps1.inner(mps2);
    auto c2 = s1.inner(s2);

    ASSERT(is_close(c1, c2));
  }

  return true;
}

bool test_mps_reverse() {
  auto rng = seeded_rng();
  std::vector<size_t> nqbs = {2, 3, 4, 5, 6};

  for (auto nqb : nqbs) {
    MatrixProductState mps(nqb, 1u << nqb);

    randomize_state_haar(rng, mps);

    MatrixProductState mps_r = mps;
    auto c1 = mps.inner(mps_r);
    auto c2 = mps_r.inner(mps);
    ASSERT(is_close(c1, std::conj(c2)));

    mps_r.reverse();
    for (size_t i = 0; i < nqb/2; i++) {
      mps_r.swap(i, nqb - i - 1);
    }

    ASSERT(states_close(mps, mps_r));
  }

  return true;
}

bool test_batch_weak_measure_sv() {
  auto rng = seeded_rng();

  constexpr size_t nqb = 6;

  Statevector state1;
  Statevector state2;

  for (size_t i = 0; i < 100; i++) {
    state1 = Statevector(nqb);
    state2 = Statevector(nqb);

    QuantumCircuit qc(nqb);
    qc.h(0);
    qc.cx(0, 1);
    qc.apply(state1, state2);

    std::vector<WeakMeasurementData> measurements;
    measurements.push_back({PauliString("+Z"), {0u}, 1.0});
    measurements.push_back({PauliString("+Z"), {1u}, 1.0});
    
    int s = rng();

    QuantumState::seed(s);
    for (auto [p, q, b] : measurements) {
      state1.weak_measure(p, q, b);
    }

    QuantumState::seed(s);
    state2.weak_measure(measurements);

    auto c = std::abs(state1.inner(state2));
    ASSERT(is_close_eps(1e-4, c, 1.0), "States not close after weak measurements.");
  }

  return true;
}

bool test_batch_measure() {
  auto rng = seeded_rng();

  constexpr size_t nqb = 16;

  MatrixProductState mps1(nqb, 128, 1e-8);
  MatrixProductState mps2(nqb, 128, 1e-8);

  for (size_t i = 0; i < 100; i++) {
    randomize_state_haar(rng, mps1, mps2);
    std::vector<MeasurementData> measurements;

    size_t num_measurements = rng() % (nqb/2);
    for (size_t i = 0; i < num_measurements; i++) {
      std::vector<uint32_t> qubits;
      PauliString P;
      if (rng() % 2) {
        uint32_t q = rng() % (nqb - 1);
        qubits = {q, q+1};
        P = PauliString::rand(2, rng);
      } else {
        uint32_t q = rng() % nqb;
        qubits = {q};
        P = PauliString::rand(1, rng);
      }
      measurements.push_back({P, qubits});
    }

    std::sort(measurements.begin(), measurements.end(), [](const MeasurementData& m1, const MeasurementData& m2) {
      return std::get<1>(m1)[0] < std::get<1>(m2)[0];
    });

    int s = rng();

    QuantumState::seed(s);
    for (auto [p, q] : measurements) {
      mps1.measure(p, q);
    }

    QuantumState::seed(s);
    mps2.measure(measurements);

    auto c = std::abs(mps1.inner(mps2));

    ASSERT(is_close_eps(1e-2, c, 1.0), "States not equal after batch measurements.");
  }

  return true;
}

bool test_batch_weak_measure() {
  auto rng = seeded_rng();

  constexpr size_t nqb = 8;

  MatrixProductState mps1(nqb, 32);
  MatrixProductState mps2(nqb, 32);
  Statevector sv(nqb);

  for (size_t i = 0; i < 100; i++) {
    randomize_state_haar(rng, mps1, mps2, sv);
    std::vector<WeakMeasurementData> measurements;


    size_t num_measurements = rng() % (nqb/2);
    std::vector<bool> mask(nqb, false);
    for (size_t i = 0; i < num_measurements; i++) {
      std::vector<uint32_t> qubits;
      PauliString P;
      if (rng() % 2) {
        uint32_t q = rng() % (nqb - 1);
        qubits = {q, q+1};
        P = PauliString::rand(2, rng);
      } else {
        uint32_t q = rng() % nqb;
        qubits = {q};
        P = PauliString::rand(1, rng);
      }
      double beta = 1.0;
      measurements.push_back({P, qubits, beta});
    }

    std::sort(measurements.begin(), measurements.end(), [](const WeakMeasurementData& m1, const WeakMeasurementData& m2) {
      return std::get<1>(m1)[0] < std::get<1>(m2)[0];
    });

    int s = rng();

    QuantumState::seed(s);
    auto c0 = std::abs(mps1.inner(mps2));
    for (auto [p, q, b] : measurements) {
      mps1.weak_measure(p, q, b);
    }

    QuantumState::seed(s);
    mps2.weak_measure(measurements);

    QuantumState::seed(s);
    sv.weak_measure(measurements);

    auto c1 = std::abs(mps1.inner(mps2));

    // TODO states likely differ for LOCAL observables. Need to confirm by checking against local observables.
    ASSERT(is_close_eps(1e-2, c1, 1.0), fmt::format("States not equal after batch weak measurements. Inner product = {:.5f}, c0 = {:.5f}", c1, c0));
    ASSERT(states_close(sv, mps2), fmt::format("MPS does not match statevector after batch weak_measurements."));
  }

  return true;
}

bool test_statevector_to_mps() {
  auto rng = seeded_rng();
  constexpr size_t nqb = 4;

  Statevector sv(nqb);
  MatrixProductState mps1(nqb, 1u << nqb);

  for (size_t i = 0; i < 10; i++) {
    randomize_state_haar(rng, sv, mps1);
    MatrixProductState mps2(sv, 1u << nqb);
    ASSERT(is_close(std::abs(mps1.inner(mps2)), 1.0), "States not close after translating from Statevector to MatrixProductState.");
  }

  return true;
}

bool test_pauli_expectation_sweep() {
  auto rng = seeded_rng();
  constexpr size_t nqb = 8;

  MatrixProductState mps(nqb, 1u << 8);
  Statevector sv(nqb);
  for (size_t i = 0; i < 10; i++) {
    randomize_state_haar(rng, mps, sv);
  }

  for (size_t i = 0; i < 20; i++) {
    randomize_state_haar(rng, mps, sv);
    PauliString P = PauliString::rand(nqb, rng);
    size_t q1_ = rng() % nqb;
    size_t q2_ = rng() % nqb;
    while (q2_ == q1_) {
      q2_ = rng() % nqb;
    }
    size_t q1 = std::min(q1_, q2_);
    size_t q2 = std::max(q1_, q2_);
    std::vector<double> expectation1 = mps.pauli_expectation_left_sweep(P, q1, q2);
    std::vector<double> expectation2;
    std::vector<double> expectation2_sv;
    std::vector<uint32_t> sites(q1);
    std::iota(sites.begin(), sites.end(), 0);
    for (uint32_t i = q1; i < q2; i++) {
      sites.push_back(i);
      double c = mps.expectation(P.substring(sites));
      expectation2.push_back(c);

      c = sv.expectation(P.substring(sites));
      expectation2_sv.push_back(c);
    }

    ASSERT(expectation1.size() == expectation2.size(), "Did not get the same number of values.");
    for (size_t j = 0; j < expectation1.size(); j++) {
      ASSERT(is_close_eps(1e-4, std::abs(expectation1[j]), std::abs(expectation2[j]), std::abs(expectation2_sv[j])), 
          fmt::format("Partial expectations did not match on left sweep: \nexp1 = {}\nexp2 = {}\nexp2_sv = {}\n", expectation1, expectation2, expectation2_sv));
    }

    expectation1 = mps.pauli_expectation_right_sweep(P, q1, q2);
    expectation2.clear();
    expectation2_sv.clear();
    sites.clear();

    sites = std::vector<uint32_t>(nqb - q2 - 1);
    std::iota(sites.begin(), sites.end(), q2 + 1);
    std::reverse(sites.begin(), sites.end());

    for (uint32_t i = q2; i > q1; i--) {
      sites.push_back(i);
      double c = mps.expectation(P.substring(sites));
      expectation2.push_back(c);

      c = sv.expectation(P.substring(sites));
      expectation2_sv.push_back(c);
    }

    ASSERT(expectation1.size() == expectation2.size(), "Did not get the same number of values.");
    for (size_t j = 0; j < expectation1.size(); j++) {
      ASSERT(is_close_eps(1e-4, std::abs(expectation1[j]), std::abs(expectation2[j]), std::abs(expectation2_sv[j])), 
          fmt::format("Partial expectations did not match on right sweep: \nexp1 = {}\nexp2 = {}\nexp2_sv = {}\n", expectation1, expectation2, expectation2_sv));
    }
  }

  return true;
}

bool test_purity() {
  auto rng = seeded_rng();
  constexpr size_t nqb = 6;

  MatrixProductState mps(nqb, 1u << nqb);
  for (size_t i = 0; i < 10; i++) {
    randomize_state_haar(rng, mps);

    uint32_t num_traced_qubits = rng() % nqb;
    std::vector<uint32_t> traced_qubits(nqb);
    std::iota(traced_qubits.begin(), traced_qubits.end(), 0);
    std::shuffle(traced_qubits.begin(), traced_qubits.end(), rng);
    traced_qubits = std::vector<uint32_t>(traced_qubits.begin(), traced_qubits.begin() + num_traced_qubits);

    MatrixProductMixedState mpo = mps.partial_trace_mpo({0, 1, 3, 5});
    DensityMatrix dm(mpo);

    double p1 = mpo.purity();
    double p2 = dm.purity();

    ASSERT(is_close(p1, p2), "Purity of DensityMatrix and MatrixProductMixedState do not match.");
  }

  return true;
}


bool test_projector() {
  auto rng = seeded_rng();
  constexpr size_t nqb = 6;

  MatrixProductState mps(nqb, 1u << nqb);

  for (size_t i = 0; i < 100; i++) {
    randomize_state_haar(rng, mps);
    size_t n = rng() % 3 + 1;
    PauliString P = PauliString::rand(n, rng);
    uint32_t q = rng() % (nqb - n + 1);
    std::vector<uint32_t> qubits(n);
    std::iota(qubits.begin(), qubits.end(), q);
    Eigen::MatrixXcd id = Eigen::MatrixXcd::Identity(1u << n, 1u << n);
    Eigen::MatrixXcd proj = (id + P.to_matrix())/2.0;

    PauliString Pp = P.superstring(qubits, nqb);
    double d1 = std::abs(mps.expectation(proj, qubits));
    double d2 = (1.0 + mps.expectation(Pp))/2.0;

    ASSERT(is_close(d1, d2));
  }

  return true;
}

bool test_pauli() {
  Pauli id = Pauli::I;
  Pauli x = Pauli::X;
  Pauli y = Pauli::Y;
  Pauli z = Pauli::Z;

  ASSERT(pauli_to_char(id) == 'I');
  ASSERT(pauli_to_char(x) == 'X');
  ASSERT(pauli_to_char(y) == 'Y');
  ASSERT(pauli_to_char(z) == 'Z');

  ASSERT(pauli_to_char(id * id) == 'I');
  ASSERT(pauli_to_char(id * x) == 'X');
  ASSERT(pauli_to_char(x * id) == 'X');
  ASSERT(pauli_to_char(id * y) == 'Y');
  ASSERT(pauli_to_char(y * id) == 'Y');
  ASSERT(pauli_to_char(id * z) == 'Z');
  ASSERT(pauli_to_char(z * id) == 'Z');
  ASSERT(pauli_to_char(x * x) == 'I');
  ASSERT(pauli_to_char(x * y) == 'Z');
  ASSERT(pauli_to_char(y * x) == 'Z');
  ASSERT(pauli_to_char(x * z) == 'Y');
  ASSERT(pauli_to_char(z * x) == 'Y');
  ASSERT(pauli_to_char(y * z) == 'X');
  ASSERT(pauli_to_char(z * y) == 'X');

  return true;
}

bool test_mpo_sample_paulis_montecarlo() {
  auto rng = seeded_rng();
  constexpr size_t nqb = 8;

  int s = rng();
  MatrixProductState mps(nqb, 1u << nqb);
  Statevector sv(nqb);

  randomize_state_haar(rng, mps, sv);
  MatrixProductMixedState mpo = mps.partial_trace_mpo({0,3,4,6});
  std::cout << "PRINTING!\n";
  mpo.print_mps();

  size_t num_samples = 100;
  ProbabilityFunc p = [](double t) { return t*t; };

  std::vector<uint32_t> qubitsA = {0, 1, 2};
  std::vector<uint32_t> qubitsB = {3, 4};
  std::vector<uint32_t> qubitsC = {5, 6, 7};
  std::vector<QubitSupport> supports = {qubitsA, qubitsB, qubitsC};
  size_t num_supports = supports.size();
  

  auto samples = mpo.sample_paulis_montecarlo({}, num_samples, 0, p);


  QuantumState::seed(s);
  //auto samples1 = mps.sample_paulis_montecarlo(supports, num_samples, 0, p);

  QuantumState::seed(s);
  //auto samples2 = sv.sample_paulis_montecarlo(supports, num_samples, 0, p);



  //for (size_t i = 0; i < num_samples; i++) {
  //  auto [p1, t1] = samples1[i];
  //  auto [p2, t2] = samples2[i];

  //  ASSERT(p1 == p2, fmt::format("Paulis {} and {} do not match.", p1, p2));
  //  for (size_t i = 0; i < num_supports; i++) {
  //    ASSERT(is_close(t1[i], t2[i]), fmt::format("Amplitudes {:.3f} and {:.3f} on qubits {} do not match.", t1[i], t2[i], to_qubits(supports[i])));
  //  }
  //}

  return true;
}

bool test_sample_paulis_exhaustive() {
  auto rng = seeded_rng();
  constexpr size_t nqb = 6;

  int s = rng();
  Statevector sv(nqb);

  randomize_state_haar(rng, sv);

  std::vector<uint32_t> qubitsA = {0, 1, 2};
  std::vector<uint32_t> qubitsB = {3, 4, 5};
  double mmi = sv.magic_mutual_information_montecarlo(qubitsA, qubitsB, 100, 100);
  std::cout << fmt::format("{:.3f}\n", mmi);


  return true;
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

int main(int argc, char *argv[]) {
  std::map<std::string, TestResult> tests;
  std::set<std::string> test_names;

  bool run_all = (argc == 1);

  if (!run_all) {
    for (size_t i = 1; i < argc; i++) {
      test_names.insert(argv[i]);
    }
  }

  ADD_TEST(test_nonlocal_mps);
  ADD_TEST(test_statevector);
  ADD_TEST(test_mps_vs_statevector);
  ADD_TEST(test_mps_expectation);
  ADD_TEST(test_mpo_expectation);
  ADD_TEST(test_partial_trace);
  ADD_TEST(test_clifford_states_unitary);
  ADD_TEST(test_pauli_reduce);
  ADD_TEST(test_mps_measure);  
  ADD_TEST(test_weak_measure);
  ADD_TEST(test_z2_clifford);
  ADD_TEST(test_mpo_sample_paulis);
  ADD_TEST(test_mps_inner);
  ADD_TEST(test_mps_reverse);
  ADD_TEST(test_statevector_to_mps);
  ADD_TEST(test_batch_weak_measure);
  ADD_TEST(test_batch_measure);
  ADD_TEST(test_batch_weak_measure_sv);
  ADD_TEST(test_pauli_expectation_sweep);
  ADD_TEST(test_purity);
  ADD_TEST(test_projector);
  ADD_TEST(test_mpo_sample_paulis_montecarlo);
  ADD_TEST(test_pauli);
  ADD_TEST(test_sample_paulis_exhaustive);


  double total_duration = 0.0;
  for (const auto& [name, result] : tests) {
    auto [passed, duration] = result;
    std::cout << fmt::format("{:>35}: {} ({:.2f} seconds)\n", name, passed ? "\033[1;32m PASSED \033[0m" : "\033[1;31m FAILED\033[0m", duration/1e6);
    total_duration += duration;
  }

  std::cout << fmt::format("Total duration: {:.2f} seconds\n", total_duration/1e6);
}
