#include <random>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "QuantumState.h"
#include "CliffordState.h"
#include "BinaryPolynomial.h"
#include "Graph.hpp"
#include "Display.h"
#include <iostream>

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
    auto c1 = first.expectation(p);
    auto c2 = second.expectation(p);
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
void randomize_state_clifford(std::minstd_rand& rng, size_t depth, QuantumStates&... states) {
  size_t num_qubits = get_num_qubits(states...);

  QuantumCircuit qc(num_qubits);

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

Qubits random_qubits(size_t num_qubits, size_t k, std::minstd_rand& rng) {
  Qubits qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  std::shuffle(qubits.begin(), qubits.end(), rng);

  Qubits r(qubits.begin(), qubits.begin() + k);
  return r;
}

QubitInterval random_interval(size_t num_qubits, size_t k, std::minstd_rand& rng) {
  uint32_t q1 = rng() % (num_qubits - k + 1);
  uint32_t q2 = q1 + k;

  return std::make_pair(q1, q2);
}

std::minstd_rand seeded_rng() {
  thread_local std::random_device gen;
  int seed = gen();
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

    double d = s.expectation(Z).real();

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

    ASSERT(is_close_eps(1e-4, s1, s2), fmt::format("Entanglement does not match! s1 = {}, s2 = {}", s1, s2));
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
      MatrixProductState mpo = mps.partial_trace_mps({});

      PauliString P = PauliString::rand(nqb, rng);
      PauliString Pp = P.superstring(qubits, num_qubits);

      std::complex<double> d1 = s.expectation(Pp);
      std::complex<double> d2 = mps.expectation(Pp);
      std::complex<double> d3 = mpo.expectation(Pp);

      ASSERT(is_close(d1, d2, d3), fmt::format("<{}> on {} = {:.3f} + {:.3f}i, {:.3f} + {:.3f}i, {:.3f} + {:.3f}i\n", P, qubits, d1.real(), d1.imag(), d2.real(), d2.imag(), d3.real(), d3.imag()));
    } else if (r == 1) {
      Eigen::MatrixXcd M = haar_unitary(nqb, rng);

      std::complex<double> d1 = s.expectation(M, qubits);
      std::complex<double> d2 = mps.expectation(M, qubits);
      ASSERT(is_close(d1, d2), fmt::format("<{}> = {:.3f} + {:.3f}i, {:.3f} + {:.3f}i\n", mat_to_str(M), d1.real(), d1.imag(), d2.real(), d2.imag()));
    }
  }

  //ASSERT(mps.debug_tests(), "MPS failed debug tests.");

  return true;
}

bool test_mpo_expectation() {
  constexpr size_t nqb = 6;
  auto rng = seeded_rng();

  Statevector s(nqb);
  MatrixProductState mps(nqb, 1u << nqb);
  randomize_state_haar(rng, mps, s);
  ASSERT(states_close(s, mps), "States are not close.");

  for (size_t i = 0; i < 100; i++) {
    size_t num_traced_qubits = 3;
    Qubits qubits = random_qubits(nqb, num_traced_qubits, rng);

    size_t num_remaining_qubits = nqb - num_traced_qubits;

    size_t nqb = rng() % (num_remaining_qubits - 1) + 1;
    //size_t nqb = 1;

    size_t q = rng() % (num_remaining_qubits - nqb + 1);
    std::vector<uint32_t> qubits_p(nqb);
    std::iota(qubits_p.begin(), qubits_p.end(), q);

    auto mpo = mps.partial_trace(qubits);
    auto dm = s.partial_trace(qubits);

    PauliString P = PauliString::rand(nqb, rng);
    PauliString Pp = P.superstring(qubits_p, num_remaining_qubits);

    std::complex<double> d1 = dm->expectation(Pp);
    std::complex<double> d2 = mpo->expectation(Pp);

    ASSERT(is_close(d1, d2), fmt::format("<{}> on {} = {:.3f} + {:.3f}i, {:.3f} + {:.3f}i\n", P, qubits_p, d1.real(), d1.imag(), d2.real(), d2.imag()));
  }

  return true;
}

bool test_clifford_states_unitary() {
  constexpr size_t nqb = 6;
  QuantumCHPState chp(nqb);
  QuantumGraphState graph(nqb);
  Statevector sv(nqb);

  auto rng = seeded_rng();

  for (size_t i = 0; i < 10; i++) {
    QuantumCircuit qc(nqb);
    qc.append(random_clifford(nqb, rng));

    // TODO include measurements
    //for (size_t j = 0; j < 3; j++) {
    //  size_t q = rng() % nqb;
    //  qc.mzr(q);
    //}

    qc.apply(sv, chp, graph);



    Statevector sv_chp = chp.to_statevector();
    //Statevector sv_graph = graph.to_statevector();

    ASSERT(states_close(sv, sv_chp), fmt::format("Clifford simulators disagree."));
  }

  return true;
}

bool test_pauli_reduce() {
  auto rng = seeded_rng();

  for (size_t i = 0; i < 100; i++) {
    size_t nqb =  rng() % 20 + 1;
    PauliString p1 = PauliString::randh(nqb, rng);
    PauliString p2 = PauliString::randh(nqb, rng);
    while (p2.commutes(p1)) {
      p2 = PauliString::randh(nqb, rng);
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

bool test_mpo_constructor() {
  size_t nqb = 6;
  auto rng = seeded_rng();
  MatrixProductState mps(nqb, 1u << nqb);
  DensityMatrix rho(nqb);

  randomize_state_haar(rng, mps, rho);
  std::vector<uint32_t> traced_qubits = {0, 1, 3, 4};

  MatrixProductState mpo = mps.partial_trace_mps(traced_qubits);

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
    Qubits qubitsA = random_qubits(nqb, qA, rng);

    size_t qB = rng() % (nqb / 2);
    Qubits qubitsB = random_qubits(nqb - qA, qB, rng);

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

    auto mpo0 = mps.partial_trace_mps({});
    auto mpoA = mps.partial_trace_mps(qubitsA);
    auto mpoB = mpoA.partial_trace_mps(qubitsB);

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
      PauliString P = PauliString::randh(r, rng);
      uint32_t q = rng() % (nqb + 1 - r);
      std::vector<uint32_t> qubits(r);
      std::iota(qubits.begin(), qubits.end(), q);

      int s = rng();
      QuantumState::seed(s);
      bool b1 = mps.measure(P, qubits);
      QuantumState::seed(s);
      bool b2 = sv.measure(P, qubits);

      //ASSERT(mps.debug_tests(), fmt::format("MPS failed debug tests for P = {} on {}.", P, qubits));
      ASSERT(b1 == b2, fmt::format("Different measurement outcomes observed for {}.", P));
      ASSERT(states_close(sv, mps), fmt::format("States don't match after measurement of {} on {}.\n{}\n{}", P, qubits, sv.to_string(), mps.to_string()));
    }
  }

  return true;
}

bool test_mps_weak_measure() {
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
        P = PauliString::randh(2, rng);
        uint32_t q = rng() % (nqb - 1);
        qubits = {q, q + 1};
      } else {
        P = PauliString::randh(1, rng);
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
    double tz1 = state.expectation(Tz).real();
    tZZ.apply_random(rng, {q, q+1}, state);
    double tz2 = state.expectation(Tz).real();
    ASSERT(is_close_eps(1e-5, tz1, tz2), fmt::format("Expectation of {} changed from {} to {}", Tz, tz1, tz2));

    q = rng() % (nqb - 1);
    double tx1 = state.expectation(Tx).real();
    tXX.apply_random(rng, {q, q+1}, state);
    double tx2 = state.expectation(Tx).real();
    ASSERT(is_close_eps(1e-5, tx1, tx2), fmt::format("Expectation of {} changed from {} to {}", Tx, tx1, tx2));
  }
  
  return true;
}

// Check that trivial partial trace and MPS give same results from sample_pauli
bool test_mpo_sample_paulis() {
  auto rng = seeded_rng();

  constexpr size_t nqb = 8;
  
  MatrixProductState mps(nqb, 1u << nqb);
  randomize_state_haar(rng, mps);
  MatrixProductState mpo = mps.partial_trace_mps({});

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

  mpo = mps.partial_trace_mps({0, 1, 2});

  auto paulis = mpo.sample_paulis({}, 1);

  mpo = mps.partial_trace_mps({0, 1, 5});
  bool found_error = false;
  try {
    paulis = mpo.sample_paulis({}, 1);
  } catch (const std::runtime_error& e) {
    found_error = true;
  }

  ASSERT(found_error, "Did not find appropriate runtime error.");


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
  std::vector<size_t> nqbs = {3, 4, 5, 6};

  for (auto nqb : nqbs) {
    MatrixProductState mps(nqb, 1u << nqb);

    randomize_state_haar(rng, mps);

    Qubits qubits(nqb);
    std::iota(qubits.begin(), qubits.end(), 0);
    size_t k = rng() % (nqb - 2) + 1;

    Qubits traced_qubits;
    if (rng() % 2) {
      traced_qubits = Qubits(qubits.begin(), qubits.begin() + k);
    } else {
      traced_qubits = Qubits(qubits.end() - k, qubits.end());
    }

    auto mps_m = mps.partial_trace_mps(traced_qubits);
    MatrixProductState mps_r_m = mps_m;
    ASSERT(states_close(mps_m, mps_r_m));

    mps_r_m.reverse();

    size_t remaining_nqb = mps_r_m.num_qubits;
    for (size_t i = 0; i < remaining_nqb/2; i++) {
      mps_r_m.swap(i, remaining_nqb - i - 1);
    }
    
    DensityMatrix d1(mps_m);
    DensityMatrix d2(mps_r_m);

    ASSERT(states_close(mps_m, mps_r_m));
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

  constexpr size_t nqb = 12;

  MatrixProductState mps1(nqb, 64, 1e-8);
  MatrixProductState mps2(nqb, 64, 1e-8);

  int t1 = 0;
  int t2 = 0;

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
        P = PauliString::randh(2, rng);
      } else {
        uint32_t q = rng() % nqb;
        qubits = {q};
        P = PauliString::randh(1, rng);
      }
      measurements.push_back({P, qubits});
    }

    std::sort(measurements.begin(), measurements.end(), [](const MeasurementData& m1, const MeasurementData& m2) {
      return std::get<1>(m1)[0] < std::get<1>(m2)[0];
    });

    int s = rng();


    auto start = std::chrono::high_resolution_clock::now();
    QuantumState::seed(s);
    for (auto [p, q] : measurements) {
      mps1.measure(p, q);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    t1 += duration_cast<std::chrono::microseconds>(stop - start).count();

    start = std::chrono::high_resolution_clock::now();
    QuantumState::seed(s);
    mps2.measure(measurements);
    stop = std::chrono::high_resolution_clock::now();
    t2 += duration_cast<std::chrono::microseconds>(stop - start).count();

    auto c = std::abs(mps1.inner(mps2));

    ASSERT(is_close_eps(1e-2, c, 1.0), fmt::format("States not equal after batch measurements. Inner product = {:.5f}", std::abs(c)));
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
        P = PauliString::randh(2, rng);
      } else {
        uint32_t q = rng() % nqb;
        qubits = {q};
        P = PauliString::randh(1, rng);
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

    MatrixProductState mpo = mps.partial_trace_mps({0, 1, 3, 5});
    DensityMatrix dm(mpo);

    double p1 = mpo.purity();
    double p2 = dm.purity();

    ASSERT(is_close(p1, p2), fmt::format("Purity of DensityMatrix and MatrixProductState do not match: {:.3f} and {:.3f}", p1, p2));
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
    PauliString P = PauliString::randh(n, rng);
    uint32_t q = rng() % (nqb - n + 1);
    std::vector<uint32_t> qubits(n);
    std::iota(qubits.begin(), qubits.end(), q);
    Eigen::MatrixXcd id = Eigen::MatrixXcd::Identity(1u << n, 1u << n);
    Eigen::MatrixXcd proj = (id + P.to_matrix())/2.0;

    PauliString Pp = P.superstring(qubits, nqb);
    double d1 = std::abs(mps.expectation(proj, qubits));
    double d2 = (1.0 + mps.expectation(Pp).real())/2.0;
    ASSERT(is_close(d1, d2));
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

bool test_pauli_expectation_tree() {
  auto rng = seeded_rng();
  constexpr size_t nqb = 8;

  MatrixProductState mps(nqb, 64);

  int t1 = 0;
  int t2 = 0;

  for (size_t i = 0; i < 10; i++) {
    randomize_state_haar(rng, mps);

    size_t tqb = rng() % (nqb/2);
    size_t rqb = nqb - tqb;
    Qubits tqubits = random_qubits(nqb, tqb, rng);
    MatrixProductState mpo = mps.partial_trace_mps(tqubits);

    PauliString p = PauliString::rand(rqb, rng);

    PauliExpectationTree tree(mpo, p);

    size_t k = rng() % (rqb / 2) + 1;

    auto test_subsystem = [&](const QubitSupport& support) {
      PauliString p_ = p.substring(support);

      std::complex<double> c1 = mpo.expectation(p_);

      auto interval = p_.support_range();
      std::complex<double> c2;
      if (interval) {
        auto [q1, q2] = interval.value();
        c2 = tree.partial_expectation(q1, q2);
      } else {
        c2 = p_.sign();
      }

      ASSERT(is_close_eps(1e-1, c1, c2), fmt::format("Partial expectations failed. P = {}, P_ = {}, subsystem = {}, c1 = {:.3f} + {:.3f}i, c2 = {:.3f} + {:.3f}i", p, p_, to_qubits(support), c1.real(), c1.imag(), c2.real(), c2.imag()));

      return true;
    };

    for (size_t j = 0; j < 10; j++) {
      QubitSupport subsystem = random_interval(rqb, k, rng);

      ASSERT(test_subsystem(subsystem));

      PauliString p_mut = PauliString::rand(rqb, rng);
      p = p_mut * p;
      tree.modify(p_mut);
    }
  }

  return true;
}

bool test_mps_debug_tests() {
  auto rng = seeded_rng();
  constexpr size_t nqb = 8;

  MatrixProductState mps(nqb, 1u << nqb);

  ASSERT(mps.debug_tests(), "Failed debug tests.");
  randomize_state_haar(rng, mps);
  ASSERT(mps.debug_tests(), "Failed debug tests.");

  return true;
}

bool test_mpo_sample_paulis_montecarlo() {
  auto rng = seeded_rng();
  constexpr size_t nqb = 8;

  MatrixProductState mps(nqb, 1u << nqb);
  Statevector sv(nqb);

  randomize_state_haar(rng, mps, sv);

  size_t num_samples = 100;
  ProbabilityFunc p = [](double t) { return t*t; };

  std::vector<uint32_t> qubitsA = {0, 1, 2};
  std::vector<uint32_t> qubitsB = {3, 4};
  std::vector<uint32_t> qubitsC = {5, 6, 7};
  std::vector<QubitSupport> supports = {qubitsA, qubitsB, qubitsC};
  size_t num_supports = supports.size();
  
  auto test_samples = [&](const auto& samples1, const auto& samples2) {
    for (size_t i = 0; i < num_samples; i++) {
      auto [p1, t1] = samples1[i];
      auto [p2, t2] = samples2[i];

      //std::cout << fmt::format("p1 = {}, p2 = {}\nt1 = {}\nt2 = {}\n", p1, p2, t1, t2);

      ASSERT(p1 == p2, fmt::format("Paulis {} and {} do not match.", p1, p2));
      for (size_t j = 0; j < t1.size(); j++) {
        ASSERT(is_close(t1[j], t2[j]), fmt::format("Amplitudes of {} and {} of {:.3f} and {:.3f} on qubits {} do not match.", p1, p2, t1[j], t2[j], to_qubits(supports[i])));
      }
    }

    return true;
  };

  int s = rng();

  QuantumState::seed(s);
  auto samples1 = mps.sample_paulis_montecarlo(supports, num_samples, 0, p);

  QuantumState::seed(s);
  auto samples2 = sv.sample_paulis_montecarlo(supports, num_samples, 0, p);

  ASSERT(test_samples(samples1, samples2));

  for (size_t i = 0; i < 10; i++) {
    size_t tqb = 4;
    Qubits qubits = random_qubits(nqb, tqb, rng);
    auto mpo = mps.partial_trace(qubits);
    auto rho = sv.partial_trace(qubits);

    size_t rqb = nqb - tqb;

    supports.clear();
    for (uint32_t k = 0; k < rqb; k++) {
      auto rqubits = random_qubits(rqb, rng() % rqb, rng);
      std::sort(rqubits.begin(), rqubits.end());
      rqubits = {k};
      supports.push_back(rqubits);
    }

    QuantumState::seed(s);
    samples1 = mpo->sample_paulis_montecarlo(supports, num_samples, 0, p);

    QuantumState::seed(s);
    samples2 = rho->sample_paulis_montecarlo(supports, num_samples, 0, p);

    ASSERT(test_samples(samples1, samples2));
  }

  return true;
}

// NOTE:
// this test is broken since QuantumState uses calculate_magic_mutual_information_from_samples, and 
// MatrixProductState uses calculate_magic_mutual_information_from_samples2. Need a better test.
bool test_mpo_bipartite_mmi() {
  auto rng = seeded_rng();
  constexpr size_t nqb = 6;

  int s = rng();
  Statevector sv(nqb);
  MatrixProductState mps(nqb, 1u << nqb);

  randomize_state_haar(rng, sv, mps);

  size_t num_samples = 10;
  size_t equilibration_timesteps = 100;

  QuantumState::seed(s);
  auto samples1 = sv.bipartite_magic_mutual_information_montecarlo(num_samples, equilibration_timesteps);
  QuantumState::seed(s);
  auto samples2 = mps.bipartite_magic_mutual_information_montecarlo(num_samples, equilibration_timesteps);

  for (size_t i = 0; i < samples1.size(); i++) {
    ASSERT(is_close_eps(1e-4, samples1[i], samples2[i]), fmt::format("Bipartite magic mutual information samples not equal: {:.3f}, {:.3f}", samples1[i], samples2[i]));
  }



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


  return true;
}

bool test_mps_ising_model() {
  constexpr size_t nqb = 32;
  auto rng = seeded_rng();

  MatrixProductState mps = MatrixProductState::ising_ground_state(nqb, 0.0, 64, 1e-8, 100);
  PauliString P(nqb);
  //for (size_t i = 0; i < nqb; i++) {
  //  P.set_z(i, 1);
  //}
  P.set_z(5, 1);
  P.set_z(6, 1);

  std::complex<double> c = mps.expectation(P);

  return true;
}

bool test_mps_random_clifford() {
  constexpr size_t nqb = 20;
  auto rng = seeded_rng();

  MatrixProductState mps(nqb, 32, 0.0);

  for (size_t i = 0; i < 50; i++) {
    randomize_state_clifford(rng, 2, mps);
    for (size_t q = 0; q < nqb; q++) {
      double r = double(rng())/double(RAND_MAX);
      if (r < 0.1) {
        mps.mzr(q);
      }
    }
  }


  for (size_t i = 0; i < nqb - 1; i++) {
    double t = mps.trace();
    ASSERT(is_close(t, 1.0), fmt::format("Trace was not preserved. Trace = {:.3f}", t));
  }


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

  ADD_TEST(test_mps_debug_tests);
  ADD_TEST(test_nonlocal_mps);
  ADD_TEST(test_mpo_constructor);
  ADD_TEST(test_statevector);
  ADD_TEST(test_mps_vs_statevector);
  ADD_TEST(test_mps_expectation);
  ADD_TEST(test_mpo_expectation);
  ADD_TEST(test_partial_trace);
  ADD_TEST(test_clifford_states_unitary);
  ADD_TEST(test_pauli_reduce);
  ADD_TEST(test_mps_measure);  
  ADD_TEST(test_mps_weak_measure);
  ADD_TEST(test_z2_clifford);
  ADD_TEST(test_mps_inner);
  ADD_TEST(test_mps_reverse);
  ADD_TEST(test_statevector_to_mps);
  ADD_TEST(test_batch_weak_measure);
  ADD_TEST(test_batch_measure);
  ADD_TEST(test_batch_weak_measure_sv);
  ADD_TEST(test_purity);
  ADD_TEST(test_projector);
  ADD_TEST(test_mpo_sample_paulis);
  ADD_TEST(test_pauli_expectation_tree);
  ADD_TEST(test_mpo_sample_paulis_montecarlo);
  //ADD_TEST(test_mpo_bipartite_mmi);
  ADD_TEST(test_sample_paulis_exhaustive);
  ADD_TEST(test_pauli);
  ADD_TEST(test_mps_ising_model);
  ADD_TEST(test_mps_random_clifford);


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
      std::cout << fmt::format("{:>35}: {} ({:.2f} seconds)\n", name, test_passed_str(passed), duration/1e6);
      total_duration += duration;
    }

    std::cout << fmt::format("Total duration: {:.2f} seconds\n", total_duration/1e6);
  }
}
