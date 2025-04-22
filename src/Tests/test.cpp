#include <memory>
#include <random>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "QuantumState.h"
#include "CliffordState.h"
#include "LinearCode.h"
#include "Graph.hpp"
#include "Display.h"
#include "Samplers.h"
#include <iostream>

#include <Frame.h>
using namespace dataframe;
using namespace dataframe::utils;

#define MPS_DEBUG_LEVEL 2

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

  if (d1.get_num_qubits() != d2.get_num_qubits()) {
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
bool states_close_pauli_fuzz(const T& first, const V& second) {
  ASSERT(first.num_qubits == second.num_qubits);

  for (size_t i = 0; i < 100; i++) {
    PauliString p = PauliString::rand(first.num_qubits);
    auto c1 = first.expectation(p);
    auto c2 = second.expectation(p);
    ASSERT(is_close_eps(1e-4, c1, c2));
  }

  return true;
}

template <typename T, typename V, typename... Args>
bool states_close_pauli_fuzz(const T& first, const V& second, const Args&... args) {
  if (!states_close_pauli_fuzz(first, second)) {
    return false;
  } else {
    return states_close(first, args...);
  }
}

template <typename T, typename... QuantumStates>
size_t get_num_qubits(const T& first, const QuantumStates&... args) {
  size_t num_qubits = first.get_num_qubits();

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
      ([&] {
       if constexpr (std::is_same_v<std::decay_t<QuantumStates>, QuantumCircuit>) {
         states.add_gate(random, qubits);
       } else {
         states.evolve(random, qubits);
       }
      }(), ...);
    }
  }
}

template <typename... QuantumStates>
void randomize_state_haar(QuantumStates&... states) {
  size_t num_qubits = get_num_qubits(states...);

  QuantumCircuit qc(num_qubits);
  size_t depth = 2;

  qc.append(generate_haar_circuit(num_qubits, depth, false));
  qc.apply(states...);
}

template <typename... QuantumStates>
void randomize_state_clifford(size_t depth, QuantumStates&... states) {
  size_t num_qubits = get_num_qubits(states...);

  QuantumCircuit qc(num_qubits);

  for (size_t k = 0; k < depth; k++) {
    for (size_t i = 0; i < num_qubits/2 - 1; i++) {
      uint32_t q1 = (k % 2) ? 2*i : 2*i + 1;
      uint32_t q2 = q1 + 1;

      QuantumCircuit rc = random_clifford(2);
      qc.append(random_clifford(2), {q1, q2});
    }
  }

  qc.apply(states...);
}

Qubits random_qubits(size_t num_qubits, size_t k) {
  std::minstd_rand rng(randi());
  Qubits qubits(num_qubits);
  std::iota(qubits.begin(), qubits.end(), 0);
  std::shuffle(qubits.begin(), qubits.end(), rng);

  Qubits r(qubits.begin(), qubits.begin() + k);
  return r;
}

Qubits random_boundary_qubits(size_t num_qubits, size_t k) {
  Qubits qubits;
  if (k == 0) {
    return qubits;
  }

  size_t k1 = randi(0, k+1);
  size_t k2 = k - k1;

  for (uint32_t q = 0; q < k1; q++) {
    qubits.push_back(q);
  }

  for (uint32_t q = 0; q < k2; q++) {
    qubits.push_back(num_qubits - q - 1);
  }
  
  return qubits;
}

QubitInterval random_interval(size_t num_qubits, size_t k) {
  uint32_t q1 = randi() % (num_qubits - k + 1);
  uint32_t q2 = q1 + k;

  return std::make_pair(q1, q2);
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

  Statevector s(num_qubits);
  MatrixProductState mps(num_qubits, bond_dimension);
  mps.set_debug_level(MPS_DEBUG_LEVEL);

  for (size_t i = 0; i < 100; i++) {
    randomize_state_haar(mps, s);
    size_t q = randi() % (num_qubits - 2) + 1;
    std::vector<uint32_t> qubits(q);
    std::iota(qubits.begin(), qubits.end(), 0);

    size_t index = randi() % 4 + 1;
    double s1 = s.entropy(qubits, index);
    double s2 = mps.entropy(qubits, index);

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

  Statevector s(num_qubits);
  MatrixProductState mps(num_qubits, bond_dimension);
  mps.set_debug_level(MPS_DEBUG_LEVEL);
  randomize_state_haar(mps, s);
  ASSERT(states_close(s, mps), "States are not close.");

  auto mat_to_str = [](const Eigen::MatrixXcd& mat) {
    std::stringstream ss;
    ss << mat;
    return ss.str();
  };

  for (size_t i = 0; i < 100; i++) {
    size_t r = randi() % 2;
    size_t nqb;
    if (r == 0) {
      nqb = randi() % num_qubits + 1;
    } else if (r == 1) {
      nqb = randi() % 4 + 1;
    }

    size_t q = randi() % (num_qubits - nqb + 1);
    std::vector<uint32_t> qubits(nqb);
    std::iota(qubits.begin(), qubits.end(), q);

    if (r == 0) {
      MatrixProductState mpo = mps.partial_trace_mps({});

      PauliString P = PauliString::rand(nqb);
      PauliString Pp = P.superstring(qubits, num_qubits);

      std::complex<double> d1 = s.expectation(Pp);
      std::complex<double> d2 = mps.expectation(Pp);
      std::complex<double> d3 = mpo.expectation(Pp);

      ASSERT(is_close(d1, d2, d3), fmt::format("<{}> on {} = {:.3f} + {:.3f}i, {:.3f} + {:.3f}i, {:.3f} + {:.3f}i\n", P, qubits, d1.real(), d1.imag(), d2.real(), d2.imag(), d3.real(), d3.imag()));
    } else if (r == 1) {
      Eigen::MatrixXcd M = haar_unitary(nqb);

      std::complex<double> d1 = s.expectation(M, qubits);
      std::complex<double> d2 = mps.expectation(M, qubits);
      ASSERT(is_close(d1, d2), fmt::format("<{}> = {:.3f} + {:.3f}i, {:.3f} + {:.3f}i\n", mat_to_str(M), d1.real(), d1.imag(), d2.real(), d2.imag()));
    }
  }

  ASSERT(mps.state_valid(), "MPS failed debug tests.");

  return true;
}

bool test_mpo_expectation() {
  constexpr size_t nqb = 6;

  Statevector s(nqb);
  MatrixProductState mps(nqb, 1u << nqb);
  mps.set_debug_level(MPS_DEBUG_LEVEL);
  randomize_state_haar(mps, s);
  ASSERT(states_close(s, mps), "States are not close.");

  for (size_t i = 0; i < 100; i++) {
    size_t num_traced_qubits = 3;
    Qubits qubits = random_boundary_qubits(nqb, num_traced_qubits);

    size_t num_remaining_qubits = nqb - num_traced_qubits;

    size_t nqb = randi() % (num_remaining_qubits - 1) + 1;
    //size_t nqb = 1;

    size_t q = randi() % (num_remaining_qubits - nqb + 1);
    std::vector<uint32_t> qubits_p(nqb);
    std::iota(qubits_p.begin(), qubits_p.end(), q);

    auto mpo = mps.partial_trace(qubits);
    auto dm = s.partial_trace(qubits);

    PauliString P = PauliString::rand(nqb);
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

  for (size_t i = 0; i < 10; i++) {
    QuantumCircuit qc(nqb);
    qc.append(random_clifford(nqb));

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

bool test_nonlocal_mps() {
  size_t nqb = 6;

  std::minstd_rand rng(randi());
  for (size_t i = 0; i < 4; i++) {
    QuantumCircuit qc = generate_haar_circuit(nqb, nqb, true); 

    std::vector<uint32_t> qubit_map(nqb);
    std::iota(qubit_map.begin(), qubit_map.end(), 0);
    std::shuffle(qubit_map.begin(), qubit_map.end(), rng);
    qc.apply_qubit_map(qubit_map);

    MatrixProductState mps(nqb, 20);
    mps.set_debug_level(MPS_DEBUG_LEVEL);
    Statevector sv(nqb);

    qc.apply(sv, mps);

    ASSERT(states_close(sv, mps), fmt::format("States not close after nonlocal circuit: \n{}\n{}", sv.to_string(), mps.to_string()));
  }

  return true;
}

bool test_partial_trace() {
  constexpr size_t nqb = 6;
  static_assert(nqb % 2 == 0);

  for (size_t i = 0; i < 10; i++) {
    MatrixProductState mps(nqb, 1u << nqb);
    mps.set_debug_level(MPS_DEBUG_LEVEL);
    DensityMatrix rho(nqb);
    randomize_state_haar(mps, rho);


    size_t qA = randi() % (nqb / 2);
    qA = 4;
    Qubits qubitsA = random_boundary_qubits(nqb, qA);

    size_t qB = randi() % (nqb / 2);
    if (qA + qB == nqb) qB--;
    Qubits qubitsB = random_boundary_qubits(nqb - qA, qB);

    auto rho0 = rho.partial_trace_density_matrix({});
    auto rhoA = rho.partial_trace_density_matrix(qubitsA);
    auto rhoB = rhoA.partial_trace_density_matrix(qubitsB);

    auto mpo0 = mps.partial_trace_mps({});
    auto mpoA = mps.partial_trace_mps(qubitsA);
    auto mpoB = mpoA.partial_trace_mps(qubitsB);

    PauliString P = PauliString::rand(nqb);

    Qubits qubitsA_ = to_qubits(support_complement(qubitsA, mps.get_num_qubits()));
    Qubits qubitsB_ = to_qubits(support_complement(qubitsB, mpoA.get_num_qubits()));

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

  MatrixProductState mps(nqb, 1u << nqb);
  mps.set_debug_level(MPS_DEBUG_LEVEL);
  Statevector sv(nqb);

  for (size_t i = 0; i < 100; i++) {
    randomize_state_haar(mps, sv);

    ASSERT(states_close(sv, mps), fmt::format("States do not agree before measurement.\n"));

    for (size_t j = 0; j < 5; j++) {
      //uint32_t r = rng() % 2 + 1;
      uint32_t r = 2;
      PauliString P = PauliString::randh(r);
      uint32_t q = randi() % (nqb + 1 - r);
      std::vector<uint32_t> qubits(r);
      std::iota(qubits.begin(), qubits.end(), q);

      int s = randi();
      Random::seed_rng(s);
      bool b1 = mps.measure(Measurement(qubits, P));
      Random::seed_rng(s);
      bool b2 = sv.measure(Measurement(qubits, P));

      ASSERT(mps.state_valid(), fmt::format("MPS failed debug tests for P = {} on {}.", P, qubits));
      ASSERT(b1 == b2, fmt::format("Different measurement outcomes observed for {}.", P));
      ASSERT(states_close(sv, mps), fmt::format("States don't match after measurement of {} on {}.\n{}\n{}", P, qubits, sv.to_string(), mps.to_string()));
    }
  }

  return true;
}

bool test_mps_weak_measure() {
  constexpr size_t nqb = 6;

  MatrixProductState mps(nqb, 1u << nqb);
  mps.set_debug_level(MPS_DEBUG_LEVEL);
  Statevector sv(nqb);

  for (size_t i = 0; i < 100; i++) {
    randomize_state_haar(mps, sv);

    ASSERT(states_close(sv, mps), fmt::format("States do not agree before measurement.\n"));

    for (size_t j = 0; j < 5; j++) {
      PauliString P;
      std::vector<uint32_t> qubits;
      if (randi() % 2) {
        P = PauliString::randh(2);
        uint32_t q = randi() % (nqb - 1);
        qubits = {q, q + 1};
      } else {
        P = PauliString::randh(1);
        uint32_t q = randi() % nqb;
        qubits = {q};
      }


      constexpr double beta = 1.0;
      uint32_t s = randi();
      Random::seed_rng(s);
      bool b1 = mps.weak_measure(WeakMeasurement(qubits, beta, P));
      Random::seed_rng(s);
      bool b2 = sv.weak_measure(WeakMeasurement(qubits, beta, P));

      double d = std::abs(sv.inner(Statevector(mps)));

      ASSERT(b1 == b2, "Different measurement outcomes observed.");
      ASSERT(states_close(sv, mps), fmt::format("States don't match after weak measurement of {} on {}. d = {} \n{}\n{}", P, qubits, d, sv.to_string(), mps.to_string()));
    }
  }

  return true;
}

bool test_z2_clifford() {
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
  state.set_debug_level(MPS_DEBUG_LEVEL);
  randomize_state_haar(state);

  std::string sx, sz;
  for (size_t i = 0; i < nqb; i++) {
    sx += "X";
    sz += "Z";
  }

  PauliString Tx(sx);
  PauliString Tz(sz);

  for (size_t i = 0; i < 1000; i++) {
    uint32_t q = randi() % (nqb - 1);
    double tz1 = state.expectation(Tz).real();
    tZZ.apply_random({q, q+1}, state);
    double tz2 = state.expectation(Tz).real();
    ASSERT(is_close_eps(1e-5, tz1, tz2), fmt::format("Expectation of {} changed from {} to {}", Tz, tz1, tz2));

    q = randi() % (nqb - 1);
    double tx1 = state.expectation(Tx).real();
    tXX.apply_random({q, q+1}, state);
    double tx2 = state.expectation(Tx).real();
    ASSERT(is_close_eps(1e-5, tx1, tx2), fmt::format("Expectation of {} changed from {} to {}", Tx, tx1, tx2));
  }
  
  return true;
}

// Check that trivial partial trace and MPS give same results from sample_pauli
bool test_mpo_sample_paulis() {
  constexpr size_t nqb = 8;
  
  MatrixProductState mps(nqb, 1u << nqb);
  mps.set_debug_level(MPS_DEBUG_LEVEL);
  randomize_state_haar(mps);
  MatrixProductState mpo = mps.partial_trace_mps({});

  uint32_t s = randi();

  constexpr size_t num_samples = 100;

  Random::seed_rng(s);
  auto paulis1 = mps.sample_paulis({}, num_samples);
  Random::seed_rng(s);
  auto paulis2 = mpo.sample_paulis({}, num_samples);
  for (size_t i = 0; i < num_samples; i++) {
    auto [P1, t1_] = paulis1[i];
    auto [P2, t2_] = paulis2[i];

    double t1 = t1_[0];
    double t2 = t2_[0];
    ASSERT(P1 == P2 && is_close(t1, t2), fmt::format("Paulis <{}> = {:.3f} and <{}> = {:.3f} do not match in sample_paulis.", P1, t1, P2, t2));
  }

  // TODO add sample partially traced tests

  return true;
}

bool test_mps_inner() {
  constexpr size_t nqb = 6;

  MatrixProductState mps1(nqb, 64);
  mps1.set_debug_level(MPS_DEBUG_LEVEL);
  MatrixProductState mps2(nqb, 64);
  mps2.set_debug_level(MPS_DEBUG_LEVEL);
  Statevector s1(nqb);
  Statevector s2(nqb);

  for (size_t i = 0; i < 10; i++) {
    randomize_state_haar(mps1, s1);
    randomize_state_haar(mps2, s2);

    auto c1 = mps1.inner(mps2);
    auto c2 = s1.inner(s2);

    ASSERT(is_close(c1, c2));
  }

  return true;
}

bool test_mps_reverse() {
  std::vector<size_t> nqbs = {3, 4, 5, 6};

  for (auto nqb : nqbs) {
    MatrixProductState mps(nqb, 1u << nqb);
    mps.set_debug_level(MPS_DEBUG_LEVEL);

    randomize_state_haar(mps);

    Qubits qubits(nqb);
    std::iota(qubits.begin(), qubits.end(), 0);
    size_t k = randi() % (nqb - 2) + 1;

    Qubits traced_qubits;
    if (randi() % 2) {
      traced_qubits = Qubits(qubits.begin(), qubits.begin() + k);
    } else {
      traced_qubits = Qubits(qubits.end() - k, qubits.end());
    }

    auto mps_m = mps.partial_trace_mps(traced_qubits);
    MatrixProductState mps_r_m = mps_m;
    ASSERT(states_close(mps_m, mps_r_m));

    mps_r_m.reverse();

    size_t remaining_nqb = mps_r_m.get_num_qubits();
    for (size_t i = 0; i < remaining_nqb/2; i++) {
      mps_r_m.swap(i, remaining_nqb - i - 1);
    }
    
    DensityMatrix d1(mps_m);
    DensityMatrix d2(mps_r_m);

    ASSERT(states_close(mps_m, mps_r_m));
  }

  return true;
}

bool test_statevector_to_mps() {
  constexpr size_t nqb = 4;

  Statevector sv(nqb);
  MatrixProductState mps1(nqb, 1u << nqb);
  mps1.set_debug_level(MPS_DEBUG_LEVEL);

  for (size_t i = 0; i < 10; i++) {
    randomize_state_haar(sv, mps1);
    MatrixProductState mps2(sv, 1u << nqb);
    ASSERT(is_close(std::abs(mps1.inner(mps2)), 1.0), "States not close after translating from Statevector to MatrixProductState.");
  }

  return true;
}

bool test_purity() {
  constexpr size_t nqb = 6;
  std::minstd_rand rng(randi());

  MatrixProductState mps(nqb, 1u << nqb);
  mps.set_debug_level(MPS_DEBUG_LEVEL);
  for (size_t i = 0; i < 10; i++) {
    randomize_state_haar(mps);

    uint32_t k = randi() % (nqb - 1);
    Qubits traced_qubits = random_boundary_qubits(nqb, k);

    MatrixProductState mpo = mps.partial_trace_mps(traced_qubits);
    DensityMatrix dm(mpo);

    double t1 = mpo.trace();
    double t2 = dm.trace();
    double p1 = mpo.purity();
    double p2 = dm.purity();

    ASSERT(is_close(p1, p2), fmt::format("Purity of DensityMatrix and MatrixProductState do not match: {:.3f} and {:.3f}", p1, p2));
  }

  return true;
}


bool test_projector() {
  constexpr size_t nqb = 6;

  MatrixProductState mps(nqb, 1u << nqb);
  mps.set_debug_level(MPS_DEBUG_LEVEL);

  for (size_t i = 0; i < 100; i++) {
    randomize_state_haar(mps);
    size_t n = randi() % 3 + 1;
    PauliString P = PauliString::randh(n);
    uint32_t q = randi() % (nqb - n + 1);
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

bool test_mps_debug_tests() {
  constexpr size_t nqb = 8;

  MatrixProductState mps(nqb, 1u << nqb);
  mps.set_debug_level(MPS_DEBUG_LEVEL);

  ASSERT(mps.state_valid(), "Failed debug tests.");
  randomize_state_haar(mps);
  ASSERT(mps.state_valid(), "Failed debug tests.");

  return true;
}

bool test_mpo_sample_paulis_montecarlo() {
  constexpr size_t nqb = 8;

  MatrixProductState mps(nqb, 1u << nqb);
  mps.set_debug_level(MPS_DEBUG_LEVEL);
  Statevector sv(nqb);

  randomize_state_haar(mps, sv);

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

      ASSERT(p1 == p2, fmt::format("Paulis {} and {} do not match.", p1, p2));
      for (size_t j = 0; j < t1.size(); j++) {
        ASSERT(is_close(t1[j], t2[j]), fmt::format("Amplitudes of {} and {} of {:.3f} and {:.3f} on qubits {} do not match.", p1, p2, t1[j], t2[j], to_qubits(supports[i])));
      }
    }

    return true;
  };

  uint32_t s = randi();

  Random::seed_rng(s);
  auto samples1 = mps.sample_paulis_montecarlo(supports, num_samples, 0, p);

  Random::seed_rng(s);
  auto samples2 = sv.sample_paulis_montecarlo(supports, num_samples, 0, p);

  ASSERT(test_samples(samples1, samples2));

  for (size_t i = 0; i < 10; i++) {
    size_t tqb = 4;
    Qubits qubits = random_boundary_qubits(nqb, tqb);
    auto mpo = std::dynamic_pointer_cast<MagicQuantumState>(mps.partial_trace(qubits));
    auto rho = std::dynamic_pointer_cast<MagicQuantumState>(sv.partial_trace(qubits));

    size_t rqb = nqb - tqb;

    supports.clear();
    for (uint32_t k = 0; k < rqb; k++) {
      auto rqubits = random_qubits(rqb, randi() % rqb);
      std::sort(rqubits.begin(), rqubits.end());
      supports.push_back(rqubits);
    }

    Random::seed_rng(s);
    samples1 = mpo->sample_paulis_montecarlo(supports, num_samples, 0, p);

    Random::seed_rng(s);
    samples2 = rho->sample_paulis_montecarlo(supports, num_samples, 0, p);

    ASSERT(test_samples(samples1, samples2));
  }

  return true;
}

// NOTE:
// this test is broken since QuantumState uses calculate_magic_mutual_information_from_samples, and 
// MatrixProductState uses calculate_magic_mutual_information_from_samples2. Need a better test.
bool test_mpo_bipartite_mmi() {
  constexpr size_t nqb = 6;

  uint32_t s = randi();
  Statevector sv(nqb);
  MatrixProductState mps(nqb, 1u << nqb);

  randomize_state_haar(sv, mps);

  size_t num_samples = 10;
  size_t equilibration_timesteps = 100;

  Random::seed_rng(s);
  auto samples1 = sv.bipartite_magic_mutual_information_montecarlo(num_samples, equilibration_timesteps);
  Random::seed_rng(s);
  auto samples2 = mps.bipartite_magic_mutual_information_montecarlo(num_samples, equilibration_timesteps);

  for (size_t i = 0; i < samples1.size(); i++) {
    ASSERT(is_close_eps(1e-4, samples1[i], samples2[i]), fmt::format("Bipartite magic mutual information samples not equal: {:.3f}, {:.3f}", samples1[i], samples2[i]));
  }

  return true;
}

bool test_sample_paulis_exhaustive() {
  constexpr size_t nqb = 6;

  Statevector sv(nqb);

  randomize_state_haar(sv);

  std::vector<uint32_t> qubitsA = {0, 1, 2};
  std::vector<uint32_t> qubitsB = {3, 4, 5};
  double mmi = sv.magic_mutual_information_montecarlo(qubitsA, qubitsB, 100, 100);


  return true;
}

bool test_mps_ising_model() {
  constexpr size_t nqb = 32;

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
  constexpr size_t nqb = 8;

  MatrixProductState mps(nqb, 2, 1e-8);
  mps.set_debug_level(MPS_DEBUG_LEVEL);

  for (size_t i = 0; i < 3; i++) {
    randomize_state_clifford(2, mps);

    for (size_t q = 0; q < nqb; q++) {
      double r = randf();
      if (r < 0.1) {
        mps.measure(Measurement({static_cast<uint32_t>(q)}, PauliString("+Z")));
      }
    }
  }


  for (size_t i = 0; i < nqb - 1; i++) {
    double t = mps.trace();
    ASSERT(is_close(t, 1.0), fmt::format("Trace was not preserved. Trace = {:.3f}", t));
  }

  return true;
}

bool test_mps_trace_conserved() {
  constexpr size_t nqb = 32;

  MatrixProductState mps(nqb, 2, 1e-8);
  mps.set_debug_level(MPS_DEBUG_LEVEL);

  auto apply_gates = [&](auto gate, std::vector<uint32_t> qubits) {
    mps.evolve(gate, qubits);
  };


  auto partial_trace = [&](MatrixProductState& mps, uint32_t q) {
    std::vector<uint32_t> qubits;
    for (size_t i = 0; i < nqb; i++) {
      if (i != q) {
        qubits.push_back(i);
      }
    }
    
    return mps.partial_trace_mps(qubits).trace();
  };

  auto expz = [&](MatrixProductState& mps, uint32_t q) {
    PauliString Z(mps.get_num_qubits());
    Z.set_z(q, 1);
    return std::abs(mps.expectation(Z));
  };


  for (size_t d = 0; d < 10; d++) {
    randomize_state_haar(mps);
    double t = mps.trace();
    ASSERT(is_close(t, 1.0), fmt::format("Trace = {:.5f}\n", t));
  }

  ASSERT(mps.state_valid());
  
  return true;
}

bool test_serialize() {
  MatrixProductState mps(32, 16, 1e-8);
  Statevector sv(8);
  DensityMatrix rho(4);

  for (size_t i = 0; i < 10; i++) {
    randomize_state_haar(mps);
    auto data = mps.serialize();

    MatrixProductState mps_;
    mps_.deserialize(data);

    ASSERT(is_close(mps.inner(mps_), 1.0), "MatrixProductStates were not equal after (de)serialization.");

    randomize_state_haar(sv);
    data = sv.serialize();

    Statevector sv_;
    sv_.deserialize(data);
    ASSERT(is_close(sv.inner(sv_), 1.0), "Statevectors were not equal after (de)serialization.");

    randomize_state_haar(rho);
    data = rho.serialize();

    DensityMatrix rho_;
    rho_.deserialize(data);

    ASSERT(states_close(rho, rho_), "DensityMatrices were not equal after (de)serialization.");
  }
  
  return true;
}

bool test_circuit_measurements() {
  constexpr size_t nqb = 6;

  Statevector psi(nqb);
  DensityMatrix rho(nqb);
  MatrixProductState mps(nqb, 1u << nqb);

  for (size_t i = 0; i < 4; i++) {
    QuantumCircuit qc(nqb);
    randomize_state_haar(qc);

    for (size_t j = 0; j < 3; j++) {
      size_t k = randi() % 2 + 1;
      auto qubits = to_qubits(random_interval(nqb, k));
      PauliString P = PauliString::randh(k);
      qc.add_measurement(Measurement(qubits, P, randi() % 2));
    }

    for (size_t j = 0; j < 3; j++) {
      double beta = randf();
      size_t k = randi() % 2 + 1;
      auto qubits = to_qubits(random_interval(nqb, k));
      PauliString P = PauliString::randh(k);
      qc.add_weak_measurement(WeakMeasurement(qubits, beta, P, randi() % 2));
    }

    qc.apply(psi, rho, mps);

    ASSERT(states_close(psi, rho, mps), fmt::format("States do not match after applying mid-circuit measurements."));
  }

  return true;
}

bool test_forced_measurement() {
  constexpr size_t nqb = 2;

  Statevector psi(nqb);
  DensityMatrix rho(nqb);
  MatrixProductState mps(nqb, 1u << nqb);

  auto test_circuit = [](const QuantumCircuit& qc, MatrixProductState& psi) {
    qc.apply(psi);
  };
  
  QuantumCircuit qc(nqb);
  qc.add_measurement({0}, PauliString("+Z"), true);
  test_circuit(qc, mps);

  qc = QuantumCircuit(nqb);
  qc.add_measurement({0, 1}, PauliString("+ZZ"), true);
  test_circuit(qc, mps);

  return true;
}

bool test_statevector_diagonal_gate() {
  constexpr size_t nqb = 6;

  QuantumCircuit qc(nqb);
  for(size_t i = 0; i < nqb; i++) {
    qc.h(i);
  }

  Statevector psi1(qc);
  Statevector psi2(qc);

  auto randexp = [&]() { return std::exp(-std::complex<double>(0.0, randf() * 2 * M_PI)); };
  for (size_t i = 0; i < 5; i++) {
    size_t k = randi() % (nqb - 1) + 1;
    auto qubits = random_qubits(nqb, k);

    size_t h = 1u << k;
    Eigen::VectorXcd U(h);
    for (size_t j = 0; j < h; j++) {
      U(j) = randexp();
    }

    psi1.evolve_diagonal(U, qubits);
    psi2.evolve(U.asDiagonal(), qubits);

    ASSERT(is_close(std::abs(psi1.inner(psi2)), 1.0), "Statevectors not equal after evolving diagonal gates.");
  }

  return true;
}

bool test_sample_bitstrings() {
  // TODO implement
  
  return true;
}

double kl_divergence(const std::vector<double>& P, const std::vector<double>& Q) {
  if (P.size() != Q.size()) {
    throw std::runtime_error("Cannot compare distributions of difference size.");
  }

  double kl_div = 0.0;
  for (int i = 0; i < P.size(); ++i) {
    if (!is_close(Q[i], 0) && !is_close(P[i], 0)) {
      kl_div += Q[i] * std::log(Q[i] / P[i]);
    }
  }
  
  return kl_div;
}

bool test_mps_sample_bitstrings() {
  constexpr size_t nqb = 8;
  constexpr size_t N = 1u << nqb;

  MatrixProductState mps(nqb, N);
  Statevector psi(nqb);

  randomize_state_haar(mps, psi);

  std::vector<double> P = psi.probabilities();

  constexpr size_t num_samples = 5000;
  auto samples = mps.sample_bitstrings({}, num_samples);

  std::vector<double> Q(N, 0.0);
  for (const auto& [bits, p] : samples) {
    uint32_t z = bits.to_integer();
    double p_ = std::pow(std::abs(psi.data(z)), 2.0);
    ASSERT(is_close(p[0], p_));

    Q[z] += 1.0 / num_samples;
  }

  // Compute KL divergence and check that it is small
  double kl_div = kl_divergence(P, Q);
  ASSERT(kl_div < 0.05);

  return true;
}

bool test_mps_mixed_sample_bitstrings() {
  constexpr size_t nqb = 8;
  constexpr size_t N = 1u << nqb;

  MatrixProductState mps(nqb, N);
  Statevector psi(nqb);

  randomize_state_haar(mps, psi);

  constexpr size_t num_samples = 1000;

  for (size_t i = 0; i < 10; i++) {
    auto qubits = random_boundary_qubits(nqb, 4);
    auto rho = psi.partial_trace(qubits);
    auto mpo = mps.partial_trace(qubits);

    std::vector<double> P = rho->probabilities();

    auto samples = mpo->sample_bitstrings({}, num_samples);

    std::vector<double> Q(N, 0.0);
    for (const auto& [bits, p] : samples) {
      uint32_t z = bits.to_integer();
      double p_ = P[z];
      ASSERT(is_close(p[0], p_));

      Q[z] += 1.0 / num_samples;
    }

    // Compute KL divergence and check that it is small
    double kl_div = kl_divergence(P, Q);
    ASSERT(kl_div < 0.05);
  }

  return true;
}

bool test_marginal_distributions() {
  constexpr size_t nqb = 8;

  Statevector psi(nqb);
  randomize_state_haar(psi);

  for (size_t i = 0; i < 10; i++) {
    std::vector<QubitSupport> supports;

    size_t num_supports = 5;
    for (size_t j = 0; j < num_supports; j++) {
      size_t k = randi(0, nqb);
      auto qubits = random_qubits(nqb, k);
      supports.push_back(qubits);
    }

    auto marginals = psi.marginal_probabilities(supports);
    for (size_t j = 0; j < num_supports; j++) {
      auto qubits = to_qubits(supports[j]);
      std::sort(qubits.begin(), qubits.end());
      auto _qubits = to_qubits(support_complement(qubits, nqb));
      auto rho = psi.partial_trace(_qubits);

      auto prob_s = rho->probabilities();

      for (uint32_t z = 0; z < (1u << nqb); z++) {
        uint32_t zA = quantumstate_utils::reduce_bits(z, qubits);
        ASSERT(is_close(prob_s[zA], marginals[j + 1][z]));
      }
    }
  }

  return true;
}

bool test_bitstring_expectation() {
  constexpr size_t nqb = 6;
  DensityMatrix rho(nqb);
  Statevector psi(nqb);
  MatrixProductState mps(nqb, 1u << nqb);

  randomize_state_haar(rho, psi, mps);

  for (size_t i = 0; i < 100; i++) {
    BitString bits = BitString::random(nqb);
    auto d1 = rho.expectation(bits);
    auto d2 = psi.expectation(bits);
    auto d3 = mps.expectation(bits);

    ASSERT(is_close(d1, d2, d3), fmt::format("Bits {} disagree: {:.6f}, {:.6f}, {:.6f}", bits, d1, d2, d3));
  }

  return true;
}

bool test_configurational_entropy() {
  constexpr size_t nqb = 8;

  MatrixProductState mps(nqb, 1u << nqb);
  Statevector psi(nqb);

  for (size_t i = 0; i < 10; i++) {
    randomize_state_haar(mps, psi);

    size_t t = 4;
    auto qubits = random_boundary_qubits(nqb, t);
    auto mpo = mps.partial_trace(qubits);

    uint32_t k = randi(0, nqb - t);
    Qubits qubitsA(k);
    std::iota(qubitsA.begin(), qubitsA.end(), 0);
    Qubits qubitsB(mpo->get_num_qubits() - k);
    std::iota(qubitsB.begin(), qubitsB.end(), k);
    std::vector<QubitSupport> supports = {qubitsA, qubitsB};

    auto samples = extract_amplitudes(mpo->sample_bitstrings(supports, 500));

    size_t idx = randi(0, 4);

    auto rho = psi.partial_trace(qubits);
    auto s1 = estimate_renyi_entropy(idx, samples[0]);
    auto s2 = renyi_entropy(idx, rho->probabilities());
    ASSERT(is_close_eps(0.05, s1, s2), fmt::format("Configurational entropy does not match. Estimate = {:.5f}, exact = {:.5f}\n", s1, s2));

    auto rhoA = rho->partial_trace(qubitsB);
    auto s1A = estimate_renyi_entropy(idx, samples[1]);
    auto s2A = renyi_entropy(idx, rhoA->probabilities());
    ASSERT(is_close_eps(0.05, s1, s2), fmt::format("Configurational entropy does not match on qubitsA = {}. Estimate = {:.5f}, exact = {:.5f}\n", qubitsA, s1, s2));
    
    auto rhoB = rho->partial_trace(qubitsA);
    auto s1B = estimate_renyi_entropy(idx, samples[2]);
    auto s2B = renyi_entropy(idx, rhoB->probabilities());
    ASSERT(is_close_eps(0.05, s1, s2), fmt::format("Configurational entropy does not match on qubitsB = {}. Estimate = {:.5f}, exact = {:.5f}\n", qubitsB, s1, s2));
  } 

  
  return true;
}

bool test_quantum_state_sampler() {
  constexpr size_t nqb = 8;

  std::shared_ptr<MatrixProductState> mps = std::make_shared<MatrixProductState>(nqb, 1u << nqb);
  randomize_state_haar(*mps.get());

  ExperimentParams params;
  params["sample_configurational_entropy"] = 0;
  params["sample_configurational_entropy_mutual"] = 1;
  params["sample_configurational_entropy_bipartite"] = 1;
  params["num_configurational_entropy_samples"] = 1000,
  params["configurational_entropy_method"] = "virtual";

  QuantumStateSampler sampler(params);

  SampleMap samples;
  sampler.add_samples(samples, mps);
  std::cout << fmt::format("samples = \n{}\n", samples);
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
  ADD_TEST(test_statevector);
  ADD_TEST(test_mps_vs_statevector);
  ADD_TEST(test_mps_expectation);
  ADD_TEST(test_mpo_expectation);
  ADD_TEST(test_partial_trace);
  ADD_TEST(test_clifford_states_unitary);
  ADD_TEST(test_pauli_reduce);
  ADD_TEST(test_z2_clifford);
  ADD_TEST(test_mps_inner);
  //ADD_TEST(test_mps_reverse);
  ADD_TEST(test_statevector_to_mps);
  ADD_TEST(test_mps_measure);  
  ADD_TEST(test_mps_weak_measure);
  ADD_TEST(test_purity);
  ADD_TEST(test_projector);
  ADD_TEST(test_mpo_sample_paulis);
  ADD_TEST(test_mpo_sample_paulis_montecarlo);
  ADD_TEST(test_sample_paulis_exhaustive);
  ADD_TEST(test_pauli);
  ADD_TEST(test_mps_ising_model);
  ADD_TEST(test_mps_random_clifford);
  ADD_TEST(test_mps_trace_conserved);
  ADD_TEST(test_serialize);
  ADD_TEST(test_circuit_measurements);
  ADD_TEST(test_forced_measurement);
  ADD_TEST(test_statevector_diagonal_gate);
  ADD_TEST(test_mps_sample_bitstrings);
  ADD_TEST(test_mps_mixed_sample_bitstrings);
  ADD_TEST(test_marginal_distributions);
  ADD_TEST(test_bitstring_expectation);
  ADD_TEST(test_sample_bitstrings);
  ADD_TEST(test_quantum_state_sampler);
  ADD_TEST(test_configurational_entropy);

  //ADD_TEST(inspect_svd_error);


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
