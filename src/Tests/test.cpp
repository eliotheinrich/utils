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
  if (!is_close_eps(eps, first, second)) {
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
  constexpr size_t nqb = 3;

  Statevector psi(nqb);
  psi.x(0);

  for (size_t i = 0; i < nqb; i++) {
    PauliString Z(nqb);
    Z.set_z(i, 1);

    double d = psi.expectation(Z).real();

    if (i == 0) {
      ASSERT(is_close(d, -1.0));
    } else {
      ASSERT(is_close(d, 1.0));
    }
  }

  return true;
}

bool test_measure() {
  constexpr size_t nqb = 6;

  DensityMatrix rho(nqb);
  Statevector psi(nqb);
  MatrixProductState mps(nqb, 1u << nqb);

  randomize_state_haar(rho, psi, mps);

  // To measure
  PauliString P = PauliString::randh(nqb);

  double p = psi.expectation(P).real();
  ASSERT(is_close(p, rho.expectation(P).real(), mps.expectation(P).real()), fmt::format("Expectation of measured Pauli not equal for different state types."));
  double p0 = (1 + p)/2.0;
  double p1 = (1 - p)/2.0;

  Qubits qubits(nqb);
  std::iota(qubits.begin(), qubits.end(), 0);

  rho.measure(Measurement(qubits, P));

  Statevector psi0(psi);
  psi0.measure(Measurement(qubits, P, false));
  Statevector psi1(psi);
  psi1.measure(Measurement(qubits, P, true));

  MatrixProductState mps0(mps);
  mps0.measure(Measurement(qubits, P, false));
  MatrixProductState mps1(mps);
  mps1.measure(Measurement(qubits, P, true));

  for (size_t i = 0; i < 10; i++) {
    PauliString P_exp = PauliString::randh(nqb);
    std::complex<double> c1 = rho.expectation(P_exp);
    std::complex<double> c2 = p0 * psi0.expectation(P_exp) + p1 * psi1.expectation(P_exp);
    std::complex<double> c3 = p0 * mps0.expectation(P_exp) + p1 * mps1.expectation(P_exp);
    ASSERT(is_close(c1, c2, c3), fmt::format("c1 = {:.3f} + {:.3f}i, c2 = {:.3f} + {:.3f}i, c3 = {:.3f} + {:.3f}i are not close for P = {}", c1.real(), c1.imag(), c2.real(), c2.imag(), c3.real(), c3.imag(), P_exp));
  }

  return true;
}

bool test_weak_measure() {
  constexpr size_t nqb = 6;

  DensityMatrix rho(nqb);
  Statevector psi(nqb);
  MatrixProductState mps(nqb, 1u << nqb);

  randomize_state_haar(rho, psi, mps);

  // To measure
  PauliString P = PauliString::randh(nqb);

  double p = psi.expectation(P).real();
  ASSERT(is_close(p, rho.expectation(P).real(), mps.expectation(P).real()), fmt::format("Expectation of measured Pauli not equal for different state types."));
  double beta = randf(0, 2*M_PI);
  double p0 = (1 + std::tanh(2*beta)*p)/2.0;
  double p1 = (1 - std::tanh(2*beta)*p)/2.0;

  Qubits qubits(nqb);
  std::iota(qubits.begin(), qubits.end(), 0);

  rho.weak_measure(WeakMeasurement(qubits, beta, P));

  Statevector psi0(psi);
  psi0.weak_measure(WeakMeasurement(qubits, beta, P, false));
  Statevector psi1(psi);
  psi1.weak_measure(WeakMeasurement(qubits, beta, P, true));

  MatrixProductState mps0(mps);
  mps0.weak_measure(WeakMeasurement(qubits, beta, P, false));
  MatrixProductState mps1(mps);
  mps1.weak_measure(WeakMeasurement(qubits, beta, P, true));

  for (size_t i = 0; i < 10; i++) {
    PauliString P_exp = PauliString::randh(nqb);
    std::complex<double> c1 = rho.expectation(P_exp);
    std::complex<double> c2 = p0 * psi0.expectation(P_exp) + p1 * psi1.expectation(P_exp);
    std::complex<double> c3 = p0 * mps0.expectation(P_exp) + p1 * mps1.expectation(P_exp);
    ASSERT(is_close(c1, c2, c3), fmt::format("c1 = {:.3f} + {:.3f}i, c2 = {:.3f} + {:.3f}i, c3 = {:.3f} + {:.3f}i are not close for P = {}", c1.real(), c1.imag(), c2.real(), c2.imag(), c3.real(), c3.imag(), P_exp));
  }

  return true;
}

bool test_mps_vs_statevector() {
  constexpr size_t nqb = 6;

  Statevector psi(nqb);
  MatrixProductState mps(nqb, 1u << nqb);
  mps.set_debug_level(MPS_DEBUG_LEVEL);

  for (size_t i = 0; i < 100; i++) {
    randomize_state_haar(mps, psi);
    size_t q = randi() % (nqb - 2) + 1;
    Qubits qubits(q);
    std::iota(qubits.begin(), qubits.end(), 0);

    if (randf() < 0.5) {
      qubits = to_qubits(support_complement(qubits, nqb));
    }


    size_t index = randi() % 4 + 1;
    double s1 = psi.entanglement(qubits, index);
    double s2 = mps.entanglement(qubits, index);

    auto e1 = psi.get_entanglement<double>(index);
    auto e2 = mps.get_entanglement<double>(index);


    ASSERT(is_close_eps(1e-4, s1, s2), fmt::format("Entanglement does not match! s1 = {}, s2 = {}", s1, s2));
    ASSERT(states_close(psi, mps), "States do not match!");
  }

  return true;
}

bool test_mps_expectation() {
  constexpr size_t nqb = 6;

  Statevector psi(nqb);
  MatrixProductState mps(nqb, 1u << nqb);
  mps.set_debug_level(MPS_DEBUG_LEVEL);
  randomize_state_haar(mps, psi);
  ASSERT(states_close(psi, mps), "States are not close.");

  auto mat_to_str = [](const Eigen::MatrixXcd& mat) {
    std::stringstream ss;
    ss << mat;
    return ss.str();
  };

  for (size_t i = 0; i < 100; i++) {
    size_t r = randi() % 2;
    size_t k;
    if (r == 0) {
      k = randi() % nqb + 1;
    } else if (r == 1) {
      k = randi() % 4 + 1;
    }

    size_t q = randi() % (nqb - k + 1);
    Qubits qubits(k);
    std::iota(qubits.begin(), qubits.end(), q);

    if (r == 0) {
      MatrixProductState mpo = mps.partial_trace_mps({});

      PauliString P = PauliString::rand(k);
      PauliString Pp = P.superstring(qubits, nqb);

      std::complex<double> d1 = psi.expectation(Pp);
      std::complex<double> d2 = mps.expectation(Pp);
      std::complex<double> d3 = mpo.expectation(Pp);

      ASSERT(is_close(d1, d2, d3), fmt::format("<{}> on {} = {:.3f} + {:.3f}i, {:.3f} + {:.3f}i, {:.3f} + {:.3f}i\n", P, qubits, d1.real(), d1.imag(), d2.real(), d2.imag(), d3.real(), d3.imag()));
    } else if (r == 1) {
      Eigen::MatrixXcd M = haar_unitary(k);

      std::complex<double> d1 = psi.expectation(M, qubits);
      std::complex<double> d2 = mps.expectation(M, qubits);
      ASSERT(is_close(d1, d2), fmt::format("<{}> = {:.3f} + {:.3f}i, {:.3f} + {:.3f}i\n", mat_to_str(M), d1.real(), d1.imag(), d2.real(), d2.imag()));
    }
  }

  ASSERT(mps.state_valid(), "MPS failed debug tests.");

  return true;
}

bool test_mpo_expectation() {
  constexpr size_t nqb = 6;

  Statevector psi(nqb);
  MatrixProductState mps(nqb, 1u << nqb);
  mps.set_debug_level(MPS_DEBUG_LEVEL);
  randomize_state_haar(mps, psi);
  ASSERT(states_close(psi, mps), "States are not close.");

  for (size_t i = 0; i < 100; i++) {
    size_t num_traced_qubits = 3;
    Qubits qubits = random_boundary_qubits(nqb, num_traced_qubits);

    size_t num_remaining_qubits = nqb - num_traced_qubits;

    size_t nqb = randi() % (num_remaining_qubits - 1) + 1;

    size_t q = randi() % (num_remaining_qubits - nqb + 1);
    std::vector<uint32_t> qubits_p(nqb);
    std::iota(qubits_p.begin(), qubits_p.end(), q);

    auto mpo = mps.partial_trace(qubits);
    auto rho = psi.partial_trace(qubits);

    PauliString P = PauliString::rand(nqb);
    PauliString Pp = P.superstring(qubits_p, num_remaining_qubits);

    std::complex<double> d1 = rho->expectation(Pp);
    std::complex<double> d2 = mpo->expectation(Pp);

    ASSERT(is_close(d1, d2), fmt::format("<{}> on {} = {:.3f} + {:.3f}i, {:.3f} + {:.3f}i\n", P, qubits_p, d1.real(), d1.imag(), d2.real(), d2.imag()));
  }

  return true;
}

bool test_clifford_states_unitary() {
  constexpr size_t nqb = 6;
  QuantumCHPState chp(nqb);
  QuantumGraphState graph(nqb);
  Statevector psi(nqb);

  for (size_t i = 0; i < 10; i++) {
    QuantumCircuit qc(nqb);
    qc.append(random_clifford(nqb));

    // TODO include measurements
    //for (size_t j = 0; j < 3; j++) {
    //  size_t q = rng() % nqb;
    //  qc.mzr(q);
    //}

    qc.apply(psi, chp, graph);



    Statevector sv_chp = chp.to_statevector();
    //Statevector sv_graph = graph.to_statevector();

    ASSERT(states_close(psi, sv_chp), fmt::format("Clifford simulators disagree."));
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
  constexpr size_t nqb = 6;

  std::minstd_rand rng(randi());
  for (size_t i = 0; i < 4; i++) {
    QuantumCircuit qc = generate_haar_circuit(nqb, nqb, true); 

    std::vector<uint32_t> qubit_map(nqb);
    std::iota(qubit_map.begin(), qubit_map.end(), 0);
    std::shuffle(qubit_map.begin(), qubit_map.end(), rng);
    qc.apply_qubit_map(qubit_map);

    MatrixProductState mps(nqb, 20);
    mps.set_debug_level(MPS_DEBUG_LEVEL);
    Statevector psi(nqb);

    qc.apply(psi, mps);

    ASSERT(states_close(psi, mps), fmt::format("States not close after nonlocal circuit: \n{}\n{}", psi.to_string(), mps.to_string()));
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
  Statevector psi(nqb);

  for (size_t i = 0; i < 100; i++) {
    randomize_state_haar(mps, psi);

    ASSERT(states_close(psi, mps), fmt::format("States do not agree before measurement.\n"));

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
      bool b2 = psi.measure(Measurement(qubits, P));

      ASSERT(mps.state_valid(), fmt::format("MPS failed debug tests for P = {} on {}.", P, qubits));
      ASSERT(b1 == b2, fmt::format("Different measurement outcomes observed for {}.", P));
      ASSERT(states_close(psi, mps), fmt::format("States don't match after measurement of {} on {}.\n{}\n{}", P, qubits, psi.to_string(), mps.to_string()));
    }
  }

  return true;
}

bool test_mps_weak_measure() {
  constexpr size_t nqb = 6;

  MatrixProductState mps(nqb, 1u << nqb);
  mps.set_debug_level(MPS_DEBUG_LEVEL);
  Statevector psi(nqb);

  for (size_t i = 0; i < 100; i++) {
    randomize_state_haar(mps, psi);

    ASSERT(states_close(psi, mps), fmt::format("States do not agree before measurement.\n"));

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
      bool b2 = psi.weak_measure(WeakMeasurement(qubits, beta, P));

      double d = std::abs(psi.inner(Statevector(mps)));

      ASSERT(b1 == b2, "Different measurement outcomes observed.");
      ASSERT(states_close(psi, mps), fmt::format("States don't match after weak measurement of {} on {}. d = {} \n{}\n{}", P, qubits, d, psi.to_string(), mps.to_string()));
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

bool test_statevector_to_mps() {
  constexpr size_t nqb = 4;

  Statevector psi(nqb);
  MatrixProductState mps1(nqb, 1u << nqb);
  mps1.set_debug_level(MPS_DEBUG_LEVEL);

  for (size_t i = 0; i < 10; i++) {
    randomize_state_haar(psi, mps1);
    MatrixProductState mps2(psi, 1u << nqb);

    for (size_t j = 0; j < 10; j++) {
      PauliString P = PauliString::randh(nqb);
      auto c1 = mps1.expectation(P);
      auto c2 = mps2.expectation(P);
      auto c3 = psi.expectation(P);
      ASSERT(is_close(c1, c2, c3), fmt::format("P = {}, c1 = {:.3f} + {:.3f}i, c2 = {:.3f} + {:.3f}i, c3 = {:.3f} + {:.3f}i", P, c1.real(), c1.imag(), c2.real(), c2.imag(), c3.real(), c3.imag()));
    }


    for (size_t j = 0; j < 10; j++) {
      size_t k = randi(0, nqb);
      Qubits qubits = to_qubits(std::make_pair(0, k));
      int index = randi(1, 4);
      double s1 = mps1.entanglement(qubits, index);
      double s2 = mps2.entanglement(qubits, index);
      double s3 = psi.entanglement(qubits, index);
      ASSERT(is_close(s1, s2, s3), fmt::format("qubits = {}, index = {}, s1 = {:.3f}, s2 = {:.3f}, s3 = {:.3f}\n", qubits, index, s1, s2, s3));
    }

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
  Statevector psi(nqb);

  randomize_state_haar(mps, psi);

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
  auto samples2 = psi.sample_paulis_montecarlo(supports, num_samples, 0, p);

  ASSERT(test_samples(samples1, samples2));

  for (size_t i = 0; i < 10; i++) {
    size_t tqb = 4;
    Qubits qubits = random_boundary_qubits(nqb, tqb);
    auto mpo = std::dynamic_pointer_cast<MagicQuantumState>(mps.partial_trace(qubits));
    auto rho = std::dynamic_pointer_cast<MagicQuantumState>(psi.partial_trace(qubits));

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
  Statevector psi(nqb);
  MatrixProductState mps(nqb, 1u << nqb);

  randomize_state_haar(psi, mps);

  size_t num_samples = 10;
  size_t equilibration_timesteps = 100;

  Random::seed_rng(s);
  auto samples1 = psi.bipartite_magic_mutual_information_montecarlo(num_samples, equilibration_timesteps);
  Random::seed_rng(s);
  auto samples2 = mps.bipartite_magic_mutual_information_montecarlo(num_samples, equilibration_timesteps);

  for (size_t i = 0; i < samples1.size(); i++) {
    ASSERT(is_close_eps(1e-4, samples1[i], samples2[i]), fmt::format("Bipartite magic mutual information samples not equal: {:.3f}, {:.3f}", samples1[i], samples2[i]));
  }

  return true;
}

bool test_stabilizer_entropy_sampling() {
  constexpr size_t nqb = 6;

  std::shared_ptr<Statevector> psi = std::make_shared<Statevector>(nqb);
  std::shared_ptr<MatrixProductState> mps = std::make_shared<MatrixProductState>(nqb, 1u << nqb);

  ExperimentParams params;
  params["sample_stabilizer_entropy"] = 1;
  params["sample_stabilizer_entropy_mutual"] = 1;
  params["sample_stabilizer_entropy_bipartite"] = 1;
  params["stabilizer_entropy_mutual_subsystem_size"] = static_cast<int>(nqb/2);
  params["stabilizer_entropy_indices"] = "1,2,3";
  params["sre_num_samples"] = 10000;

  ExperimentParams params1 = params;
  params1["sre_method"] = "exhaustive";
  GenericMagicSampler sampler1(params1);

  ExperimentParams params2 = params;
  params2["sre_method"] = "exact";
  GenericMagicSampler sampler2(params2);

  ExperimentParams params3 = params;
  MPSMagicSampler sampler3(params3);

  randomize_state_haar(*psi.get(), *mps.get());

  SampleMap samples1;
  sampler1.add_samples(samples1, psi);

  SampleMap samples2;
  sampler2.add_samples(samples2, psi);

  SampleMap samples3;
  sampler3.add_samples(samples3, mps);

  for (int i = 1; i <= 3; i++) {
    double M1 = samples1[STABILIZER_ENTROPY(i)][0][0];
    double M2 = samples2[STABILIZER_ENTROPY(i)][0][0];
    double M3 = samples3[STABILIZER_ENTROPY(i)][0][0];

    ASSERT(is_close_eps(0.1, M1, M2, M3), fmt::format("SRE does not match: {:.5f}, {:.5f}, {:.5f}\n", M1, M2, M3));
  }

  double L1 = samples1[STABILIZER_ENTROPY_MUTUAL][0][0];
  double L2 = samples2[STABILIZER_ENTROPY_MUTUAL][0][0];
  double L3 = samples3[STABILIZER_ENTROPY_MUTUAL][0][0];

  ASSERT(is_close_eps(0.1, L1, L2, L3), fmt::format("SRE mutual information does not match: {:.5f}, {:.5f}, {:.5f}\n", L1, L2, L3));

  std::vector<double> L1b = samples1[STABILIZER_ENTROPY_BIPARTITE][0];
  std::vector<double> L2b = samples2[STABILIZER_ENTROPY_BIPARTITE][0];
  std::vector<double> L3b = samples3[STABILIZER_ENTROPY_BIPARTITE][0];
  for (size_t i = 0; i < nqb/2 - 1; i++) {
    ASSERT(is_close_eps(0.1, L1b[i], L2b[i], L3b[i]), fmt::format("SRE mutual bipartite information does not match: {::.5f}, {::.5f}, {::.5f}\n", L1b, L2b, L3b));
  }

  return true;
}

bool test_participation_entropy_sampling() {
  constexpr size_t nqb = 6;

  std::shared_ptr<Statevector> psi = std::make_shared<Statevector>(nqb);
  std::shared_ptr<MatrixProductState> mps = std::make_shared<MatrixProductState>(nqb, 1u << nqb);

  ExperimentParams params;
  params["sample_participation_entropy"] = 1;
  params["sample_participation_entropy_mutual"] = 1;
  params["sample_participation_entropy_bipartite"] = 1;
  params["participation_entropy_num_samples"] = 10000;

  ExperimentParams params1 = params;
  params1["participation_entropy_method"] = "exhaustive";
  GenericParticipationSampler sampler1(params1);

  ExperimentParams params2 = params;
  params2["participation_entropy_method"] = "sampled";
  GenericParticipationSampler sampler2(params2);

  ExperimentParams params3 = params;
  MPSParticipationSampler sampler3(params3);

  for (size_t k = 0; k < 5; k++) {
    randomize_state_haar(*psi.get(), *mps.get());

    SampleMap samples1;
    sampler1.add_samples(samples1, psi);

    SampleMap samples2;
    sampler2.add_samples(samples2, psi);

    SampleMap samples3;
    sampler3.add_samples(samples3, mps);

    double W1 = samples1[PARTICIPATION_ENTROPY][0][0];
    double W2 = samples2[PARTICIPATION_ENTROPY][0][0];
    double W3 = samples3[PARTICIPATION_ENTROPY][0][0];

    ASSERT(is_close_eps(0.1, W1, W2, W3), fmt::format("PE does not match: {:.5f}, {:.5f}, {:.5f}\n", W1, W2, W3));

    double L1 = samples1[PARTICIPATION_ENTROPY_MUTUAL][0][0];
    double L2 = samples2[PARTICIPATION_ENTROPY_MUTUAL][0][0];
    double L3 = samples3[PARTICIPATION_ENTROPY_MUTUAL][0][0];

    ASSERT(is_close_eps(0.1, L1, L2, L3), fmt::format("PE mutual information does not match: {:.5f}, {:.5f}, {:.5f}\n", L1, L2, L3));

    std::vector<double> L1b = samples1[PARTICIPATION_ENTROPY_BIPARTITE][0];
    std::vector<double> L2b = samples2[PARTICIPATION_ENTROPY_BIPARTITE][0];
    std::vector<double> L3b = samples3[PARTICIPATION_ENTROPY_BIPARTITE][0];
    for (size_t i = 0; i < nqb/2 - 1; i++) {
      ASSERT(is_close_eps(0.1, L1b[i], L2b[i], L3b[i]), fmt::format("PE mutual bipartite information does not match: {::.5f}, {::.5f}, {::.5f}\n", L1b, L2b, L3b));
    }
  }

  return true;
}

bool test_mps_ising_model() {
  constexpr size_t nqb = 32;

  MatrixProductState mps = MatrixProductState::ising_ground_state(nqb, 0.0, 64, 1e-8, 100);
  PauliString P(nqb);
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

bool test_mps_conjugate() {
  constexpr size_t nqb = 2;
  MatrixProductState mps(nqb, 2);

  for (size_t i = 0; i < 10; i++) {
    randomize_state_haar(mps);

    auto mps_conj = mps;
    mps_conj.conjugate();

    Statevector s1(mps);
    Statevector s2(mps_conj);

    for (size_t s = 0; s < (1u << nqb); s++) {
      ASSERT(is_close(s1.data(s), std::conj(s2.data(s))));
    }
  }

  return true;
}

bool test_mps_concatenate() {
  for (size_t i = 0; i < 100; i++) {
    uint32_t nqb1 = randi(1, 10);
    uint32_t nqb2 = randi(1, 10);

    MatrixProductState mps1(nqb1, 32);
    randomize_state_haar(mps1);
    MatrixProductState mps2(nqb2, 32);
    randomize_state_haar(mps2);

    MatrixProductState mps = mps1.concatenate(mps2);

    for (uint32_t q = 0; q < nqb1; q++) {
      Eigen::Matrix2cd M = Eigen::Matrix2cd::Random();
      auto c1 = mps1.expectation(M, {q});
      auto c2 = mps.expectation(M, {q});

      ASSERT(is_close(c1, c2));
    }

    for (uint32_t q = 0; q < nqb2; q++) {
      Eigen::Matrix2cd M = Eigen::Matrix2cd::Random();
      auto c1 = mps2.expectation(M, {q});
      auto c2 = mps.expectation(M, {nqb1 + q});

      ASSERT(is_close(c1, c2));
    }
  }

  return true;
}

bool test_mps_many_qubit_gate() {
  constexpr size_t nqb = 8;

  MatrixProductState mps(nqb, 1u << nqb);
  Statevector psi(nqb);


  for (size_t i = 0; i < 10; i++) {
    size_t n = randi(1, 5);
    Qubits qubits = random_qubits(nqb, n);
    qubits = random_qubits(nqb, n);
    //std::sort(qubits.begin(), qubits.end());
    Eigen::MatrixXcd U = haar_unitary(n);

    mps.evolve(U, qubits);
    psi.evolve(U, qubits);

    for (size_t j = 0; j < 10; j++) {
      PauliString P = PauliString::randh(nqb);
      auto c1 = mps.expectation(P);
      auto c2 = psi.expectation(P);
      ASSERT(is_close(c1, c2), fmt::format("P = {}, c1 = {:.3f} + {:.3f}i, c2 = {:.3f} + {:.3f}i", P, c1.real(), c1.imag(), c2.real(), c2.imag()));
    }


    for (size_t j = 0; j < 10; j++) {
      size_t k = randi(0, nqb);
      Qubits qubits = to_qubits(std::make_pair(0, k));
      int index = randi(1, 4);
      double s1 = mps.entanglement(qubits, index);
      double s2 = psi.entanglement(qubits, index);
      ASSERT(is_close(s1, s2), fmt::format("qubits = {}, index = {}, s1 = {:.3f}, s2 = {:.3f}", qubits, index, s1, s2));
    }

    ASSERT(states_close(mps, psi), "States not equal after applying multi-qubit gate.");
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
  Statevector psi(8);
  DensityMatrix rho(4);

  for (size_t i = 0; i < 10; i++) {
    randomize_state_haar(mps);
    auto data = mps.serialize();

    MatrixProductState mps_;
    mps_.deserialize(data);

    ASSERT(is_close(mps.inner(mps_), 1.0), "MatrixProductStates were not equal after (de)serialization.");

    randomize_state_haar(psi);
    data = psi.serialize();

    Statevector psi_;
    psi_.deserialize(data);
    ASSERT(is_close(psi.inner(psi_), 1.0), "Statevectors were not equal after (de)serialization.");

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

template <class StateType>
bool apply_circuit_check_error(const QuantumCircuit& qc, StateType& state, bool should_error) {
  bool found_error = false;
  try {
    qc.apply(state);
  } catch (const std::runtime_error& error) {
    found_error = true;
  }

  return found_error == should_error;
}

bool test_forced_measurement() {
  constexpr size_t nqb = 4;

  Statevector psi(nqb);
  DensityMatrix rho(nqb);
  MatrixProductState mps(nqb, 1u << nqb);

  for (size_t i = 0; i < 10; i++) {
    randomize_state_haar(psi, rho, mps);

    uint32_t k = randi(1, 3);
    PauliString P = PauliString::randh(k);
    Qubits qubits = random_qubits(nqb, k);
    bool outcome = randi() % 2;

    QuantumCircuit qc(nqb);
    qc.add_measurement(qubits, P, outcome);
    ASSERT(apply_circuit_check_error(qc, psi, false));
    ASSERT(apply_circuit_check_error(qc, rho, false));
    ASSERT(apply_circuit_check_error(qc, mps, false));

    qc = QuantumCircuit(nqb);
    qc.add_measurement(qubits, P, !outcome);
    ASSERT(apply_circuit_check_error(qc, psi, true));
    ASSERT(apply_circuit_check_error(qc, rho, true));
    ASSERT(apply_circuit_check_error(qc, mps, true));
  }

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

    std::vector<double> Q(1u << rho->get_num_qubits(), 0.0);
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

bool test_sv_entanglement() {
  constexpr size_t nqb = 8;

  Statevector psi(nqb); 

  for (size_t i = 0; i < 5; i++) {
    randomize_state_haar(psi);
    DensityMatrix rho(psi);

    for (size_t j = 0; j < 10; j++) {
      size_t index = randi(0, 4) + 1;

      Qubits qubits;
      int r = randi();
      size_t k = randi(0, nqb);
      if (r % 3 == 0) {
        qubits = Qubits(k);
        std::iota(qubits.begin(), qubits.end(), 0);
      } else if (r % 3 == 1) {
        qubits = Qubits(k);
        std::iota(qubits.begin(), qubits.end(), 0);
        qubits = to_qubits(support_complement(qubits, nqb));
      } else {
        qubits = random_qubits(nqb, k);
      }

      std::vector<double> s1 = psi.get_entanglement(index);
      std::vector<double> s2 = rho.get_entanglement(index);

      for (size_t i = 0; i < s1.size(); i++) {
        ASSERT(is_close_eps(1e-4, s1[i], s2[i]), fmt::format("Entropies do not match: {::.5f}, {::.5f}\n", s1, s2));
      }
    }
  }

  return true;
}


// Functions for generating matchgates
static constexpr std::complex<double> imag_unit = {0.0, 1.0};

auto T(double theta, uint32_t q, size_t nqb) {
  double sin = std::sin(theta);
  double cos = std::cos(theta);
  Eigen::Matrix4cd Tm;
  Tm << 1.0, 0.0,           0.0,           0.0,
        0.0, cos,           imag_unit*sin, 0.0,
        0.0, imag_unit*sin, cos,           0.0,
        0.0, 0.0,           0.0,           1.0;

  FreeFermionHamiltonian H(nqb);
  H.add_term(q+1, q, theta);
  return std::make_pair(Tm, H);
}


auto G(double theta, uint32_t q, size_t nqb) {
  double sin = std::sin(theta);
  double cos = std::cos(theta);
  Eigen::Matrix4cd Gm;
  Gm << cos,           0.0, 0.0, imag_unit*sin,
        0.0,           1.0, 0.0, 0.0,
        0.0,           0.0, 1.0, 0.0,
        imag_unit*sin, 0.0, 0.0, cos;
  FreeFermionHamiltonian H(nqb);
  H.add_nonconserving_term(q+1, q, theta);
  return std::make_pair(Gm, H);
}

auto R(double theta, uint32_t q, size_t nqb) {
  Eigen::Matrix2cd Rm;
  Rm << 1.0, 0.0,
        0.0, std::exp(imag_unit*theta);
  FreeFermionHamiltonian H(nqb);
  H.add_term(q, q, theta);
  return std::make_pair(Rm, H);
};

void normalize(std::vector<double>& p) {
  double N = 0.0;
  for (size_t i = 0; i < p.size(); i++) {
    N += p[i];
  }

  for (size_t i = 0; i < p.size(); i++) {
    p[i] /= N;
  }
}

bool test_free_fermion_state() {
  size_t nqb = 8;

  Qubits sites = random_qubits(nqb, nqb/2);
  Statevector psi(nqb);
  for (auto q : sites) {
    psi.x(q);
  }
  GaussianState fermion_state(nqb, sites);
  MajoranaState majorana_state(nqb, sites);

  std::vector<double> op_dist = {0.3, 0.3, 0.3, 0.1};
  normalize(op_dist);
  std::mt19937 gen(randi());
  std::discrete_distribution<> dist(op_dist.begin(), op_dist.end());

  for (size_t k = 0; k < 20; k++) {
    int gate_type = dist(gen);
    double theta = randf(0, 2 * M_PI);

    if (gate_type == 0) {
      uint32_t q = randi(0, nqb);
      auto [Rm, H] = R(theta, q, nqb);
      psi.evolve(Rm, q);
      fermion_state.evolve_hamiltonian(H);
      majorana_state.evolve_hamiltonian(H);
    } else if (gate_type == 1) {
      uint32_t q = randi(0, nqb - 1);
      auto [Tm, H] = T(theta, q, nqb);
      psi.evolve(Tm, {q, q+1});
      fermion_state.evolve_hamiltonian(H);
      majorana_state.evolve_hamiltonian(H);
    } else if (gate_type == 2) {
      uint32_t q = randi(0, nqb - 1);
      auto [Gm, H] = G(theta, q, nqb);
      psi.evolve(Gm, {q, q+1});
      fermion_state.evolve_hamiltonian(H);
      majorana_state.evolve_hamiltonian(H);
    } else {
      uint32_t q = randi(0, nqb);
      bool outcome = psi.mzr(q);
      fermion_state.forced_projective_measurement(q, outcome);
      majorana_state.forced_projective_measurement(q, outcome);
    }

    // Check state equality
    std::vector<double> c1;
    std::vector<double> c2;
    std::vector<double> c3;
    for (uint32_t i = 0; i < nqb; i++) {
      PauliString Z(nqb);
      Z.set_z(i, 1);
      c1.push_back((1.0 - psi.expectation(Z).real()) / 2.0);
      c2.push_back(fermion_state.occupation(i));
      c3.push_back(majorana_state.occupation(i));
    }

    uint32_t index = randi(2, 3);
    std::vector<double> s1 = psi.get_entanglement(index);
    std::vector<double> s2 = fermion_state.get_entanglement(index);
    std::vector<double> s3 = majorana_state.get_entanglement(index);
    //std::cout << fmt::format("s = \n{::.5f}\n{::.5f}\n{::.5f}\n", s1, s2, s3);

    for (size_t i = 0; i < nqb; i++) {
      ASSERT(is_close(s1[i], s2[i], s3[i]), fmt::format("Entanglement {} at {} is not equal: \n{::.5f}\n{::.5f}\n{::.5f}", index, i, s1, s2, s3));
      ASSERT(is_close(c1[i], c2[i], c3[i]), fmt::format("Occupations at {} are not equal: \n{::.5f}\n{::.5f}\n{::.5f}", i, c1, c2, c3));
    }
  }


  return true;
}

bool test_extended_majorana_state() {
  size_t nqb = 4;

  Qubits sites = random_qubits(nqb, nqb/2);
  Statevector psi(nqb);
  for (auto q : sites) {
    psi.x(q);
  }
  ExtendedMajoranaState majorana_state(nqb, sites);

  std::vector<double> op_dist = {0.2, 0.2, 0.2, 0.0, 0.1};
  normalize(op_dist);
  std::mt19937 gen(randi());
  std::discrete_distribution<> dist(op_dist.begin(), op_dist.end());

  for (size_t k = 0; k < 20; k++) {
    int gate_type = dist(gen);
    double theta = randf(0, 2 * M_PI);

    if (gate_type == 0) {
      uint32_t q = randi(0, nqb);
      auto [Rm, H] = R(theta, q, nqb);
      psi.evolve(Rm, q);
      majorana_state.evolve_hamiltonian(H);
    } else if (gate_type == 1) {
      uint32_t q = randi(0, nqb - 1);
      auto [Tm, H] = T(theta, q, nqb);
      psi.evolve(Tm, {q, q+1});
      majorana_state.evolve_hamiltonian(H);
    } else if (gate_type == 2) {
      uint32_t q = randi(0, nqb - 1);
      auto [Gm, H] = G(theta, q, nqb);
      psi.evolve(Gm, {q, q+1});
      majorana_state.evolve_hamiltonian(H);
    } else if (gate_type == 3) {
      uint32_t q = randi(0, nqb);
      bool outcome = psi.mzr(q);
      majorana_state.forced_projective_measurement(q, outcome);
    } else {
      uint32_t q = randi(0, nqb - 1);
      std::cout << fmt::format("Doing interaction at q = {}\n", q);
      std::cout << psi.to_string() << "\n";
      psi.measure({{q, q+1}, PauliString("+ZZ"), true});
      majorana_state.interaction(q);
    }

    std::vector<double> c1;
    std::vector<double> c2;
    for (uint32_t i = 0; i < nqb; i++) {
      PauliString Z(nqb);
      Z.set_z(i, 1);
      c1.push_back((1.0 - psi.expectation(Z).real()) / 2.0);
      c2.push_back(majorana_state.occupation(i));
    }

    for (size_t i = 0; i < nqb; i++) {
      ASSERT(is_close(c1[i], c2[i]), fmt::format("Occupations at {} are not equal: \n{::.5f}\n{::.5f}", i, c1, c2));
    }
  }

  std::cout << majorana_state.to_string() << "\n";

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
  ADD_TEST(test_measure);
  ADD_TEST(test_weak_measure);
  ADD_TEST(test_mps_vs_statevector);
  ADD_TEST(test_mps_expectation);
  ADD_TEST(test_mpo_expectation);
  ADD_TEST(test_partial_trace);
  ADD_TEST(test_clifford_states_unitary);
  ADD_TEST(test_pauli_reduce);
  ADD_TEST(test_z2_clifford);
  ADD_TEST(test_mps_inner);
  ADD_TEST(test_statevector_to_mps);
  ADD_TEST(test_mps_measure);  
  ADD_TEST(test_mps_weak_measure);
  ADD_TEST(test_purity);
  ADD_TEST(test_projector);
  ADD_TEST(test_mpo_sample_paulis);
  ADD_TEST(test_mpo_sample_paulis_montecarlo);
  ADD_TEST(test_stabilizer_entropy_sampling);
  ADD_TEST(test_participation_entropy_sampling);
  ADD_TEST(test_pauli);
  ADD_TEST(test_mps_ising_model);
  ADD_TEST(test_mps_random_clifford);
  ADD_TEST(test_mps_conjugate);
  ADD_TEST(test_mps_concatenate);
  ADD_TEST(test_mps_many_qubit_gate);
  ADD_TEST(test_mps_trace_conserved);
  ADD_TEST(test_serialize);
  ADD_TEST(test_circuit_measurements);
  ADD_TEST(test_forced_measurement);
  ADD_TEST(test_statevector_diagonal_gate);
  ADD_TEST(test_mps_sample_bitstrings);
  ADD_TEST(test_mps_mixed_sample_bitstrings);
  ADD_TEST(test_marginal_distributions);
  ADD_TEST(test_bitstring_expectation);
  ADD_TEST(test_sv_entanglement);
  ADD_TEST(test_free_fermion_state);
  //ADD_TEST(test_extended_majorana_state);



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
