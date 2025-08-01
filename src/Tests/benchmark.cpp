#include "QuantumState.h"
#include "CliffordState.h"
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
      (states.evolve(random, qubits), ...);
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
void randomize_state_clifford(QuantumStates&... states) {
  size_t num_qubits = get_num_qubits(states...);

  QuantumCircuit qc(num_qubits);
  size_t depth = num_qubits;

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

QubitInterval random_interval(size_t num_qubits, size_t k) {
  uint32_t q1 = randi() % (num_qubits - k + 1);
  uint32_t q2 = q1 + k;

  return std::make_pair(q1, q2);
}

void benchmark_mps_evolution() {
  constexpr size_t nqb = 64;

  MatrixProductState mps(nqb, 64);
  size_t depth = 64;

  for (size_t i = 0; i < depth; i++) {
    randomize_state_haar(mps);
  }
}

void benchmark_magic_mutual_information_montecarlo() {
  constexpr size_t nqb = 16;

  MatrixProductState mps(nqb, 112);
  size_t depth = 10;

  for (size_t i = 0; i < depth; i++) {
    randomize_state_haar(mps);
  }

  std::vector<uint32_t> qubitsA = {0, 1, 2};
  std::vector<uint32_t> qubitsB = {13, 14, 15};
  double mmi = mps.magic_mutual_information_montecarlo(qubitsA, qubitsB, 100, 100);
}

void benchmark_stabilizer_renyi_entropy() {
  constexpr size_t nqb = 16;

  MatrixProductState mps(nqb, 32);
  size_t depth = 10;

  for (size_t i = 0; i < depth; i++) {
    randomize_state_haar(mps);
  }

  size_t num_samples = 100;
  auto samples = mps.sample_paulis({}, num_samples);
  auto amplitudes = extract_amplitudes(samples);
  auto s = estimate_renyi_entropy(2, amplitudes[0]);
}

void benchmark_stabilizer_renyi_entropy_montecarlo() {
  constexpr size_t nqb = 16;

  MatrixProductState mps(nqb, 32);
  size_t depth = 10;

  for (size_t i = 0; i < depth; i++) {
    randomize_state_haar(mps);
  }

  size_t num_samples = 100;
  auto samples = mps.sample_paulis_montecarlo({}, num_samples, 100, [](double t) { return t*t; });
  auto amplitudes = extract_amplitudes(samples);
  auto s = estimate_renyi_entropy(2, amplitudes[0]);
}

void benchmark_clifford_circuit() {
  constexpr uint32_t nqb = 256;

  QuantumCHPState chp(nqb);

  for (size_t i = 0; i < nqb; i++) {
    for (uint32_t q = 0; q < nqb/2; q++) {
      Qubits qubits;
      if (i % 2) {
        qubits = {2*q, 2*q + 1};
      } else {
        qubits = {2*q + 1, (2*q + 2) % nqb};
      }
      
      chp.random_clifford(qubits);
      double entanglement = chp.entanglement(qubits, 2);
    }
  }
}

void benchmark_clifford_circuit_sv() {
  constexpr uint32_t nqb = 18;

  Statevector psi(nqb);

  for (size_t i = 0; i < nqb; i++) {
    for (uint32_t q = 0; q < nqb/2; q++) {
      Qubits qubits;
      if (i % 2) {
        qubits = {2*q, 2*q + 1};
      } else {
        qubits = {2*q + 1, (2*q + 2) % nqb};
      }
      
      psi.random_clifford(qubits);
    }
  }
}

void benchmark_clifford_circuit_mps() {
  constexpr uint32_t nqb = 18;

  MatrixProductState mps(nqb, 32);

  for (size_t i = 0; i < nqb; i++) {
    for (uint32_t q = 0; q < nqb/2; q++) {
      Qubits qubits;
      if (i % 2) {
        qubits = {2*q, 2*q + 1};
      } else {
        qubits = {2*q + 1, (2*q + 2) % nqb};
      }
      
      mps.random_clifford(qubits);
    }
  }
}


#define DO_BENCHMARK(x)                                                           \
if (run_all || benchmarks.contains(#x)) {                                         \
  auto start = std::chrono::high_resolution_clock::now();                         \
  x();                                                                            \
  auto stop = std::chrono::high_resolution_clock::now();                          \
  int duration = duration_cast<std::chrono::microseconds>(stop - start).count();  \
  benchmarks_results[#x "()"] = duration;                                         \
}                                                                                 \

int main(int argc, char *argv[]) {
  std::set<std::string> benchmarks;
  std::map<std::string, int> benchmarks_results;

  bool run_all = (argc == 1);

  if (!run_all) {
    for (size_t i = 1; i < argc; i++) {
      benchmarks.insert(argv[i]);
    }
  }

  DO_BENCHMARK(benchmark_mps_evolution);
  DO_BENCHMARK(benchmark_magic_mutual_information_montecarlo);
  DO_BENCHMARK(benchmark_stabilizer_renyi_entropy);
  DO_BENCHMARK(benchmark_stabilizer_renyi_entropy_montecarlo);
  DO_BENCHMARK(benchmark_clifford_circuit);
  DO_BENCHMARK(benchmark_clifford_circuit_sv);
  DO_BENCHMARK(benchmark_clifford_circuit_mps);

  if (benchmarks_results.size() == 0) {
    std::cout << "No benchmarks to run.\n";
  } else {
    double total_duration = 0.0;
    for (const auto& [name, duration] : benchmarks_results) {
      std::cout << fmt::format("{:>35}: {:.2f} seconds\n", name, duration/1e6);
      total_duration += duration;
    }

    std::cout << fmt::format("Total duration: {:.2f} seconds\n", total_duration/1e6);
  }
}
