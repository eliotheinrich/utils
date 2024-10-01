#include "QuantumStates.h"

std::vector<PauliAmplitude> QuantumState::stabilizer_renyi_entropy_exhaustive() {
  if (num_qubits > 31) {
    throw std::runtime_error("Cannot do exhaustive calculation of renyi entropy for n > 31 qubits.");
  }
  size_t N = 1u << (2*num_qubits);
  std::vector<PauliAmplitude> samples(N);

  for (size_t i = 0; i < N; i++) {
    PauliString P = PauliString::from_bitstring(num_qubits, i);
    samples[i] = {P, std::abs(expectation(P))};
  }

  return samples;
}

void single_qubit_random_mutation(PauliString& p, std::minstd_rand& rng, size_t j) {
  size_t g = rng() % 4;

  bool b1 = g % 2;
  bool b2 = g < 2;

  p.set_x(j, b1);
  p.set_z(j, b2);
}

void single_qubit_random_mutation(PauliString& p, std::minstd_rand& rng) {
  size_t j = rng() % p.num_qubits;
  single_qubit_random_mutation(p, rng, j);
}

void random_mutation(PauliString& p, std::minstd_rand& rng) {
  if (rng() % 2 || p.num_qubits == 1) {
    // Do single-qubit mutation
    size_t j = rng() % p.num_qubits;
    single_qubit_random_mutation(p, rng, j);
  } else {
    // Do double-qubit mutation
    size_t j1 = rng() % p.num_qubits;
    size_t j2;
    while (j2 == j1) {
      j2 = rng() % p.num_qubits;
    }

    single_qubit_random_mutation(p, rng, j1);
    single_qubit_random_mutation(p, rng, j2);
  }
}

void global_random_mutation(PauliString& p, std::minstd_rand& rng) {
  p = PauliString::rand(p.num_qubits, rng);
}

void random_bit_mutation(PauliString& p, std::minstd_rand& rng) {
  size_t j = rng() % (2*p.num_qubits);
  p.set(j, !p.get(j));
}

std::vector<PauliAmplitude> QuantumState::stabilizer_renyi_entropy_montecarlo(size_t num_samples) {
  PauliString P1 = PauliString::rand(num_qubits, rng);

  std::vector<PauliAmplitude> samples;
  for (size_t i = 0; i < num_samples; i++) {
    double t1 = std::abs(expectation(P1));
    double p1 = t1*t1/basis;

    PauliString P2 = P1.copy();
    global_random_mutation(P2, rng);

    double t2 = std::abs(expectation(P2));
    double p2 = t2*t2/basis;

    if (randf() < std::min(1.0, p2/p1)) {
      P1 = P2.copy();
    }

    samples.push_back({P1, t1});
  }

  return samples;
}
