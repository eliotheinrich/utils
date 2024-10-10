#include "QuantumStates.h"

std::vector<PauliAmplitude> QuantumState::sample_paulis_exhaustive() {
  if (num_qubits > 15) {
    throw std::runtime_error("Cannot do exhaustive calculation of renyi entropy for n > 15 qubits.");
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
  
  if (g == 0) {
    p.set_x(j, 0);
    p.set_z(j, 0);
  } else if (g == 1) {
    p.set_x(j, 1);
    p.set_z(j, 0);
  } else if (g == 2) {
    p.set_x(j, 0);
    p.set_z(j, 1);
  } else {
    p.set_x(j, 1);
    p.set_z(j, 1);
  }
}

void single_qubit_random_mutation(PauliString& p, std::minstd_rand& rng) {
  size_t j = rng() % p.num_qubits;
  single_qubit_random_mutation(p, rng, j);
}

void random_mutation(PauliString& p, std::minstd_rand& rng) {
  bool r = rng() % 2;
  if ((r) || (p.num_qubits == 1)) {
    // Do single-qubit mutation
    size_t j = rng() % p.num_qubits;
    single_qubit_random_mutation(p, rng, j);
  } else {
    // Do double-qubit mutation
    size_t j1 = rng() % p.num_qubits;
    size_t j2 = rng() % p.num_qubits;
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

std::vector<PauliAmplitude> QuantumState::sample_paulis_montecarlo(size_t num_samples, size_t equilibration_timesteps, std::function<double(double)>& prob) {
  PauliString P1 = PauliString::rand(num_qubits, rng);

  std::vector<PauliAmplitude> samples;

  for (size_t i = 0; i < equilibration_timesteps; i++) {
    double t1 = std::abs(expectation(P1));
    double p1 = prob(t1);

    PauliString P2 = P1.copy();
    random_mutation(P2, rng);

    double t2 = std::abs(expectation(P2));
    double p2 = prob(t2);

    if (randf() < std::min(1.0, p2/p1)) {
      P1 = P2.copy();
    }
  }

  double a = 0.0;
  for (size_t i = 0; i < num_samples; i++) {
    double t1 = std::abs(expectation(P1));
    double p1 = prob(t1);

    PauliString P2 = P1.copy();
    random_mutation(P2, rng);

    double t2 = std::abs(expectation(P2));
    double p2 = prob(t2);

    double r = randf();
    if (r < std::min(1.0, p2/p1)) {
      a += 1.0;
      P1 = P2.copy();
      if (p1 > p2 && p2 < 0.01) {
        //std::cout << fmt::format("Decreased from t = ({}, {}) to p = ({}, {}). r = {}\n", t1, t2, p1, p2, r);
      }
    }

    samples.push_back({P1, t1});
  }

  //std::cout << fmt::format("Accepted: {}\n", a/num_samples);

  return samples;
}

