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

void xxz_random_mutation(PauliString& p, std::minstd_rand& rng) {
  PauliString pnew(p);
  if ((rng() % 2) || (p.num_qubits == 1)) {
    // Do single-qubit mutation
    size_t j = rng() % p.num_qubits;
    PauliString Zj = PauliString(p.num_qubits);
    Zj.set_z(j, 1); 

    pnew *= Zj;
  } else {
    // Do double-qubit mutation
    size_t j1 = rng() % p.num_qubits;
    size_t j2 = rng() % p.num_qubits;
    while (j2 == j1) {
      j2 = rng() % p.num_qubits;
    }

    PauliString Xij = PauliString(p.num_qubits);
    Xij.set_x(j1, 1); 
    Xij.set_x(j2, 1); 
    pnew *= Xij;
  }

  p = pnew;

  //std::string s = pnew.to_string_ops();
  //if (std::count(s.begin(), s.end(), 'Y') % 2 == 0) {
  //  // New state respects TRS; accept it
  //  p = pnew;
  //} else {
  //  // New state does not respect TRS; try again
  //  xxz_random_mutation(p, rng);
  //}
}

void global_random_mutation(PauliString& p, std::minstd_rand& rng) {
  p = PauliString::rand(p.num_qubits, rng);
}

void random_bit_mutation(PauliString& p, std::minstd_rand& rng) {
  size_t j = rng() % (2*p.num_qubits);
  p.set(j, !p.get(j));
}

std::vector<PauliAmplitude> QuantumState::sample_paulis_montecarlo(size_t num_samples, size_t equilibration_timesteps, ProbabilityFunc prob, std::optional<PauliMutationFunc> mutation_opt) {
  PauliMutationFunc mutation = xxz_random_mutation;
  if (mutation_opt) {
    mutation = mutation_opt.value();
  }

  auto perform_mutation = [this, &prob, &mutation](PauliString& p) {
    double t1 = std::abs(expectation(p));
    double p1 = prob(t1);

    PauliString q = p.copy();
    mutation(q, rng);

    double t2 = std::abs(expectation(q));
    double p2 = prob(t2);

    if (randf() < p2 / p1) {
      p = q.copy();
    }

    return t1;
  };

  PauliString P(num_qubits);

  for (size_t i = 0; i < equilibration_timesteps; i++) {
    double t = perform_mutation(P);
  }

  std::vector<PauliAmplitude> samples;
  for (size_t i = 0; i < num_samples; i++) {
    double t = perform_mutation(P);

    samples.push_back({P, t});
  }

  return samples;
}

