#include "QuantumStates.h"

std::vector<PauliAmplitude> QuantumState::sample_paulis_exhaustive() {
  if (num_qubits > 15) {
    throw std::runtime_error("Cannot do exhaustive Pauli sampling for n > 15 qubits.");
  }
  size_t N = 1u << (2*num_qubits);
  std::vector<PauliAmplitude> samples(N);

  for (size_t i = 0; i < N; i++) {
    PauliString P = PauliString::from_bitstring(num_qubits, i);
    samples[i] = {P, std::abs(expectation(P))};
  }

  return samples;
}

std::vector<PauliAmplitude> QuantumState::sample_paulis_exact(size_t num_samples, ProbabilityFunc prob) {
  std::vector<PauliAmplitude> ps = sample_paulis_exhaustive();
  size_t s = ps.size();

  std::vector<double> pauli_pdf(s);

  for (size_t i = 0; i < s; i++) {
    pauli_pdf[i] = prob(ps[i].second);
  }

  double d = 0.0;
  for (size_t i = 0; i < s; i++) {
    d += pauli_pdf[i];
  }

  for (size_t i = 0; i < s; i++) {
    pauli_pdf[i] /= d;
  }

  std::discrete_distribution<> dist(pauli_pdf.begin(), pauli_pdf.end()); 

  std::vector<PauliAmplitude> samples;
  for (size_t i = 0; i < num_samples; i++) {
    size_t bitstring = dist(rng);
    samples.push_back(ps[bitstring]);
  }

  return samples;
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
      return t2;
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

