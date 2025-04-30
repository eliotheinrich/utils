#include "QuantumStates.h"

std::vector<std::vector<double>> QuantumState::marginal_probabilities(const std::vector<QubitSupport>& supports) const {
  std::vector<double> probs = probabilities();

  size_t num_supports = supports.size();

  std::vector<std::vector<double>> marginals(num_supports + 1, std::vector<double>(basis, 0.0));
  marginals[0] = probs;

  for (size_t i = 0; i < num_supports; i++) {
    auto qubits = to_qubits(supports[i]);
    std::sort(qubits.begin(), qubits.end());

    std::vector<double> marginal(1u << qubits.size());
    for (uint32_t z = 0; z < basis; z++) {
      uint32_t zA = quantumstate_utils::reduce_bits(z, qubits);
      marginal[zA] += probs[z];
    }

    for (uint32_t z = 0; z < basis; z++) {
      uint32_t zA = quantumstate_utils::reduce_bits(z, qubits);
      marginals[i + 1][z] = marginal[zA];
    }
  }

  return marginals;
}

std::vector<BitAmplitudes> QuantumState::sample_bitstrings(const std::vector<QubitSupport>& supports, size_t num_samples) const {
  auto marginals = marginal_probabilities(supports);
  auto probs = marginals[0];

  std::minstd_rand rng(randi());
  std::discrete_distribution<> dist(probs.begin(), probs.end());

  std::vector<BitAmplitudes> samples;

  for (size_t i = 0; i < num_samples; i++) {
    uint32_t z = dist(rng);
    BitString bits = BitString::from_bits(num_qubits, z);

    std::vector<double> amplitudes = {probs[z]};
    for (size_t j = 1; j < marginals.size(); j++) {
      amplitudes.push_back(marginals[j][z]);
    }

    samples.push_back({bits, amplitudes});
  }

  return samples;
}

void single_qubit_random_mutation(PauliString& p) {
  size_t j = randi() % p.num_qubits;
  size_t g = randi() % 4;

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

void QuantumState::validate_qubits(const Qubits& qubits) const {
  for (const auto q : qubits) {
    if (q > num_qubits) {
      throw std::runtime_error(fmt::format("Instruction called on qubit {}, which is not valid for a state with {} qubits.", q, num_qubits));
    }
  }
}

static inline double log(double x, double base) {
  return std::log(x) / std::log(base);
}

double renyi_entropy(size_t index, const std::vector<double>& probs, double base) {
  double s = 0.0;
  if (index == 1) {
    for (auto p : probs) {
      if (p > 1e-6) {
        s += p * log(p, base);
      }
    }
    return -s;
  } else {
    for (auto p : probs) {
      s += std::pow(p, index);
    }
    return log(s, base)/(1.0 - index);
  }
}

double estimate_renyi_entropy(size_t index, const std::vector<double>& samples, double base) {
  if (index == 1) {
    double q = 0.0;
    for (auto p : samples) {
      q += log(p, base);
    }

    q = q/samples.size();
    return -q;
  } else {
    double q = 0.0;
    for (auto p : samples) {
      q += std::pow(p, index - 1.0);
    }

    q = q/samples.size();
    return 1.0/(1.0 - index) * log(q, base);
  }
}

double estimate_mutual_renyi_entropy(const std::vector<double>& samplesAB, const std::vector<double>& samplesA, const std::vector<double>& samplesB, double base) {
  size_t N = samplesAB.size();

  double p = 0.0;
  for (size_t i = 0; i < N; i++) {
    p += log(samplesAB[i] / (samplesA[i] * samplesB[i]), base);
  }

  return p/N;
}
