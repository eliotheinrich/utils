#include "QuantumStates.h"
#include <stdexcept>

std::vector<BitAmplitudes> QuantumState::sample_bitstrings(size_t num_samples) const {
  std::vector<double> probs = probabilities();

  std::minstd_rand rng(randi());
  std::discrete_distribution<> dist(probs.begin(), probs.end());

  std::vector<BitAmplitudes> samples;
  for (size_t i = 0; i < num_samples; i++) {
    uint32_t z = dist(rng);
    samples.push_back({BitString::from_bits(num_qubits, z), probs[z]});
  }

  return samples;
}

std::vector<std::vector<BitAmplitudes>> QuantumState::sample_bitstrings_bipartite(size_t num_samples) const {
  auto supports = get_bipartite_supports(num_qubits);
  size_t N = num_qubits/2 - 1;

  std::vector<std::vector<BitAmplitudes>> samples(N);

  for (size_t i = 0; i < supports.size(); i++) {
    auto _qubits = complement(to_qubits(supports[i]), num_qubits);
    auto mixed_state = partial_trace(_qubits);
    samples[i] = sample_bitstrings(num_samples);
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

QubitSupport support_complement(const QubitSupport& support, size_t n) {
  auto interval = support_range(support);
  if (interval) {
    auto [q1, q2] = interval.value();
    if (q1 < 0 || q2 > n) {
      throw std::runtime_error(fmt::format("Support on [{}, {}) cannot be complemented on {} qubits.", q1, q2, n));
    }
  }

  std::vector<bool> mask(n, true);
  auto qubits = to_qubits(support);
  for (const auto q : qubits) {
    mask[q] = false;
  }

  Qubits qubits_;
  for (size_t i = 0; i < n; i++) {
    if (mask[i]) {
      qubits_.push_back(i);
    }
  }

  return QubitSupport{qubits_};
}

double estimate_renyi_entropy(size_t index, const std::vector<double>& samples) {
  if (index == 1) {
    double q = 0.0;
    for (auto p : samples) {
      q += std::log(p);
    }

    q = q/samples.size();
    return -q;
  } else {
    double q = 0.0;
    for (auto p : samples) {
      q += std::pow(p, index - 1.0);
    }

    q = q/samples.size();
    return 1.0/(1.0 - index) * std::log(q);
  }
}

double renyi_entropy(size_t index, const std::vector<double>& probs) {
  double s = 0.0;
  if (index == 1) {
    for (auto p : probs) {
      if (p > 1e-6) {
        s += p * std::log(p);
      }
    }
    return -s;
  } else {
    for (auto p : probs) {
      s += std::pow(p, index);
    }
    return std::log(s)/(1.0 - index);
  }
}
