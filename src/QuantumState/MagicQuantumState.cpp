#include "QuantumStates.h"
#include <memory>

// Returns a 3-tuple where the elements are
// 1.) Qubits not in AB (i.e. qubits to be traced over to yield AB system)
// 2.) Qubits not in A, renumbered within the AB subsystem. That is, stateB = state.partial_trace(_qubits).partial_trace(_qubitsA)
// 3.) " for qubits not B.
std::tuple<Qubits, Qubits, Qubits> get_traced_qubits(
  const Qubits& qubitsA, const Qubits& qubitsB, size_t num_qubits
) {
  std::vector<bool> mask(num_qubits, false);

  for (const auto q : qubitsA) {
    mask[q] = true;
  }

  for (const auto q : qubitsB) {
    mask[q] = true;
  }

  // Trace out qubits not in A or B
  Qubits _qubits;
  for (size_t i = 0; i < num_qubits; i++) {
    if (!mask[i]) {
      _qubits.push_back(i);
    }
  }

  std::vector<size_t> offset(num_qubits);
  size_t k = 0;
  for (size_t i = 0; i < num_qubits; i++) {
    if (!mask[i]) {
      k++;
    }
    
    offset[i] = k;
  }

  Qubits _qubitsA(qubitsA.begin(), qubitsA.end());
  for (size_t i = 0; i < qubitsA.size(); i++) {
    _qubitsA[i] -= offset[_qubitsA[i]];
  }

  Qubits _qubitsB(qubitsB.begin(), qubitsB.end());
  for (size_t i = 0; i < qubitsB.size(); i++) {
    _qubitsB[i] -= offset[_qubitsB[i]];
  }

  return {_qubits, _qubitsA, _qubitsB};
}

std::vector<QubitSupport> get_bipartite_supports(size_t num_qubits) {
  size_t N = num_qubits/2 - 1;
  std::vector<QubitSupport> supports(2*N);
  for (size_t q = 0; q < N; q++) {
    supports[q] = std::make_pair(0, q + 1);
    supports[q + N] = std::make_pair(q + 1, num_qubits);
  }

  return supports;
}

QubitSupport combine_supports(const QubitSupport& s1, const QubitSupport& s2) {
  // TODO check for overlap
  Qubits qubits1 = to_qubits(s1);
  Qubits qubits2 = to_qubits(s2);
  Qubits qubits(qubits1);
  qubits.insert(qubits.end(), qubits2.begin(), qubits2.end());
  return QubitSupport{qubits};
}

std::vector<PauliAmplitudes> MagicQuantumState::sample_paulis_exhaustive(const std::vector<QubitSupport>& supports) {
  if (num_qubits > 15) {
    throw std::runtime_error("Cannot do exhaustive Pauli sampling for n > 15 qubits.");
  }

  size_t N = 1u << (2*num_qubits);
  std::vector<PauliAmplitudes> samples(N);
  for (size_t i = 0; i < N; i++) {
    PauliString p = PauliString::from_bitstring(num_qubits, i);

    double t = std::abs(expectation(p));
    std::vector<double> amplitudes{t};

    for (const auto& support : supports) {
      Qubits qubits = to_qubits(support);
      auto c = expectation(p.substring(qubits, false));
      amplitudes.push_back(std::abs(c));
    }

    samples[i] = {p, amplitudes};
  }

  return samples;
}

std::vector<PauliAmplitudes> MagicQuantumState::sample_paulis_exact(const std::vector<QubitSupport>& supports, size_t num_samples, ProbabilityFunc prob) {
  std::vector<PauliAmplitudes> ps = sample_paulis_exhaustive(supports);
  size_t s = ps.size();

  std::vector<double> pauli_pdf(s);

  for (size_t i = 0; i < s; i++) {
    pauli_pdf[i] = prob(ps[i].second[0]);
  }

  double d = 0.0;
  for (size_t i = 0; i < s; i++) {
    d += pauli_pdf[i];
  }

  for (size_t i = 0; i < s; i++) {
    pauli_pdf[i] /= d;
  }

  std::discrete_distribution<> dist(pauli_pdf.begin(), pauli_pdf.end()); 
  std::minstd_rand rng(randi());

  std::vector<PauliAmplitudes> samples;
  for (size_t i = 0; i < num_samples; i++) {
    size_t bitstring = dist(rng);
    samples.push_back(ps[bitstring]);
  }

  return samples;
}

std::vector<PauliAmplitudes> MagicQuantumState::sample_paulis_montecarlo(const std::vector<QubitSupport>& supports, size_t num_samples, size_t equilibration_timesteps, ProbabilityFunc prob, std::optional<PauliMutationFunc> mutation_opt) {
  PauliMutationFunc mutation = single_qubit_random_mutation;
  if (mutation_opt) {
    mutation = mutation_opt.value();
  }

  auto perform_mutation = [this, &prob, &mutation](PauliString& p) {
    double t1 = std::abs(expectation(p));
    double p1 = prob(t1);

    PauliString q(p);
    mutation(q);

    double t2 = std::abs(expectation(q));
    double p2 = prob(t2);

    double r = randf();
    if (r < p2 / p1) {
      p = PauliString(q);
      return t2;
    } else {
      return t1;
    }
  };

  PauliString p(num_qubits);

  for (size_t i = 0; i < equilibration_timesteps; i++) {
    double t = perform_mutation(p);
  }

  std::vector<PauliAmplitudes> samples(num_samples);
  for (size_t i = 0; i < num_samples; i++) {
    double t = perform_mutation(p);
    std::vector<double> amplitudes{t};

    for (const auto& support : supports) {
      Qubits qubits = to_qubits(support);
      auto c = expectation(p.substring(qubits, false));
      amplitudes.push_back(std::abs(c));
    }

    samples[i] = {p, amplitudes};
  }

  return samples;
}

double MagicQuantumState::magic_mutual_information_exhaustive(const Qubits& qubitsA, const Qubits& qubitsB) {
  auto [_qubits, _qubitsA, _qubitsB] = get_traced_qubits(qubitsA, qubitsB, num_qubits);

  auto stateAB = std::dynamic_pointer_cast<MagicQuantumState>(partial_trace(_qubits));
  auto stateA = std::dynamic_pointer_cast<MagicQuantumState>(stateAB->partial_trace(_qubitsA));
  auto stateB = std::dynamic_pointer_cast<MagicQuantumState>(stateAB->partial_trace(_qubitsB));

  auto samplesA = stateA->sample_paulis_exhaustive({});
  auto samplesB = stateB->sample_paulis_exhaustive({});
  auto samplesAB = stateAB->sample_paulis_exhaustive({});

  auto power = [](double s, const PauliAmplitudes& p, double pow) {
    return s + std::pow(p.second[0], pow);
  };

  auto power_vec = [&power](const std::vector<PauliAmplitudes>& samples, double pow) {
    auto powfunc = std::bind(power, std::placeholders::_1, std::placeholders::_2, pow);
    return std::accumulate(samples.begin(), samples.end(), 0.0, powfunc);
  };

  double sumA_2 = power_vec(samplesA, 2.0);
  double sumA_4 = power_vec(samplesA, 4.0);
  double sumB_2 = power_vec(samplesB, 2.0);
  double sumB_4 = power_vec(samplesB, 4.0);
  double sumAB_2 = power_vec(samplesAB, 2.0);
  double sumAB_4 = power_vec(samplesAB, 4.0);

  double I = -std::log(sumA_2*sumB_2/sumAB_2);
  double W = -std::log(sumA_4*sumB_4/sumAB_4);

  return I - W;
}

std::vector<double> MagicQuantumState::bipartite_magic_mutual_information_exhaustive() {
  std::vector<double> magic(num_qubits/2 - 1);
  for (size_t i = 0; i < magic.size(); i++) {
    size_t j = i + 1;
    Qubits qubitsA(j);
    std::iota(qubitsA.begin(), qubitsA.end(), 0);

    Qubits qubitsB(num_qubits - j);
    std::iota(qubitsB.begin(), qubitsB.end(), j);

    magic[i] = magic_mutual_information_exhaustive(qubitsA, qubitsB);
  }

  return magic;
}

double MagicQuantumState::calculate_magic_mutual_information_from_samples(const MutualMagicAmplitudes& samples2, const MutualMagicAmplitudes& samples4) {
  const auto [tAB2, tA2, tB2] = unfold_mutual_magic_amplitudes(samples2);
  const auto [tAB4, tA4, tB4] = unfold_mutual_magic_amplitudes(samples4);
  if (tA2.size() != tB2.size() || tB2.size() != tAB2.size()) {
    throw std::invalid_argument(fmt::format("Invalid sample sizes passed to calculate_magic_mutual_information_from_samples. tA2.size() = {}, tB2.size() = {}, tAB2.size() = {}", tA2.size(), tB2.size(), tAB2.size()));
  }

  if (tA4.size() != tB4.size() || tB4.size() != tAB4.size()) {
    throw std::invalid_argument(fmt::format("Invalid sample sizes passed to calculate_magic_mutual_information_from_samples. tA4.size() = {}, tB4.size() = {}, tAB4.size() = {}", tA4.size(), tB4.size(), tAB4.size()));
  }

  size_t num_samples2 = tA2.size();
  size_t num_samples4 = tA4.size();

  double I = 0.0;
  for (size_t i = 0; i < num_samples2; i++) {
    I += std::pow(tA2[i] * tB2[i] / tAB2[i], 2.0);
  }
  I = -std::log(I / num_samples2);

  double W = 0.0;
  for (size_t i = 0; i < num_samples4; i++) {
    W += std::pow(tA4[i] * tB4[i] / tAB4[i], 4.0);
  }
  W = -std::log(W / num_samples4);

  return I - W;
}

MutualMagicData MagicQuantumState::magic_mutual_information_samples_montecarlo(
  const Qubits& qubitsA, const Qubits& qubitsB,
  size_t num_samples, size_t equilibration_timesteps, 
  std::optional<PauliMutationFunc> mutation_opt
) {
  PauliMutationFunc mutation = single_qubit_random_mutation;
  if (mutation_opt) {
    mutation = mutation_opt.value();
  }

  auto [_qubits, _qubitsA, _qubitsB] = get_traced_qubits(qubitsA, qubitsB, num_qubits);
  std::vector<QubitSupport> supports = {_qubitsA, _qubitsB};
  auto stateAB = std::dynamic_pointer_cast<MagicQuantumState>(partial_trace(_qubits));

  ProbabilityFunc p2 = [](double t) -> double { return std::pow(t, 2.0); };
  ProbabilityFunc p4 = [](double t) -> double { return std::pow(t, 4.0); };
  auto samples2 = stateAB->sample_paulis_montecarlo(supports, num_samples, equilibration_timesteps, p2, mutation);
  auto samples4 = stateAB->sample_paulis_montecarlo(supports, num_samples, equilibration_timesteps, p4, mutation);
  
  return {extract_amplitudes(samples2), extract_amplitudes(samples4)};
}

double MagicQuantumState::magic_mutual_information_montecarlo(
  const Qubits& qubitsA, const Qubits& qubitsB,
  size_t num_samples, size_t equilibration_timesteps, 
  std::optional<PauliMutationFunc> mutation_opt
) {
  auto [samples2, samples4] = magic_mutual_information_samples_montecarlo(qubitsA, qubitsB, num_samples, equilibration_timesteps, mutation_opt);
  return MagicQuantumState::calculate_magic_mutual_information_from_samples(samples2, samples4);
}

MutualMagicData MagicQuantumState::magic_mutual_information_samples_exact(
  const Qubits& qubitsA, const Qubits& qubitsB, size_t num_samples
) {
  auto [_qubits, _qubitsA, _qubitsB] = get_traced_qubits(qubitsA, qubitsB, num_qubits);
  std::vector<QubitSupport> supports = {_qubitsA, _qubitsB};

  auto stateAB = std::dynamic_pointer_cast<MagicQuantumState>(partial_trace(_qubits));

  ProbabilityFunc p1 = [](double t) -> double { return std::pow(t, 2.0); };
  ProbabilityFunc p2 = [](double t) -> double { return std::pow(t, 4.0); };
  auto samples1 = stateAB->sample_paulis_exact(supports, num_samples, p1);
  auto samples2 = stateAB->sample_paulis_exact(supports, num_samples, p2);

  return {extract_amplitudes(samples1), extract_amplitudes(samples2)};
}

double MagicQuantumState::magic_mutual_information_exact(
  const Qubits& qubitsA, const Qubits& qubitsB, size_t num_samples
) {
  auto [samples2, samples4] = magic_mutual_information_samples_exact(qubitsA, qubitsB, num_samples);
  return MagicQuantumState::calculate_magic_mutual_information_from_samples(samples2, samples4);
}

std::vector<MutualMagicData> MagicQuantumState::bipartite_magic_mutual_information_samples_montecarlo(
  size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt
) {
  PauliMutationFunc mutation = single_qubit_random_mutation;
  if (mutation_opt) {
    mutation = mutation_opt.value();
  }

  size_t N = num_qubits/2 - 1;

  std::vector<QubitSupport> supports = get_bipartite_supports(num_qubits);
  
  auto get_samples = [&](ProbabilityFunc p) {
    auto pauli_samples = sample_paulis_montecarlo(supports, num_samples, equilibration_timesteps, p, mutation);

    std::vector<MutualMagicAmplitudes> samples(N, std::vector<std::vector<double>>(3));

    for (size_t j = 0; j < num_samples; j++) {
      auto const [P, t] = pauli_samples[j];

      for (size_t i = 0; i < N; i++) {
        samples[i][0].push_back(t[0]);
        samples[i][1].push_back(t[i + 1]);
        samples[i][2].push_back(t[i + N + 1]);
      }
    }

    return samples;
  };

  ProbabilityFunc p2 = [](double t) -> double { return std::pow(t, 2.0); };
  auto t2 = get_samples(p2);
  ProbabilityFunc p4 = [](double t) -> double { return std::pow(t, 4.0); };
  auto t4 = get_samples(p4);

  std::vector<MutualMagicData> samples;
  for (size_t n = 0; n < num_qubits/2 - 1; n++) {
    samples.push_back({t2[n], t4[n]});
  }

  return samples;
}

std::vector<double> MagicQuantumState::bipartite_magic_mutual_information_montecarlo(
  size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt
) {
  auto samples = bipartite_magic_mutual_information_samples_montecarlo(num_samples, equilibration_timesteps, mutation_opt);
  std::vector<double> data(samples.size());
  std::transform(samples.begin(), samples.end(), data.begin(), 
    [](const MutualMagicData& s) { return MagicQuantumState::calculate_magic_mutual_information_from_samples(s); }
  );
  return data;
}

std::vector<MutualMagicData> MagicQuantumState::bipartite_magic_mutual_information_samples_exact(size_t num_samples) {
  size_t N = num_qubits/2 - 1;

  std::vector<QubitSupport> supports = get_bipartite_supports(num_qubits);

  auto get_samples = [&](ProbabilityFunc p) {
    auto pauli_samples = sample_paulis_exact(supports, num_samples, p);

    std::vector<MutualMagicAmplitudes> samples(N, std::vector<std::vector<double>>(3));

    for (size_t j = 0; j < num_samples; j++) {
      auto const [P, t] = pauli_samples[j];

      for (size_t i = 0; i < N; i++) {
        samples[i][0].push_back(t[0]);
        samples[i][1].push_back(t[i + 1]);
        samples[i][2].push_back(t[i + N + 1]);
      }
    }

    return samples;
  };

  ProbabilityFunc p1 = [](double t) -> double { return std::pow(t, 2.0); };
  auto t2 = get_samples(p1);
  ProbabilityFunc p2 = [](double t) -> double { return std::pow(t, 4.0); };
  auto t4 = get_samples(p2);

  std::vector<MutualMagicData> samples;
  for (size_t n = 0; n < num_qubits/2 - 1; n++) {
    samples.push_back({t2[n], t4[n]});
  }

  return samples;
}

std::vector<double> MagicQuantumState::bipartite_magic_mutual_information_exact(size_t num_samples) {
  auto samples = bipartite_magic_mutual_information_samples_exact(num_samples);
  std::vector<double> data(samples.size());
  std::transform(samples.begin(), samples.end(), data.begin(), 
    [](const MutualMagicData& s) { return MagicQuantumState::calculate_magic_mutual_information_from_samples(s); }
  );
  return data;
}
