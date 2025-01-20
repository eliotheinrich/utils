#include "QuantumStates.h"
#include <stdexcept>

std::vector<double> QuantumState::pauli_expectation_left_sweep(const PauliString& P, uint32_t q1_, uint32_t q2_) const {
  uint32_t q1 = std::min(q1_, q2_);
  uint32_t q2 = std::max(q1_, q2_);
  std::vector<double> expectations;
  
  std::vector<uint32_t> sites(q1);
  std::iota(sites.begin(), sites.end(), 0);
  for (uint32_t i = q1; i < q2; i++) {
    sites.push_back(i);
    double c = expectation(P.substring(sites));
    expectations.push_back(c);
  }

  return expectations;
}

std::vector<double> QuantumState::pauli_expectation_right_sweep(const PauliString& P, uint32_t q1_, uint32_t q2_) const {
  uint32_t q1 = std::min(q1_, q2_);
  uint32_t q2 = std::max(q1_, q2_);
  std::vector<double> expectations;
  
  std::vector<uint32_t> sites(num_qubits - q2 - 1);
  std::iota(sites.begin(), sites.end(), q2 + 1);
  std::reverse(sites.begin(), sites.end());
  for (uint32_t i = q2; i > q1; i--) {
    sites.push_back(i);
    double c = expectation(P.substring(sites));
    expectations.push_back(c);
  }

  return expectations;
}

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

static void single_qubit_random_mutation_at_site(PauliString& p, std::minstd_rand& rng, size_t j) {
  size_t g = rng() % 4;

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
  single_qubit_random_mutation_at_site(p, rng, j);
}

static void random_mutation(PauliString& p, std::minstd_rand& rng) {
  bool r = rng() % 2;
  if ((r) || (p.num_qubits == 1)) {
    // Do single-qubit mutation
    size_t j = rng() % p.num_qubits;
    single_qubit_random_mutation_at_site(p, rng, j);
  } else {
    // Do double-qubit mutation
    size_t j1 = rng() % p.num_qubits;
    size_t j2 = rng() % p.num_qubits;
    while (j2 == j1) {
      j2 = rng() % p.num_qubits;
    }

    single_qubit_random_mutation_at_site(p, rng, j1);
    single_qubit_random_mutation_at_site(p, rng, j2);
  }
}

static void xxz_random_mutation(PauliString& p, std::minstd_rand& rng) {
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
}

static void global_random_mutation(PauliString& p, std::minstd_rand& rng) {
  p = PauliString::rand(p.num_qubits, rng);
}

static void random_bit_mutation(PauliString& p, std::minstd_rand& rng) {
  size_t j = rng() % (2*p.num_qubits);
  p.set(j, !p.get(j));
}

std::vector<PauliAmplitude> QuantumState::sample_paulis_montecarlo(size_t num_samples, size_t equilibration_timesteps, ProbabilityFunc prob, std::optional<PauliMutationFunc> mutation_opt) {
  PauliMutationFunc mutation = single_qubit_random_mutation;
  if (mutation_opt) {
    mutation = mutation_opt.value();
  }

  auto perform_mutation = [this, &prob, &mutation](PauliString& p) {
    double t1 = std::abs(expectation(p));
    double p1 = prob(t1);

    PauliString q(p);
    mutation(q, rng);

    double t2 = std::abs(expectation(q));
    double p2 = prob(t2);

    if (randf() < p2 / p1) {
      p = PauliString(q);
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

double QuantumState::stabilizer_renyi_entropy(size_t index, const std::vector<PauliAmplitude>& samples) const {
  std::vector<double> amplitude_samples;
  for (const auto &[_, p] : samples) {
    amplitude_samples.push_back(p);
  }

  return stabilizer_renyi_entropy(index, amplitude_samples);
}

// amplitude sampled according to p = Tr(ρP)^2/(2^N * Tr(ρ^2))
double QuantumState::stabilizer_renyi_entropy(size_t index, const std::vector<double>& amplitude_samples) const {
  if (index == 1) {
    double q = 0.0;
    for (size_t i = 0; i < amplitude_samples.size(); i++) {
      double p = amplitude_samples[i]*amplitude_samples[i];
      q += std::log(p);
    }

    q = q/amplitude_samples.size();
    return -q;
  } else {
    double q = 0.0;
    for (size_t i = 0; i < amplitude_samples.size(); i++) {
      double p = amplitude_samples[i]*amplitude_samples[i];
      q += std::pow(p, index - 1.0);
    }

    q = q/amplitude_samples.size();
    return 1.0/(1.0 - index) * std::log(q);
  }
}

// Returns a 3-tuple where the elements are
// 1.) Qubits not in AB (i.e. qubits to be traced over to yield AB system)
// 2.) Qubits not in A, renumbered within the AB subsystem. That is, stateB = state.partial_trace(_qubits).partial_trace(_qubitsA)
// 3.) " for qubits not B.
std::tuple<std::vector<uint32_t>, std::vector<uint32_t>, std::vector<uint32_t>> get_traced_qubits(
  const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_qubits
) {
  std::vector<bool> mask(num_qubits, false);

  for (const auto q : qubitsA) {
    mask[q] = true;
  }

  for (const auto q : qubitsB) {
    mask[q] = true;
  }

  // Trace out qubits not in A or B
  std::vector<uint32_t> _qubits;
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

  std::vector<uint32_t> _qubitsA(qubitsA.begin(), qubitsA.end());
  for (size_t i = 0; i < qubitsA.size(); i++) {
    _qubitsA[i] -= offset[_qubitsA[i]];
  }

  std::vector<uint32_t> _qubitsB(qubitsB.begin(), qubitsB.end());
  for (size_t i = 0; i < qubitsB.size(); i++) {
    _qubitsB[i] -= offset[_qubitsB[i]];
  }

  return {_qubits, _qubitsA, _qubitsB};
}

double QuantumState::magic_mutual_information_exhaustive(const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB) {
  auto [_qubits, _qubitsA, _qubitsB] = get_traced_qubits(qubitsA, qubitsB, num_qubits);

  std::vector<bool> maskA(num_qubits, true);
  for (auto q : qubitsA) {
    maskA[q] = false;
  }

  std::vector<bool> maskB(num_qubits, true);
  for (auto q : qubitsB) {
    maskB[q] = false;
  }

  auto stateAB = partial_trace(_qubits);
  auto stateA = stateAB->partial_trace(_qubitsB);
  auto stateB = stateAB->partial_trace(_qubitsA);

  auto samplesA = stateA->sample_paulis_exhaustive();
  auto samplesB = stateB->sample_paulis_exhaustive();
  auto samplesAB = stateAB->sample_paulis_exhaustive();

  auto power = [](double s, const PauliAmplitude& p, double pow) {
    return s + std::pow(p.second, pow);
  };

  auto power_vec = [&power](const std::vector<PauliAmplitude>& samples, double pow) {
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

std::vector<double> QuantumState::bipartite_magic_mutual_information_exhaustive() {
  std::vector<double> magic(num_qubits/2 - 1);
  for (size_t i = 0; i < magic.size(); i++) {
    size_t j = i + 1;
    std::vector<uint32_t> qubitsA(j);
    std::iota(qubitsA.begin(), qubitsA.end(), 0);

    std::vector<uint32_t> qubitsB(num_qubits - j);
    std::iota(qubitsB.begin(), qubitsB.end(), j);

    magic[i] = magic_mutual_information_exhaustive(qubitsA, qubitsB);
  }

  return magic;
}

static double QuantumState::calculate_magic_mutual_information_from_chi_samples(const MonteCarloSamples& samples) {
  const auto [tA, tB, tAB] = samples;
  if (tA.size() != tB.size() || tB.size() != tAB.size()) {
    throw std::invalid_argument(fmt::format("Invalid sample sizes passed to calculate_magic_from_chi_samples. tA.size() = {}, tB.size() = {}, tAB.size() = {}", tA.size(), tB.size(), tAB.size()));
  }

  size_t num_samples = tA.size();

  double I = 0.0;
  for (size_t i = 0; i < num_samples; i++) {
    I += std::pow(tA[i] * tB[i] / tAB[i], 2.0);
  }
  I = -std::log(I/num_samples);

  double W = 0.0;
  for (size_t i = 0; i < num_samples; i++) {
    W += std::pow(tA[i] * tB[i], 4.0) / std::pow(tAB[i], 2.0);
  }
  W = -std::log(W/num_samples);

  double q = 0.0;
  for (size_t i = 0; i < num_samples; i++) {
    q += std::pow(tAB[i], 2.0);
  }
  double M = -std::log(q / num_samples);

  return I - W + M;
}


static double QuantumState::calculate_magic_mutual_information_from_samples(const MMIMonteCarloSamples& samples) {
  const auto [t2, t4] = samples;
  const auto [tA2, tB2, tAB2] = t2;
  const auto [tA4, tB4, tAB4] = t4;
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
  I = -std::log(I/num_samples2);

  double W = 0.0;
  for (size_t i = 0; i < num_samples4; i++) {
    W += std::pow(tA4[i] * tB4[i] / tAB4[i], 4.0);
  }
  W = -std::log(W/num_samples4);

  return I - W;
}

MMIMonteCarloSamples process_magic_mutual_information_samples(
  std::shared_ptr<QuantumState> stateAB, 
  const std::vector<uint32_t>& _qubitsA, const std::vector<uint32_t>& _qubitsB, 
  const std::vector<PauliAmplitude>& samples1, const std::vector<PauliAmplitude>& samples2
) {
  auto stateA = stateAB->partial_trace(_qubitsB);
  auto stateB = stateAB->partial_trace(_qubitsA);

  auto extract_amplitudes = [&](const std::vector<PauliAmplitude>& samples) -> MonteCarloSamples {
    std::vector<double> tA;
    std::vector<double> tB;
    std::vector<double> tAB;
    for (const auto &[P, t] : samples) {
      PauliString PA = P.substring(_qubitsA, true);
      PauliString PB = P.substring(_qubitsB, true);

      tA.push_back(std::abs(stateA->expectation(PA)));
      tB.push_back(std::abs(stateB->expectation(PB)));
      tAB.push_back(std::abs(t));
    }

    return {tA, tB, tAB};
  };

  return {extract_amplitudes(samples1), extract_amplitudes(samples2)};
}

MMIMonteCarloSamples QuantumState::magic_mutual_information_samples_montecarlo(
  const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB,
  size_t num_samples, size_t equilibration_timesteps, 
  std::optional<PauliMutationFunc> mutation_opt
) {
  PauliMutationFunc mutation = single_qubit_random_mutation;
  if (mutation_opt) {
    mutation = mutation_opt.value();
  }

  auto [_qubits, _qubitsA, _qubitsB] = get_traced_qubits(qubitsA, qubitsB, num_qubits);

  auto stateAB = partial_trace(_qubits);

  ProbabilityFunc p1 = [](double t) -> double { return std::pow(t, 2.0); };
  ProbabilityFunc p2 = [](double t) -> double { return std::pow(t, 4.0); };
  auto samples1 = stateAB->sample_paulis_montecarlo(num_samples, equilibration_timesteps, p1, mutation);
  auto samples2 = stateAB->sample_paulis_montecarlo(num_samples, equilibration_timesteps, p2, mutation);
  
  return process_magic_mutual_information_samples(stateAB, _qubitsA, _qubitsB, samples1, samples2);
}

MMIMonteCarloSamples QuantumState::magic_mutual_information_samples_exact(
  const std::vector<uint32_t>& qubitsA, const std::vector<uint32_t>& qubitsB, size_t num_samples
) {
  auto [_qubits, _qubitsA, _qubitsB] = get_traced_qubits(qubitsA, qubitsB, num_qubits);

  auto stateAB = partial_trace(_qubits);

  ProbabilityFunc p1 = [](double t) -> double { return std::pow(t, 2.0); };
  ProbabilityFunc p2 = [](double t) -> double { return std::pow(t, 4.0); };
  auto samples1 = stateAB->sample_paulis_exact(num_samples, p1);
  auto samples2 = stateAB->sample_paulis_exact(num_samples, p2);

  return process_magic_mutual_information_samples(stateAB, _qubitsA, _qubitsB, samples1, samples2);
}

std::vector<MMIMonteCarloSamples> QuantumState::bipartite_magic_mutual_information_samples_montecarlo(
  size_t num_samples, size_t equilibration_timesteps, std::optional<PauliMutationFunc> mutation_opt
) {
  PauliMutationFunc mutation = single_qubit_random_mutation;
  if (mutation_opt) {
    mutation = mutation_opt.value();
  }

  auto get_samples = [&](ProbabilityFunc p) {
    auto pauli_samples = sample_paulis_montecarlo(num_samples, equilibration_timesteps, p, mutation);

    std::vector<MonteCarloSamples> samples(num_qubits/2 - 1);

    for (size_t i = 0; i < num_samples; i++) {
      auto const [P, t] = pauli_samples[i];

      std::vector<double> tA = pauli_expectation_left_sweep(P, 0, num_qubits/2);
      std::vector<double> tB = pauli_expectation_right_sweep(P, num_qubits/2, 0);
      std::reverse(tB.begin(), tB.end());

      for (size_t j = 0; j < num_qubits/2 - 1; j++) {
        samples[j][0].push_back(tA[j]);
        samples[j][1].push_back(tB[j]);
        samples[j][2].push_back(t);
      }
    }

    return samples;
  };

  ProbabilityFunc p1 = [](double t) -> double { return std::pow(t, 2.0); };
  auto t2 = get_samples(p1);
  ProbabilityFunc p2 = [](double t) -> double { return std::pow(t, 4.0); };
  auto t4 = get_samples(p2);

  std::vector<MMIMonteCarloSamples> samples;
  for (size_t n = 0; n < num_qubits/2 - 1; n++) {
    samples.push_back({t2[n], t4[n]});
  }

  return samples;
}

std::vector<MMIMonteCarloSamples> QuantumState::bipartite_magic_mutual_information_samples_exact(size_t num_samples) {
  auto get_samples = [&](ProbabilityFunc p) {
    // for a fixed probability function
    // returns a std::vector<MonteCarloSamples>
    // at the end, rearrange into pairs
    auto pauli_samples = sample_paulis_exact(num_samples, p);

    std::vector<MonteCarloSamples> samples(num_qubits/2 - 1);

    for (size_t i = 0; i < num_samples; i++) {
      auto const [P, t] = pauli_samples[i];

      std::vector<double> tA = pauli_expectation_left_sweep(P, 0, num_qubits/2);
      std::vector<double> tB = pauli_expectation_right_sweep(P, num_qubits/2, 0);
      std::reverse(tB.begin(), tB.end());

      for (size_t j = 0; j < num_qubits/2 - 1; j++) {
        samples[j][0].push_back(tA[j]);
        samples[j][1].push_back(tB[j]);
        samples[j][2].push_back(t);
      }
    }

    return samples;
  };

  ProbabilityFunc p1 = [](double t) -> double { return std::pow(t, 2.0); };
  auto t2 = get_samples(p1);
  ProbabilityFunc p2 = [](double t) -> double { return std::pow(t, 4.0); };
  auto t4 = get_samples(p2);

  std::vector<MMIMonteCarloSamples> samples;
  for (size_t n = 0; n < num_qubits/2 - 1; n++) {
    samples.push_back({t2[n], t4[n]});
  }

  return samples;
}

std::vector<double> QuantumState::bipartite_magic_mutual_information(size_t num_samples) {
  std::vector<double> magic(num_qubits/2 - 1);
  for (size_t i = 0; i < magic.size(); i++) {
    size_t j = i + 1;
    std::vector<uint32_t> qubitsA(j);
    std::iota(qubitsA.begin(), qubitsA.end(), 0);

    std::vector<uint32_t> qubitsB(num_qubits - j);
    std::iota(qubitsB.begin(), qubitsB.end(), j);

    magic[i] = magic_mutual_information(qubitsA, qubitsB, num_samples);
  }
  
  return magic;
}
