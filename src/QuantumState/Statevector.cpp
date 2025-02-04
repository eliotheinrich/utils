#include "QuantumStates.h"

#include <unsupported/Eigen/MatrixFunctions>

Statevector::Statevector(uint32_t num_qubits, uint32_t qregister) : QuantumState(num_qubits) {
  data = Eigen::VectorXcd::Zero(1u << num_qubits);
  data(qregister) = 1.;
}


Statevector::Statevector(uint32_t num_qubits) : Statevector(num_qubits, 0) {}

Statevector::Statevector(const QuantumCircuit &circuit) : Statevector(circuit.num_qubits) {
  evolve(circuit);
}

Statevector::Statevector(const Statevector& other) : Statevector(other.data) {}

Statevector::Statevector(const Eigen::VectorXcd& vec) : Statevector(std::log2(vec.size())) {
  uint32_t s = vec.size();
  if ((s & (s - 1)) != 0) {
    throw std::invalid_argument("Provided data to Statevector does not have a dimension which is a power of 2.");
  }

  data = vec;
}

Statevector::Statevector(const MatrixProductState& state) : Statevector(state.coefficients()) {}

std::string Statevector::to_string() const {
  Statevector tmp(*this);
  tmp.fix_gauge();

  uint32_t s = 1u << tmp.num_qubits;

  bool first = true;
  std::string st = "";
  for (uint32_t i = 0; i < s; i++) {
    if (std::abs(tmp.data(i)) > QS_ATOL) {
      std::string amplitude;
      if (std::abs(tmp.data(i).imag()) < QS_ATOL) {
        amplitude = fmt::format("{}", tmp.data(i).real());
      } else {
        amplitude = fmt::format("({}, {})", tmp.data(i).real(), tmp.data(i).imag());
      }

      std::string bin = quantumstate_utils::print_binary(i, tmp.num_qubits);

      if (!first) {
        st += " + ";
      }
      first = false;
      st += amplitude + "|" + bin + ">";
    }
  }

  return st;
}

double Statevector::entropy(const std::vector<uint32_t> &qubits, uint32_t index) {
  DensityMatrix rho(*this);
  return rho.entropy(qubits, index);
}

std::shared_ptr<QuantumState> Statevector::partial_trace(const Qubits& qubits) const {
  auto interval = to_interval(qubits);
  if (interval) {
    auto [q1, q2] = interval.value();
    if (q1 < 0 || q2 >= num_qubits) {
      throw std::runtime_error(fmt::format("qubits = {} passed to Statevector.partial_trace with {} qubits", qubits, num_qubits));
    }
  }
  DensityMatrix rho(*this);
  return rho.partial_trace(qubits);
}

double Statevector::expectation(const PauliString& p) const {
  if (p.num_qubits != num_qubits) {
    throw std::runtime_error(fmt::format("P = {} has {} qubits but state has {} qubits. Cannot compute expectation.", p.to_string_ops(), p.num_qubits, num_qubits));
  }

  Statevector s(*this);

  for (uint32_t i = 0; i < num_qubits; i++) {
    uint32_t j = i;
    Pauli op = p.to_pauli(i);
    if (op == Pauli::X) {
      s.evolve(quantumstate_utils::X::value, j);
    } else if (op == Pauli::Y) {
      s.evolve(quantumstate_utils::Y::value, j);
    } else if (op == Pauli::Z) {
      s.evolve(quantumstate_utils::Z::value, j);
    }
  }

  return p.sign() * inner(s).real();
}

std::complex<double> Statevector::expectation(const Eigen::MatrixXcd& m, const std::vector<uint32_t>& sites) const {
  Eigen::MatrixXcd M = full_circuit_unitary(m, sites, num_qubits);
  return expectation(M);
}

std::complex<double> Statevector::expectation(const Eigen::MatrixXcd& m) const {
  Statevector other(*this);
  other.evolve(m);
  return inner(other);
}

double Statevector::mzr_prob(uint32_t q, bool outcome) const {
  uint32_t s = 1u << num_qubits;

  double prob_zero = 0.0;
  for (uint32_t i = 0; i < s; i++) {
    if (((i >> q) & 1u) == 0) {
      prob_zero += std::pow(std::abs(data(i)), 2);
    }
  }

  if (outcome) {
    return 1.0 - prob_zero;
  } else {
    return prob_zero;
  }
}

bool Statevector::mzr(uint32_t q, bool outcome) {
  uint32_t s = 1u << num_qubits;
  for (uint32_t i = 0; i < s; i++) {
    if (((i >> q) & 1u) != outcome) {
      data(i) = 0.;
    }
  }

  normalize();

  return outcome;
}

bool Statevector::mzr(uint32_t q) {
  uint32_t s = 1u << num_qubits;

  double prob_zero = mzr_prob(q, 0);
  uint32_t outcome = !(QuantumState::randf() < prob_zero);

  for (uint32_t i = 0; i < s; i++) {
    if (((i >> q) & 1u) != outcome) {
      data(i) = 0.;
    }
  }

  normalize();

  return outcome;
}

MeasurementOutcome Statevector::measurement_outcome(const PauliString& p, const std::vector<uint32_t>& qubits) {
  Eigen::MatrixXcd pm = p.to_matrix();
  Eigen::MatrixXcd id = Eigen::MatrixXcd::Identity(1u << qubits.size(), 1u << qubits.size());

  Eigen::MatrixXcd proj0 = (id + pm)/2.0;
  Eigen::MatrixXcd proj1 = (id - pm)/2.0;

  double prob_zero = std::abs(expectation(proj0, qubits));

  double r = QuantumState::randf();
  bool outcome = r >= prob_zero;

  proj0 = proj0/std::sqrt(prob_zero);
  proj1 = proj1/std::sqrt(1.0 - prob_zero);

  Eigen::MatrixXcd proj = outcome ? proj1 : proj0;

  return {proj, prob_zero, outcome};
}

void Statevector::internal_measure(const MeasurementOutcome& outcome, const std::vector<uint32_t>& qubits, bool renormalize) {
  auto proj = std::get<0>(outcome);
  evolve(proj, qubits);
  if (renormalize) {
    normalize();
  }
}

bool Statevector::measure(const PauliString& p, const std::vector<uint32_t>& qubits) {
  if (qubits.size() != p.num_qubits) {
    throw std::runtime_error(fmt::format("PauliString {} has {} qubits, but {} qubits provided to measure.", p.to_string_ops(), p.num_qubits, qubits.size()));
  }

  auto outcome = measurement_outcome(p, qubits);
  internal_measure(outcome, qubits, true);

  return std::get<2>(outcome);
}

std::vector<bool> Statevector::measure(const std::vector<MeasurementData>& measurements) {
  std::vector<bool> results;
  for (auto const& [p, qubits] : measurements) {
    auto outcome = measurement_outcome(p, qubits);
    results.push_back(std::get<2>(outcome));
    internal_measure(outcome, qubits, true);
  }

  return results;
}

MeasurementOutcome Statevector::weak_measurement_outcome(const PauliString& p, const std::vector<uint32_t>& qubits, double beta) {
  PauliString p_ = p.superstring(qubits, num_qubits);

  auto pm = p.to_matrix();
  auto id = Eigen::MatrixXcd::Identity(1u << p.num_qubits, 1u << p.num_qubits);
  Eigen::MatrixXcd proj0 = (id + pm)/2.0;

  double prob_zero = std::abs(expectation(proj0, qubits));

  double r = QuantumState::randf();
  bool outcome = r >= prob_zero;

  Eigen::MatrixXcd t = pm;
  if (outcome) {
    t = -t;
  }

  Eigen::MatrixXcd proj = (beta*t).exp();

  return {proj, prob_zero, outcome};
}

bool Statevector::weak_measure(const PauliString& p, const std::vector<uint32_t>& qubits, double beta) {
  if (qubits.size() != p.num_qubits) {
    throw std::runtime_error(fmt::format("PauliString {} has {} qubits, but {} qubits provided to measure.", p.to_string_ops(), p.num_qubits, qubits.size()));
  }

  auto outcome = weak_measurement_outcome(p, qubits, beta);
  internal_measure(outcome, qubits, true);
  return std::get<2>(outcome);
}

std::vector<bool> Statevector::weak_measure(const std::vector<WeakMeasurementData>& measurements) {
  std::vector<bool> results;
  for (auto const& [p, qubits, beta] : measurements) {
    auto outcome = weak_measurement_outcome(p, qubits, beta);
    results.push_back(std::get<2>(outcome));
    internal_measure(outcome, qubits, true);
  }

  return results;
}

void Statevector::evolve(const Eigen::MatrixXcd &gate, const std::vector<uint32_t> &qubits) {
  uint32_t s = 1u << num_qubits;
  uint32_t h = 1u << qubits.size();
  if ((gate.rows() != h) || gate.cols() != h) {
    throw std::invalid_argument("Invalid gate dimensions for provided qubits.");
  }

  Eigen::VectorXcd ndata = Eigen::VectorXcd::Zero(s);

  for (uint32_t a1 = 0; a1 < s; a1++) {
    uint32_t b1 = quantumstate_utils::reduce_bits(a1, qubits);

    for (uint32_t b2 = 0; b2 < h; b2++) {
      uint32_t a2 = a1;
      for (uint32_t j = 0; j < qubits.size(); j++) {
        a2 = quantumstate_utils::set_bit(a2, qubits[j], b2, j);
      }

      ndata(a1) += gate(b1, b2)*data(a2);
    }
  }

  data = ndata;
}

void Statevector::evolve(const Eigen::MatrixXcd &gate) {
  if (!(gate.rows() == data.size() && gate.cols() == data.size())) {
    throw std::invalid_argument("Invalid gate dimensions for provided qubits. Can't do Statevector.evolve.");
  }

  data = gate*data;
}

void Statevector::evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) {
	QuantumState::evolve(gate, qubit);
}

// Vector representing diagonal gate
void Statevector::evolve_diagonal(const Eigen::VectorXcd &gate, const std::vector<uint32_t> &qubits) {
  throw std::runtime_error("evolve_diagonal is currently bugged.");

  //uint32_t s = 1u << num_qubits;
  //uint32_t h = 1u << qubits.size();

  //if (gate.size() != h) {
  //  throw std::invalid_argument("Invalid gate dimensions for provided qubits.");
  //}

  //for (uint32_t a = 0; a < s; a++) {
  //  uint32_t b = quantumstate_utils::reduce_bits(a, qubits);

  //  data(a) *= gate(h - b - 1);
  //}
}

void Statevector::evolve_diagonal(const Eigen::VectorXcd &gate) {
  throw std::runtime_error("evolve_diagonal is currently bugged.");

  //uint32_t s = 1u << num_qubits;

  //if (gate.size() != s) {
  //  throw std::invalid_argument("Invalid gate dimensions for provided qubits.");
  //}

  //for (uint32_t a = 0; a < s; a++) {
  //  data(a) *= gate(a);
  //}
}

void Statevector::evolve(const QuantumCircuit& circuit, const std::vector<bool>& outcomes) {
  uint32_t d = 0;
  for (auto const &inst : circuit.instructions) {
    if (inst.index() == 0) {
      QuantumState::evolve(inst);
    } else {
      Measurement m = std::get<Measurement>(inst);
      for (auto const& q : m.qbits) {
        mzr(q, outcomes[d]);
        d++;
      }
    }
  }
}

double Statevector::norm() const {
  double n = 0.;
  for (uint32_t i = 0; i < data.size(); i++) {
    n += std::pow(std::abs(data(i)), 2);
  }

  return std::sqrt(n);
}

void Statevector::normalize() {
  data = data/norm();
}

void Statevector::fix_gauge() {
  uint32_t j = 0;
  uint32_t s = 1u << num_qubits;
  for (uint32_t i = 0; i < s; i++) {
    if (std::abs(data(i)) > QS_ATOL) {
      j = i;
      break;
    }
  }

  std::complex<double> a = data(j)/std::abs(data(j));

  data = data/a;
}

double Statevector::probabilities(uint32_t z, const std::vector<uint32_t>& qubits) const {
  uint32_t s = 1u << num_qubits;
  double p = 0.;
  for (uint32_t i = 0; i < s; i++) {

    if (quantumstate_utils::bits_congruent(i, z, qubits)) {
      p += std::pow(std::abs(data(i)), 2);
    }
  }

  return p;
}

std::map<uint32_t, double> Statevector::probabilities_map() const {
  std::vector<double> probs = probabilities();


  std::map<uint32_t, double> probs_map;
  for (uint32_t i = 0; i < probs.size(); i++) {
    probs_map.emplace(i, probs[i]);
  }

  return probs_map; 
}

std::vector<double> Statevector::probabilities() const {
  uint32_t s = 1u << num_qubits;

  std::vector<double> probs(s);
  for (uint32_t i = 0; i < s; i++) {
    probs[i] = std::pow(std::abs(data(i)), 2);
  }

  return probs;
}

std::complex<double> Statevector::inner(const Statevector& other) const {
  uint32_t s = 1u << num_qubits;

  std::complex<double> c = 0.;
  for (uint32_t i = 0; i < s; i++) {
    c += other.data(i)*std::conj(data(i));
  }

  return c;
}


Eigen::VectorXd Statevector::svd(const std::vector<uint32_t>& qubits) const {
  Eigen::MatrixXcd matrix(data);

  uint32_t r = 1u << qubits.size();
  uint32_t c = 1u << (num_qubits - qubits.size());
  matrix.resize(r, c);

  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  return svd.singularValues();
}
