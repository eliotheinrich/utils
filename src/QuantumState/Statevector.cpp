#include "QuantumStates.h"

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


std::complex<double> Statevector::expectation(const PauliString& p) const {
  Eigen::MatrixXcd pm = p.to_matrix();
  return expectation(pm);
}

std::complex<double> Statevector::expectation(const Eigen::MatrixXcd& m, const std::vector<uint32_t>& sites) const {
  Eigen::MatrixXcd M = full_circuit_unitary(m, sites, num_qubits);
  return expectation(M);
}

std::complex<double> Statevector::expectation(const Eigen::MatrixXcd& m) const {
  Statevector other(num_qubits);
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
  uint32_t outcome = !(randf() < prob_zero);

  for (uint32_t i = 0; i < s; i++) {
    if (((i >> q) & 1u) != outcome) {
      data(i) = 0.;
    }
  }

  normalize();

  return outcome;
}

bool Statevector::measure(const PauliString& p, const std::vector<uint32_t>& qubits) {
  if (qubits.size() != p.num_qubits) {
    throw std::runtime_error(fmt::format("PauliString {} has {} qubits, but {} qubits provided to measure.", p.to_string_ops(), p.num_qubits, qubits.size()));
  }

  std::vector<Pauli> paulis(num_qubits, Pauli::I);
  for (size_t i = 0; i < qubits.size(); i++) {
    paulis[i] = p.to_pauli(i);
  }

  PauliString p_(paulis);
  Eigen::MatrixXcd matrix = p_.to_matrix();
  Eigen::MatrixXcd id = Eigen::MatrixXcd::Identity(basis, basis);

  double prob_zero = std::abs(expectation((id + matrix)/2.0));

  bool outcome = randf() >= prob_zero;

  Eigen::MatrixXcd proj0 = (id + matrix)/(2.0*std::sqrt(prob_zero));
  Eigen::MatrixXcd proj1 = (id - matrix)/(2.0*std::sqrt(1.0 - prob_zero));
  Eigen::MatrixXcd proj = outcome ? proj1 : proj0;

  evolve(proj);
  normalize();

  return outcome;
}

bool Statevector::weak_measure(const PauliString& p, const std::vector<uint32_t>& qubits, double beta) {
  std::vector<Pauli> paulis(num_qubits, Pauli::I);
  for (size_t i = 0; i < qubits.size(); i++) {
    paulis[i] = p.to_pauli(i);
  }

  PauliString p_(paulis);
  Eigen::MatrixXcd matrix = p_.to_matrix();
  Eigen::MatrixXcd id = Eigen::MatrixXcd::Identity(basis, basis);

  double prob_zero = std::abs(expectation((id + matrix)/2.0));

  bool outcome = randf() >= prob_zero;

  Eigen::MatrixXcd t = matrix;
  if (outcome) {
    t = -t;
  }

  Eigen::MatrixXcd proj = (beta*t).exp();

  evolve(proj);
  normalize();

  return outcome;
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
    throw std::invalid_argument("Invalid gate dimensions for provided qubits.");
  }

  data = gate*data;
}

void Statevector::evolve(const Eigen::Matrix2cd& gate, uint32_t qubit) {
	QuantumState::evolve(gate, qubit);
}

// Vector representing diagonal gate
void Statevector::evolve_diagonal(const Eigen::VectorXcd &gate, const std::vector<uint32_t> &qubits) {
  uint32_t s = 1u << num_qubits;
  uint32_t h = 1u << qubits.size();

  if (gate.size() != h) {
    throw std::invalid_argument("Invalid gate dimensions for provided qubits.");
  }

  for (uint32_t a = 0; a < s; a++) {
    uint32_t b = quantumstate_utils::reduce_bits(a, qubits);

    data(a) *= gate(b);
  }
}

void Statevector::evolve_diagonal(const Eigen::VectorXcd &gate) {
  uint32_t s = 1u << num_qubits;

  if (gate.size() != s) {
    throw std::invalid_argument("Invalid gate dimensions for provided qubits.");
  }

  for (uint32_t a = 0; a < s; a++) {
    data(a) *= gate(a);
  }
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
