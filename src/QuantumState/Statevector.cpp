#include "QuantumStates.h"
#include "utils.hpp"

#include <unsupported/Eigen/MatrixFunctions>

#include <glaze/glaze.hpp>

Statevector::Statevector(uint32_t num_qubits, uint32_t qregister) : MagicQuantumState(num_qubits) {
  data = Eigen::VectorXcd::Zero(1u << num_qubits);
  data(qregister) = 1.;
}

Statevector::Statevector(uint32_t num_qubits) : Statevector(num_qubits, 0) {}

Statevector::Statevector(const QuantumCircuit &circuit) : Statevector(circuit.get_num_qubits()) {
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

Statevector::Statevector(const MatrixProductState& state) : Statevector(state.coefficients_pure()) {}

std::string Statevector::to_string() const {
  Statevector tmp(*this);
  tmp.fix_gauge();

  uint32_t s = 1u << tmp.num_qubits;

  bool first = true;
  std::string st = "";
  size_t n = 0;
  for (uint32_t i = 0; i < s; i++) {
    if (std::abs(tmp.data(i)) > QS_ATOL) {
      n++;
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

  if (n == 0) {
    return "0";
  }

  return st;
}

double Statevector::entropy(const Qubits& qubits, uint32_t index) {
  if (qubits.size() == 0 || qubits.size() == num_qubits) {
    return 0.0;
  }

  Qubits qubitsA = qubits;
  std::sort(qubitsA.begin(), qubitsA.end());
  Qubits qubitsB = to_qubits(support_complement(qubitsA, num_qubits));
  if (support_contiguous(qubitsA) && qubitsA[0] == 0) {
    Qubits qubitsB = to_qubits(support_complement(qubitsA, num_qubits));
    size_t dA = 1u << qubitsA.size();
    size_t dB = 1u << qubitsB.size();

    Eigen::Map<const Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> M(data.data(), dB, dA);

    Eigen::JacobiSVD<Eigen::MatrixXcd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd schmidt = svd.singularValues().array();
    Eigen::VectorXd probs = schmidt.array().square();
    std::vector<double> p(probs.data(), probs.data() + probs.size());

    return renyi_entropy(index, p);
  } else if (support_contiguous(qubitsB) && qubitsB[0] == 0) {
    std::cout << "Complementing and trying again\n";
    return entropy(qubitsB, index);
  } else {
    std::cout << "Calling DensityMatrix version\n";
    DensityMatrix rho(*this);
    return rho.entropy(qubitsA, index);
  }
}

std::shared_ptr<QuantumState> Statevector::partial_trace(const Qubits& qubits) const {
  auto interval = support_range(qubits);
  if (interval) {
    auto [q1, q2] = interval.value();
    if (q1 < 0 || q2 > num_qubits) {
      throw std::runtime_error(fmt::format("qubits = {} passed to Statevector.partial_trace with {} qubits", qubits, num_qubits));
    }
  }
  DensityMatrix rho(*this);
  return rho.partial_trace(qubits);
}

std::complex<double> Statevector::expectation(const PauliString& p) const {
  if (p.num_qubits != num_qubits) {
    throw std::runtime_error(fmt::format("P = {} has {} qubits but state has {} qubits. Cannot compute expectation.", p.to_string_ops(), p.num_qubits, num_qubits));
  }

  Statevector s(*this);

  for (uint32_t i = 0; i < num_qubits; i++) {
    Pauli op = p.to_pauli(i);
    if (op == Pauli::X) {
      s.evolve(gates::X::value, i);
    } else if (op == Pauli::Y) {
      s.evolve(gates::Y::value, i);
    } else if (op == Pauli::Z) {
      s.evolve_diagonal(gates::Z::value, {i});
    }
  }

  return p.sign() * inner(s);
}

std::complex<double> Statevector::expectation(const Eigen::MatrixXcd& m, const Qubits& qubits) const {
  Eigen::MatrixXcd M = full_circuit_unitary(m, qubits, num_qubits);
  return expectation(M);
}

std::complex<double> Statevector::expectation(const Eigen::MatrixXcd& m) const {
  Statevector other(*this);
  other.evolve(m);
  return inner(other);
}

double Statevector::expectation(const BitString& bits, std::optional<QubitSupport> support) const {
  if (support) {
    Qubits qubits = to_qubits(support.value());
    uint32_t zA = bits.to_integer();
    double p = 0.0;
    for (uint32_t z = 0; z < basis; z++) {
      if (quantumstate_utils::bits_congruent(z, zA, qubits)) {
        p += std::pow(std::abs(data(z)), 2.0);
      }
    }

    return p;
  } else {
    uint32_t z = bits.to_integer();
    return std::pow(std::abs(data(z)), 2.0);
  }
}

double Statevector::mzr_prob(uint32_t q, bool outcome) const {
  uint32_t s = 1u << num_qubits;

  double prob_one = 0.0;
  for (uint32_t i = 0; i < s; i++) {
    if ((i >> q) & 1u) {
      prob_one += std::pow(std::abs(data(i)), 2);
    }
  }

  if (outcome) {
    return prob_one;
  } else {
    return 1.0 - prob_one;
  }
}

bool Statevector::forced_mzr(uint32_t q, bool outcome) {
  double prob_zero = mzr_prob(q, 0);
  check_forced_measure(outcome, prob_zero);

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
  double prob_zero = mzr_prob(q, 0);
  uint32_t outcome = !(randf() < prob_zero);

  return forced_mzr(q, outcome);
}

bool Statevector::measure(const Measurement& m) {
  Qubits qubits = m.qubits;
  if (m.is_basis()) {
    if (m.is_forced()) {
      return forced_mzr(qubits[0], m.get_outcome());
    } else {
      return mzr(qubits[0]);
    }
  }

  PauliString pauli = m.get_pauli();
  Eigen::MatrixXcd pm = pauli.to_matrix();
  Eigen::MatrixXcd id = Eigen::MatrixXcd::Identity(1u << qubits.size(), 1u << qubits.size());

  Eigen::MatrixXcd proj0 = (id + pm)/2.0; // false
  Eigen::MatrixXcd proj1 = (id - pm)/2.0; // true

  double prob_zero = std::abs(expectation(proj0, qubits));

  bool b;
  if (m.is_forced()) {
    b = m.get_outcome();
  } else {
    double r = randf();
    b = (r > prob_zero);
  }

  check_forced_measure(b, prob_zero);
  Eigen::MatrixXcd proj = b ? proj1 / std::sqrt(1.0 - prob_zero) : proj0 / std::sqrt(prob_zero);
  evolve(proj, qubits);
  normalize();

  return b;
}

bool Statevector::weak_measure(const WeakMeasurement& m) {
  Qubits qubits = m.qubits;
  PauliString pauli = m.get_pauli();

  auto pm = pauli.to_matrix();
  auto id = Eigen::MatrixXcd::Identity(1u << pauli.num_qubits, 1u << pauli.num_qubits);
  Eigen::MatrixXcd proj0 = (id + pm)/2.0;

  double prob_zero = (1.0 + expectation(pauli.superstring(qubits, num_qubits))).real()/2.0;

  bool b;
  if (m.is_forced()) {
    b = m.get_outcome();
  } else {
    double r = randf();
    b = (r >= prob_zero);
  }

  Eigen::MatrixXcd t = pm;
  if (b) {
    t = -t;
  }

  Eigen::MatrixXcd proj = (m.beta*t).exp();

  evolve(proj, qubits);
  normalize();

  return b;
}

void Statevector::evolve(const Eigen::MatrixXcd &gate, const Qubits& qubits) {
  uint32_t s = 1u << num_qubits;
  uint32_t h = 1u << qubits.size();
  assert_gate_shape(gate, qubits);

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
void Statevector::evolve_diagonal(const Eigen::VectorXcd& gate, const Qubits& qubits) {
  uint32_t h = 1u << qubits.size();

  if (gate.size() != h) {
    throw std::invalid_argument("Invalid gate dimensions for provided qubits.");
  }

  //  uint32_t b1 = quantumstate_utils::reduce_bits(a1, qubits);

  //  for (uint32_t b2 = 0; b2 < h; b2++) {
  //    uint32_t a2 = a1;
  //    for (uint32_t j = 0; j < qubits.size(); j++) {
  //      a2 = quantumstate_utils::set_bit(a2, qubits[j], b2, j);
  //    }

  //    ndata(a1) += gate(b1, b2)*data(a2);
  //  }


  for (uint32_t a = 0; a < basis; a++) {
    uint32_t b = quantumstate_utils::reduce_bits(a, qubits);
    data(a) *= gate(b);
  }
}

void Statevector::evolve_diagonal(const Eigen::VectorXcd& gate) {
  if (gate.size() != basis) {
    throw std::invalid_argument("Invalid gate dimensions for provided qubits.");
  }

  for (uint32_t a = 0; a < basis; a++) {
    data(a) *= gate(a);
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

//double Statevector::probabilities(uint32_t z, const Qubits& qubits) const {
//  uint32_t s = 1u << num_qubits;
//  double p = 0.;
//  for (uint32_t i = 0; i < s; i++) {
//
//    if (quantumstate_utils::bits_congruent(i, z, qubits)) {
//      p += std::pow(std::abs(data(i)), 2);
//    }
//  }
//
//  return p;
//}

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


Eigen::VectorXd Statevector::svd(const Qubits& qubits) const {
  Eigen::MatrixXcd matrix(data);

  uint32_t r = 1u << qubits.size();
  uint32_t c = 1u << (num_qubits - qubits.size());
  matrix.resize(r, c);

  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  return svd.singularValues();
}

namespace glz::detail {
   template <>
   struct from<BEVE, Eigen::VectorXcd> {
      template <auto Opts>
      static void op(Eigen::VectorXcd& value, auto&&... args) {
        std::string str;
        read<BEVE>::op<Opts>(str, args...);
        std::vector<char> buffer(str.begin(), str.end());

        const size_t header_size = sizeof(size_t);
        size_t size;
        memcpy(&size, buffer.data(), sizeof(size_t));

        Eigen::VectorXcd vector(size);
        memcpy(vector.data(), buffer.data() + header_size, size * sizeof(std::complex<double>));

        value = vector; 
      }
   };

   template <>
   struct to<BEVE, Eigen::VectorXcd> {
      template <auto Opts>
      static void op(const Eigen::VectorXcd& value, auto&&... args) noexcept {
        const size_t size = value.size();
        const size_t header_size = sizeof(size_t);
        const size_t data_size = size * sizeof(std::complex<double>);
        std::vector<char> buffer(header_size + data_size);

        memcpy(buffer.data(), &size, sizeof(size_t));
        memcpy(buffer.data() + header_size, value.data(), data_size);

        std::string data(buffer.begin(), buffer.end());
        write<BEVE>::op<Opts>(data, args...);
      }
   };
}

struct Statevector::glaze {
  using T = Statevector;
  static constexpr auto value = glz::object(
    &T::data,
    &T::use_parent,
    &T::num_qubits,
    &T::basis
  );
};

std::vector<char> Statevector::serialize() const {
  std::vector<char> bytes;
  auto write_error = glz::write_beve(*this, bytes);
  if (write_error) {
    throw std::runtime_error(fmt::format("Error writing Statevector to binary: \n{}", glz::format_error(write_error, bytes)));
  }
  return bytes;
}

void Statevector::deserialize(const std::vector<char>& bytes) {
  auto parse_error = glz::read_beve(*this, bytes);
  if (parse_error) {
    throw std::runtime_error(fmt::format("Error reading Statevector from binary: \n{}", glz::format_error(parse_error, bytes)));
  }
}
