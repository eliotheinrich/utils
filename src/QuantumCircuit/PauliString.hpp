#pragma once

#include <unsupported/Eigen/KroneckerProduct>
#include <iostream>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <ranges>

#include "QuantumCircuit.h"

template <typename T>
static void remove_even_indices(std::vector<T> &v) {
  uint32_t vlen = v.size();
  for (uint32_t i = 0; i < vlen; i++) {
    uint32_t j = vlen - i - 1;
    if ((j % 2)) {
      v.erase(v.begin() + j);
    }
  }
}

enum Pauli {
  I, X, Z, Y
};

static int multiplication_phase(uint8_t xz1, uint8_t xz2) {
  bool x1 = (xz1 >> 0u) & 1u;
  bool z1 = (xz1 >> 1u) & 1u;
  bool x2 = (xz2 >> 0u) & 1u;
  bool z2 = (xz2 >> 1u) & 1u;
  if (!x1 && !z1) { 
    return 0; 
  } else if (x1 && z1) {
    if (z2) { 
      return x2 ? 0 : 1;
    } else { 
      return x2 ? -1 : 0;
    }
  } else if (x1 && !z1) {
    if (z2) { 
      return x2 ? 1 : -1;
    } else { 
      return 0; 
    }
  } else {
    if (x2) {
      return z2 ? -1 : 1;
    } else { 
      return 0; 
    }
  }
}

constexpr static std::pair<Pauli, uint8_t> multiply_pauli(Pauli p1, Pauli p2) {
  uint8_t p1_bits = static_cast<uint8_t>(p1);
  uint8_t p2_bits = static_cast<uint8_t>(p2);

  int phase = multiplication_phase(p1_bits, p2_bits);
  uint8_t phase_bits = 0b10; // 2 -> -1 (should never happen in this context)
  if (phase == 0) {
    phase_bits = 0b00; // 0 -> 1
  } else if (phase == -1) {
    phase_bits = 0b11; // 3 -> -i
  } else if (phase == 1) {
    phase_bits = 0b01; // 1 -> i
  }

  return {static_cast<Pauli>(p1_bits ^ p2_bits), phase_bits};
}

static char pauli_to_char(Pauli p) {
  if (p == Pauli::I) {
    return 'I';
  } else if (p == Pauli::X) {
    return 'X';
  } else if (p == Pauli::Y) {
    return 'Y';
  } else if (p == Pauli::Z) {
    return 'Z';
  }

  throw std::runtime_error("Unreachable.");
}

constexpr static std::complex<double> sign_from_bits(uint8_t phase) {
  constexpr std::complex<double> i(0.0, 1.0);
  if (phase == 0) {
    return 1.0;
  } else if (phase == 1) {
    return i;
  } else if (phase == 2) {
    return -1.0;
  } else {
    return -i;
  }
}

// TODO fix segfault when gate is applied to 0-qubit PauliString
class PauliString {
  public:
    uint32_t num_qubits;
    uint8_t phase;

    // Store bitstring as an array of 32-bit words
    // The bits are formatted as:
    // x0 z0 x1 z1 ... x15 z15
    // x16 z16 x17 z17 ... etc
    // This appears to be slightly more efficient than the originally format originally described
    // by Aaronson and Gottesman (https://arxiv.org/abs/quant-ph/0406196) as it 
    // is more cache-friendly; most operations only act on a single word.
    std::vector<uint32_t> bit_string;
    uint32_t width;

    PauliString()=default;
    PauliString(uint32_t num_qubits) : num_qubits(num_qubits), phase(0) {
      width = (2u*num_qubits) / 32 + static_cast<bool>((2u*num_qubits) % 32);
      bit_string = std::vector<uint32_t>(width, 0);
    }

    PauliString(const PauliString& other) {
      num_qubits = other.num_qubits;
      bit_string = other.bit_string;
      phase = other.phase;
      width = other.width;
    }

    static uint32_t process_pauli_string(const std::string& paulis) {
      uint32_t num_qubits = paulis.size();
      if (paulis[0] == '+' || paulis[0] == '-') {
        num_qubits--;

        if (paulis[1] == 'i') {
          num_qubits--;
        }
      }
      return num_qubits;
    }

    static inline uint8_t parse_phase(std::string& s) {
      if (s.rfind("+", 0) == 0) {
        if (s.rfind("+i", 0) == 0) {
          s = s.substr(2);
          return 1;
        } else {
          s = s.substr(1);
          return 0;
        }
      } else if (s.rfind("-", 0) == 0) {
        if (s.rfind("-i", 0) == 0) {
          s = s.substr(2);
          return 3;
        } else {
          s = s.substr(1);
          return 2;
        }
      }

      return 0;
    }

    PauliString(const std::string& paulis) : PauliString(process_pauli_string(paulis)) {
      std::string s = paulis;
      phase = parse_phase(s);

      for (size_t i = 0; i < num_qubits; i++) {
        if (s[i] == 'I') {
          set_x(i, false);
          set_z(i, false);
        } else if (s[i] == 'X') {
          set_x(i, true);
          set_z(i, false);
        } else if (s[i] == 'Y') {
          set_x(i, true);
          set_z(i, true);
        } else if (s[i] == 'Z') {
          set_x(i, false);
          set_z(i, true);
        } else {
          std::cout << fmt::format("character {} not recognized\n", s[i]);
          throw std::runtime_error(fmt::format("Invalid string {} used to create PauliString.", paulis));
        }
      }
    }

    PauliString(const std::vector<Pauli>& paulis, uint8_t phase=0) : PauliString(paulis.size()) { 
      for (size_t i = 0; i < paulis.size(); i++) {
        set_op(i, paulis[i]);
      }

      set_r(phase);
    }

    static PauliString rand(uint32_t num_qubits, std::minstd_rand& rng) {
      PauliString p(num_qubits);

      for (uint32_t j = 0; j < p.width; j++) {
        p.bit_string[j] = rng();
      }

      p.set_r(rng() % 4);

      if (num_qubits == 0) {
        return p;
      }

      // Need to check that at least one bit is nonzero so that p is not the identity
      for (uint32_t j = 0; j < num_qubits; j++) {
        if (p.get_xz(j)) {
          return p;
        }
      }

      return PauliString::rand(num_qubits, rng);
    }

    static PauliString randh(uint32_t num_qubits, std::minstd_rand& rng) {
      PauliString p = PauliString::rand(num_qubits, rng);

      if (!p.hermitian()) {
        p.set_r(p.get_r() + 1);
      }

      return p;
    }

    static PauliString basis(uint32_t num_qubits, const std::string& P, uint32_t q, uint8_t r) {
      PauliString p(num_qubits);
      if (P == "X") {
        p.set_x(q, true);
      } else if (P == "Y") {
        p.set_x(q, true);
        p.set_z(q, true);
      } else if (P == "Z") {
        p.set_z(q, true);
      } else {
        std::string error_message = P + " is not a valid basis. Must provide one of X,Y,Z.\n";
        throw std::invalid_argument(error_message);
      }

      p.set_r(r);

      return p;
    }

    static PauliString from_bitstring(uint32_t num_qubits, uint32_t bits) {
      PauliString p = PauliString(num_qubits);
      p.bit_string = {bits};
      return p;
    }

    static PauliString basis(uint32_t num_qubits, const std::string& P, uint32_t q) {
      return PauliString::basis(num_qubits, P, q, false);
    }

    PauliString substring(const Qubits& qubits, bool remove_qubits=false) const {
      size_t n = remove_qubits ? qubits.size() : num_qubits;
      PauliString p(n);

      if (remove_qubits) {
        for (size_t i = 0; i < qubits.size(); i++) {
          p.set_x(i, get_x(qubits[i]));
          p.set_z(i, get_z(qubits[i]));
        }
      } else {
        for (const auto q : qubits) {
          p.set_x(q, get_x(q));
          p.set_z(q, get_z(q));
        }
      }

      p.set_r(get_r());
      return p;
    }

    PauliString substring(const QubitSupport& support, bool remove_qubits=false) const {
      return substring(to_qubits(support), remove_qubits);
    }

    PauliString superstring(const Qubits& qubits, size_t new_num_qubits) const {
      if (qubits.size() != num_qubits) {
        throw std::runtime_error(fmt::format("When constructing a superstring Pauli, provided sites must have same size as num_qubits. P = {}, qubits = {}.", to_string_ops(), qubits));
      }
      std::vector<Pauli> paulis(new_num_qubits, Pauli::I);
      for (size_t i = 0; i < num_qubits; i++) {
        uint32_t q = qubits[i];

        paulis[q] = to_pauli(i);
      }

      return PauliString(paulis, get_r());
    }

    static uint8_t get_multiplication_phase(const PauliString& p1, const PauliString& p2) {
      uint8_t s = p1.get_r() + p2.get_r();

      for (uint32_t j = 0; j < p1.num_qubits; j++) {
        s += multiplication_phase(p1.get_xz(j), p2.get_xz(j));
      }

      return s;
    }

    bool hermitian() const {
      return !(phase & 0b1); // mod 2
    }

    PauliString operator*(const PauliString& other) const {
      if (num_qubits != other.num_qubits) {
        throw std::runtime_error(fmt::format("Multiplying PauliStrings with {} qubits and {} qubits do not match.", num_qubits, other.num_qubits));
      }

      PauliString p(num_qubits);

      p.set_r(PauliString::get_multiplication_phase(*this, other));

      uint32_t width = other.width;
      for (uint32_t j = 0; j < width; j++) {
        p.bit_string[j] = bit_string[j] ^ other.bit_string[j];
      }

      return p;
    }

    //PauliString& operator*=(const PauliString& other) {
    //  if (num_qubits != other.num_qubits) {
    //    throw std::runtime_error(fmt::format("Multiplying PauliStrings with {} qubits and {} qubits do not match.", num_qubits, other.num_qubits));
    //  }

    //  set_r(PauliString::get_multiplication_phase(*this, other));

    //  uint32_t width = other.width;
    //  for (uint32_t j = 0; j < width; j++) {
    //    bit_string[j] ^= other.bit_string[j];
    //  }

    //  return *this;
    //}

    PauliString operator-() { 
      set_r(get_r() + 2);
      return *this;
    }

    bool operator==(const PauliString &rhs) const {
      if (num_qubits != rhs.num_qubits) {
        return false;
      }

      if (get_r() != rhs.get_r()) {
        return false;
      }

      for (uint32_t i = 0; i < num_qubits; i++) {
        if (get_x(i) != rhs.get_x(i)) {
          return false;
        }

        if (get_z(i) != rhs.get_z(i)) {
          return false;
        }
      }

      return true;
    }

    bool operator!=(const PauliString &rhs) const { 
      return !(this->operator==(rhs)); 
    }

    friend std::ostream& operator<< (std::ostream& stream, const PauliString& p) {
      stream << p.to_string_ops();
      return stream;
    }

    std::vector<uint32_t>::iterator begin() {
      return bit_string.begin();
    }

    std::vector<uint32_t>::iterator end() {
      return bit_string.end();
    }

    Eigen::Matrix2cd to_matrix(uint32_t i) const {
      std::string s = to_op(i);

      Eigen::Matrix2cd g;
      if (s == "I") {
        g << 1, 0, 0, 1;
      } else if (s == "X") {
        g << 0, 1, 1, 0;
      } else if (s == "Y") {
        g << 0, std::complex<double>(0.0, -1.0), std::complex<double>(0.0, 1.0), 0;
      } else {
        g << 1, 0, 0, -1;
      }

      return g;
    }

    Eigen::MatrixXcd to_matrix() const {
      Eigen::MatrixXcd g = to_matrix(0);

      for (uint32_t i = 1; i < num_qubits; i++) {
        Eigen::MatrixXcd gi = to_matrix(i);
        Eigen::MatrixXcd g0 = g;
        g = Eigen::kroneckerProduct(gi, g0);
      }

      constexpr std::complex<double> i(0.0, 1.0);

      if (phase == 1) {
        g = i*g;
      } else if (phase == 2) {
        g = -g;
      } else if (phase == 3) {
        g = -i*g;
      }

      return g;
    }

    Pauli to_pauli(uint32_t i) const {
      bool xi = get_x(i);
      bool zi = get_z(i);

      if (xi && zi) {
        return Pauli::Y;
      } else if (!xi && zi) {
        return Pauli::Z;
      } else if (xi && !zi) {
        return Pauli::X;
      } else {
        return Pauli::I;
      }
    }

    std::vector<Pauli> to_pauli() const {
      std::vector<Pauli> paulis(num_qubits);
      std::generate(paulis.begin(), paulis.end(), [n = 0, this]() mutable { return to_pauli(n++); });
      return paulis;
    }

    QubitInterval support_range() const {
      std::vector<Pauli> paulis = to_pauli();
      auto first = std::ranges::find_if(paulis, [&](Pauli pi) { return pi != Pauli::I; });
      auto last = std::ranges::find_if(paulis | std::views::reverse, [&](Pauli pi) { return pi != Pauli::I; });

      if (first == paulis.end() && last == paulis.rend()) {
        return std::nullopt;
      } else {
        uint32_t q1 = std::distance(paulis.begin(), first);
        uint32_t q2 = num_qubits - std::distance(paulis.rbegin(), last);
        return std::make_pair(q1, q2);
      }
    }

    Qubits get_support() const {
      Qubits support;
      for (size_t i = 0; i < num_qubits; i++) {
        if (to_pauli(i) != Pauli::I) {
          support.push_back(i);
        }
      }

      return support;
    }

    std::string to_op(uint32_t i) const {
      bool xi = get_x(i); 
      bool zi = get_z(i);

      if (xi && zi) {
        return "Y";
      } else if (!xi && zi) {
        return "Z";
      } else if (xi && !zi) {
        return "X";
      } else {
        return "I";
      }
    }

    inline static std::string phase_to_string(uint8_t phase) {
      if (phase == 0) {
        return "+";
      } else if (phase == 1) {
        return "+i";
      } else if (phase == 2) {
        return "-";
      } else if (phase == 3) {
        return "-i";
      }

      throw std::runtime_error("Invalid phase bits passed to phase_to_string.");
    }

    std::string to_string() const {
      std::string s = "[ ";
      for (uint32_t i = 0; i < num_qubits; i++) {
        s += get_x(i) ? "1" : "0";
      }

      for (uint32_t i = 0; i < num_qubits; i++) {
        s += get_z(i) ? "1" : "0";
      }

      s += " | ";
      s += phase_to_string(phase);
      s += " ]";

      return s;
    }

    std::string to_string_ops() const {
      std::string s = phase_to_string(phase);

      for (uint32_t i = 0; i < num_qubits; i++) {
        s += to_op(i);
      }

      return s;
    }

    void evolve(const QuantumCircuit& qc) {
      if (!qc.is_clifford()) {
        throw std::runtime_error("Provided circuit is not Clifford.");
      }

      if (qc.num_qubits != num_qubits) {
        throw std::runtime_error(fmt::format("Cannot evolve a Paulistring with {} qubits with a QuantumCircuit with {} qubits.", num_qubits, qc.num_qubits));
      }

			for (auto const &inst : qc.instructions) {
				evolve(inst);
			}
    }

		void evolve(const Instruction& inst) {
			std::visit(quantumcircuit_utils::overloaded{
				[this](std::shared_ptr<Gate> gate) { 
          std::string name = gate->label();

          if (name == "H") {
            h(gate->qubits[0]);
          } else if (name == "S") {
            s(gate->qubits[0]);
          } else if (name == "Sd") {
            sd(gate->qubits[0]);
          } else if (name == "X") {
            x(gate->qubits[0]);
          } else if (name == "Y") {
            y(gate->qubits[0]);
          } else if (name == "Z") {
            z(gate->qubits[0]);
          } else if (name == "sqrtX") {
            sqrtX(gate->qubits[0]);
          } else if (name == "sqrtY") {
            sqrtY(gate->qubits[0]);
          } else if (name == "sqrtZ") {
            sqrtZ(gate->qubits[0]);
          } else if (name == "sqrtXd") {
            sqrtXd(gate->qubits[0]);
          } else if (name == "sqrtYd") {
            sqrtYd(gate->qubits[0]);
          } else if (name == "sqrtZd") {
            sqrtZd(gate->qubits[0]);
          } else if (name == "CX") {
            cx(gate->qubits[0], gate->qubits[1]);
          } else if (name == "CY") {
            cy(gate->qubits[0], gate->qubits[1]);
          } else if (name == "CZ") {
            cz(gate->qubits[0], gate->qubits[1]);
          } else if (name == "SWAP") {
            swap(gate->qubits[0], gate->qubits[1]);
          } else {
            throw std::runtime_error(fmt::format("Invalid instruction \"{}\" provided to PauliString.evolve.", name));
          }
				},
				[](Measurement m) { 
          throw std::runtime_error(fmt::format("Cannot mzr a single PauliString."));
				},
			}, inst);
		}

    void s(uint32_t a) {
      uint8_t xza = get_xz(a);
      bool xa = (xza >> 0u) & 1u;
      bool za = (xza >> 1u) & 1u;

      uint8_t r = phase;

      constexpr uint8_t s_phase_lookup[] = {0, 0, 0, 2};
      set_r(r + s_phase_lookup[xza]);
      set_z(a, xa != za);
    }

    void sd(uint32_t a) {
      s(a);
      s(a);
      s(a);
    }

    void h(uint32_t a) {
      uint8_t xza = get_xz(a);
      bool xa = (xza >> 0u) & 1u;
      bool za = (xza >> 1u) & 1u;

      uint8_t r = phase;
      
      constexpr uint8_t h_phase_lookup[] = {0, 0, 0, 2};
      set_r(r + h_phase_lookup[xza]);
      set_x(a, za);
      set_z(a, xa);
    }

    void x(uint32_t a) {
      h(a);
      s(a);
      s(a);
      h(a);
    }

    void y(uint32_t a) {
      h(a);
      s(a);
      s(a);
      h(a);
      s(a);
      s(a);
    }

    void z(uint32_t a) {
      s(a);
      s(a);
    }

    void sqrtX(uint32_t a) {
      sd(a);
      h(a);
      sd(a);
    }

    void sqrtXd(uint32_t a) {
      s(a);
      h(a);
      s(a);
    }

    void sqrtY(uint32_t a) {
      z(a);
      h(a);
    }

    void sqrtYd(uint32_t a) {
      h(a);
      z(a);
    }

    void sqrtZ(uint32_t a) {
      s(a);
    }

    void sqrtZd(uint32_t a) {
      sd(a);
    }

    void cx(uint32_t a, uint32_t b) {
      uint8_t xza = get_xz(a);
      bool xa = (xza >> 0u) & 1u;
      bool za = (xza >> 1u) & 1u;

      uint8_t xzb = get_xz(b);
      bool xb = (xzb >> 0u) & 1u;
      bool zb = (xzb >> 1u) & 1u;

      uint8_t bitcode = xzb + (xza << 2);

      uint8_t r = phase;

      constexpr uint8_t cx_phase_lookup[] = {0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2};
      set_r(r + cx_phase_lookup[bitcode]);
      set_x(b, xa != xb);
      set_z(a, za != zb);
    }

    void cy(uint32_t a, uint32_t b) {
      s(a);
      h(a);
      cz(b, a);
      h(a);
      s(a);
      s(a);
      s(a);
    }

    void cz(uint32_t a, uint32_t b) {
      h(b);
      cx(a, b);
      h(b);
    }

    void swap(uint32_t a, uint32_t b) {
      cx(a, b);
      cx(b, a);
      cx(a, b);
    }

    bool commutes_at(const PauliString& p, uint32_t i) const {
      if ((get_x(i) == p.get_x(i)) && (get_z(i) == p.get_z(i))) { // operators are identical
        return true;
      } else if (!get_x(i) && !get_z(i)) { // this is identity
        return true;
      } else if (!p.get_x(i) && !p.get_z(i)) { // other is identity
        return true;
      } else {
        return false; 
      }
    }

    bool commutes(const PauliString& p) const {
      if (num_qubits != p.num_qubits) {
        throw std::invalid_argument(fmt::format("p = {} has {} qubits and q = {} has {} qubits; cannot check commutation.", p.to_string_ops(), p.num_qubits, to_string_ops(), num_qubits));
      }

      uint32_t anticommuting_indices = 0u;
      for (uint32_t i = 0; i < num_qubits; i++) {
        if (!commutes_at(p, i)) {
          anticommuting_indices++;
        }
      }

      return anticommuting_indices % 2 == 0;
    }

    template <typename... Args>
    void reduce(bool z, Args... args) const {
      PauliString p(*this);
      p.reduce_inplace(z, args...);
    }

    template <typename... Args>
    void reduce_inplace(bool z, Args... args) {
      if (z) {
        h(0);
        (args.first->h(args.second[0]), ...);
      }

      // Step one
      for (uint32_t i = 0; i < num_qubits; i++) {
        if (get_z(i)) {
          if (get_x(i)) {
            s(i);
            (args.first->s(args.second[i]), ...);
          } else {
            h(i);
            (args.first->h(args.second[i]), ...);
          }
        }
      }

      // Step two
      std::vector<uint32_t> nonzero_idx;
      for (uint32_t i = 0; i < num_qubits; i++) {
        if (get_x(i)) {
          nonzero_idx.push_back(i);
        }
      }

      while (nonzero_idx.size() > 1) {
        for (uint32_t j = 0; j < nonzero_idx.size()/2; j++) {
          uint32_t q1 = nonzero_idx[2*j];
          uint32_t q2 = nonzero_idx[2*j+1];
          cx(q1, q2);
          (args.first->cx(args.second[q1], args.second[q2]), ...);
        }

        remove_even_indices(nonzero_idx);
      }

      // Step three
      uint32_t ql = nonzero_idx[0];
      if (ql != 0) {
        for (uint32_t i = 0; i < num_qubits; i++) {
          if (get_x(i)) {
            cx(0, ql);
            cx(ql, 0);
            cx(0, ql);

            (args.first->cx(args.second[0], args.second[ql]), ...);
            (args.first->cx(args.second[ql], args.second[0]), ...);
            (args.first->cx(args.second[0], args.second[ql]), ...);

            break;
          }
        }
      }

      if (z) {
        // tableau is discarded after function exits, so no need to apply it here. Just add to circuit.
        h(0);
        (args.first->h(args.second[0]), ...);
      }
    }

    // Returns the circuit which maps this PauliString onto p
    QuantumCircuit transform(PauliString const &p) const {
      Qubits qubits(p.num_qubits);
      std::iota(qubits.begin(), qubits.end(), 0);

      QuantumCircuit qc1(p.num_qubits);
      reduce(true, std::make_pair(&qc1, qubits));

      QuantumCircuit qc2(p.num_qubits);
      p.reduce(true, std::make_pair(&qc2, qubits));

      qc1.append(qc2.adjoint());

      return qc1;
    }

    // It is slightly faster (~20-30%) to query both the x and z bits at a given site
    // at the same time, storing them in the first two bits of the return value.
    inline uint8_t get_xz(uint32_t i) const {
      uint32_t word = bit_string[i / 16u];
      uint32_t bit_ind = 2u*(i % 16u);

      return 0u | (((word >> bit_ind) & 3u) << 0u);
    }

    inline bool get_x(uint32_t i) const { 
      uint32_t word = bit_string[i / 16u];
      uint32_t bit_ind = 2u*(i % 16u);
      return (word >> bit_ind) & 1u;
    }

    inline bool get_z(uint32_t i) const { 
      uint32_t word = bit_string[i / 16u];
      uint32_t bit_ind = 2u*(i % 16u) + 1u;
      return (word >> bit_ind) & 1u; 
    }

    inline uint8_t get_r() const { 
      return phase; 
    }

    inline std::complex<double> sign() const {
      return sign_from_bits(phase);
    }

    inline void set(size_t i, bool v) {
      uint32_t word_ind = i / 32u;
      uint32_t bit_ind = i % 32u;
      bit_string[word_ind] = (bit_string[word_ind] & ~(1u << bit_ind)) | (v << bit_ind);
    }

    inline bool get(size_t i) const {
      uint32_t word_ind = i / 32u;
      uint32_t bit_ind = i % 32u;
      return (bit_string[word_ind] >> bit_ind) & 1u;
    }

    inline void set_op(size_t i, Pauli p) {
      if (p == Pauli::I) {
        set_x(i, false);
        set_z(i, false);
      } else if (p == Pauli::X) {
        set_x(i, true);
        set_z(i, false);
      } else if (p == Pauli::Y) {
        set_x(i, true);
        set_z(i, true);
      } else if (p == Pauli::Z) {
        set_x(i, false);
        set_z(i, true);
      }
    }

    inline void set_x(uint32_t i, bool v) { 
      uint32_t word_ind = i / 16u;
      uint32_t bit_ind = 2u*(i % 16u);
      bit_string[word_ind] = (bit_string[word_ind] & ~(1u << bit_ind)) | (v << bit_ind);
    }

    inline void set_z(uint32_t i, bool v) { 
      uint32_t word_ind = i / 16u;
      uint32_t bit_ind = 2u*(i % 16u) + 1u;
      bit_string[word_ind] = (bit_string[word_ind] & ~(1u << bit_ind)) | (v << bit_ind);
    }

    inline void set_r(uint8_t v) { 
      phase = v & 0b11; 
    }
};

namespace fmt {
  template <>
  struct formatter<PauliString> {
    constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
      return ctx.begin();
    }

    // Format function
    template <typename FormatContext>
      auto format(const PauliString& ps, FormatContext& ctx) -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "{}", ps.to_string_ops());
      }
  };
}


template <class T>
void single_qubit_clifford_impl(T& qobj, size_t q, size_t r) {
  // r == 0 is identity, so do nothing in this case
  // Conjugates are marked as comments next to each case
  if (r == 1) { // 1
    qobj.x(q);
  } else if (r == 2) { // 2
    qobj.y(q);
  } else if (r == 3) { // 3
    qobj.z(q);
  } else if (r == 4) { // 8
    qobj.h(q);
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
  } else if (r == 5) { // 11
    qobj.h(q);
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
    qobj.x(q);
  } else if (r == 6) { // 9
    qobj.h(q);
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
    qobj.y(q);
  } else if (r == 7) { // 10
    qobj.h(q);
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
    qobj.z(q);
  } else if (r == 8) { // 4
    qobj.h(q);
    qobj.s(q);
  } else if (r == 9) { // 6
    qobj.h(q);
    qobj.s(q);
    qobj.x(q);
  } else if (r == 10) { // 7
    qobj.h(q);
    qobj.s(q);
    qobj.y(q);
  } else if (r == 11) { // 5
    qobj.h(q);
    qobj.s(q);
    qobj.z(q);
  } else if (r == 12) { // 12
    qobj.h(q);
  } else if (r == 13) { // 15
    qobj.h(q);
    qobj.x(q);
  } else if (r == 14) { // 14
    qobj.h(q);
    qobj.y(q);
  } else if (r == 15) { // 13
    qobj.h(q);
    qobj.z(q);
  } else if (r == 16) { // 17
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
  } else if (r == 17) { // 16
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
    qobj.x(q);
  } else if (r == 18) { // 18
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
    qobj.y(q);
  } else if (r == 19) { // 19
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
    qobj.z(q);
  } else if (r == 20) { // 23
    qobj.s(q);
  } else if (r == 21) { // 21
    qobj.s(q);
    qobj.x(q);
  } else if (r == 22) { // 22
    qobj.s(q);
    qobj.y(q);
  } else if (r == 23) { // 20
    qobj.s(q);
    qobj.z(q);
  }
}


template<typename... Args>
void reduce_paulis(const PauliString& p1, const PauliString& p2, const Qubits& qubits, Args&... args) {
  PauliString p1_ = p1;
  PauliString p2_ = p2;

  reduce_paulis_inplace(p1_, p2_, qubits, args...);
}

template<typename... Args>
void reduce_paulis_inplace(PauliString& p1, PauliString& p2, const Qubits& qubits, Args&... args) {
  size_t num_qubits = p1.num_qubits;
  if (p2.num_qubits != num_qubits) {
    throw std::runtime_error(fmt::format("Cannot reduce tableau for provided PauliStrings {} and {}; mismatched number of qubits.", p1, p2));
  }

  Qubits qubits_(num_qubits);
  std::iota(qubits_.begin(), qubits_.end(), 0);

  p1.reduce_inplace(false, std::make_pair(&args, qubits)..., std::make_pair(&p2, qubits_));

  PauliString z1p = PauliString::basis(num_qubits, "Z", 0, 0);
  PauliString z1m = PauliString::basis(num_qubits, "Z", 0, 2);

  if (p2 != z1p && p2 != z1m) {
    p2.reduce_inplace(true, std::make_pair(&args, qubits)..., std::make_pair(&p1, qubits_));
  }

  uint8_t sa = p1.get_r();
  uint8_t sb = p2.get_r();

  auto interpret_sign = [](uint8_t s) {
    if (s == 0) {
      return false;
    } else if (s == 2) {
      return true;
    } else {
      throw std::runtime_error("Anomolous phase detected in reduce.");
    }
  };

  if (interpret_sign(sa)) {
    if (interpret_sign(sb)) {
      // apply y
      (args.y(qubits[0]), ...);
      p1.y(0);
      p2.y(0);
    } else {
      // apply z
      (args.z(qubits[0]), ...);
      p1.z(0);
      p2.z(0);
    }
  } else {
    if (interpret_sign(sb)) {
      // apply x
      (args.x(qubits[0]), ...);
      p1.x(0);
      p2.x(0);
    }
  }
}

// Performs an iteration of the random clifford algorithm outlined in https://arxiv.org/pdf/2008.06011.pdf
template <typename... Args>
void random_clifford_iteration_impl(const Qubits& qubits, std::minstd_rand& rng, Args&... args) {
  size_t num_qubits = qubits.size();

  // If only acting on one qubit, can easily lookup from a table
  if (num_qubits == 1) {
    size_t r = rng() % 24;
    (single_qubit_clifford_impl(args, {qubits[0]}, r), ...);
    return;
  }

  Qubits qubits_(num_qubits);
  std::iota(qubits_.begin(), qubits_.end(), 0);

  PauliString p1 = PauliString::randh(num_qubits, rng);
  PauliString p2 = PauliString::randh(num_qubits, rng);
  while (p1.commutes(p2)) {
    p2 = PauliString::randh(num_qubits, rng);
  }

  reduce_paulis_inplace(p1, p2, qubits, args...);
}

template <typename... Args>
void random_clifford_impl(const Qubits& qubits, std::minstd_rand& rng, Args&... args) {
  Qubits qubits_(qubits.begin(), qubits.end());

  for (uint32_t i = 0; i < qubits.size(); i++) {
    random_clifford_iteration_impl(qubits_, rng, args...);
    qubits_.pop_back();
  }
}

class CliffordTable {
  private:
    using TableauBasis = std::tuple<PauliString, PauliString, size_t>;
    std::vector<TableauBasis> circuits;

  public:
    CliffordTable()=default;

    CliffordTable(std::function<bool(const QuantumCircuit&)> filter) : circuits() {
      std::vector<std::pair<PauliString, PauliString>> basis;

      for (size_t s1 = 1; s1 < 16; s1++) {
        PauliString X, Z;
        X = PauliString::from_bitstring(2, s1);
        for (size_t s2 = 1; s2 < 16; s2++) {
          Z = PauliString::from_bitstring(2, s2);
          
          // Z should anticommute with X
          if (Z.commutes(X)) {
            continue;
          }

          basis.push_back({ X,  Z });
          basis.push_back({ X, -Z });
          basis.push_back({-X,  Z });
          basis.push_back({-X, -Z });
        }
      }

      Qubits qubits{0, 1};
      for (const auto& [X, Z] : basis) {
        for (size_t r = 0; r < 24; r++) {
          QuantumCircuit qc(2);
          reduce_paulis(X, Z, qubits, qc);
          single_qubit_clifford_impl(qc, 0, r);
          if (filter(qc)) {
            circuits.push_back(std::make_tuple(X, Z, r));
          }
        }
      }
    }

    CliffordTable(const CliffordTable& other) : circuits(other.circuits) {}

    size_t num_elements() const {
      return circuits.size();
    }

    QuantumCircuit get_circuit(uint32_t r) const {
      QuantumCircuit qc(2);
      auto [X, Z, r2] = circuits[r];

      reduce_paulis(X, Z, {0, 1}, qc);
      single_qubit_clifford_impl(qc, 0, r2);

      return qc;
    }

    std::vector<QuantumCircuit> get_circuits() const {
      std::vector<QuantumCircuit> circuits(num_elements());
      size_t r = 0;
      std::generate(circuits.begin(), circuits.end(), [&r, this]() { return get_circuit(r++); });
      return circuits;
    }

    template <typename... Args>
    void apply_random(std::minstd_rand& rng, const Qubits& qubits, Args&... args) {
      size_t r1 = rng() % circuits.size();

      auto [X, Z, r2] = circuits[r1];

      reduce_paulis(X, Z, qubits, args...);
      (single_qubit_clifford_impl(args, qubits[0], r2), ...);
    } 
};



