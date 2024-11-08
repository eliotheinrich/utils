#pragma once

#include <unsupported/Eigen/KroneckerProduct>
#include <iostream>
#include <fmt/format.h>
#include <fmt/ranges.h>

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
  I, X, Y, Z
};

class PauliString {
  public:
    uint32_t num_qubits;
    bool phase;

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
    PauliString(uint32_t num_qubits) : num_qubits(num_qubits), phase(false) {
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
      }
      return num_qubits;
    }

    PauliString(const std::string& paulis) : PauliString(process_pauli_string(paulis)) {
      std::string s = paulis;
      if (s[0] == '-') {
        phase = true;
        s = s.substr(1);
      } else if (s[0] == '+') {
        s = s.substr(1);
      }

      for (size_t i = 0; i < num_qubits; i++) {
        if (std::toupper(s[i]) == 'I') {
          set_x(i, false);
          set_z(i, false);
        } else if (std::toupper(s[i]) == 'X') {
          set_x(i, true);
          set_z(i, false);
        } else if (std::toupper(s[i]) == 'Y') {
          set_x(i, true);
          set_z(i, true);
        } else if (std::toupper(s[i]) == 'Z') {
          set_x(i, false);
          set_z(i, true);
        } else {
          throw std::runtime_error(fmt::format("Invalid string {} used to create PauliString.", paulis));
        }
      }
    }

    PauliString(const std::vector<Pauli>& paulis) : PauliString(paulis.size()) { 
      for (size_t i = 0; i < paulis.size(); i++) {
        if (paulis[i] == Pauli::I) {
          set_x(i, false);
          set_z(i, false);
        } else if (paulis[i] == Pauli::X) {
          set_x(i, true);
          set_z(i, false);
        } else if (paulis[i] == Pauli::Y) {
          set_x(i, true);
          set_z(i, true);
        } else if (paulis[i] == Pauli::Z) {
          set_x(i, false);
          set_z(i, true);
        }
      }
    }

    static PauliString rand(uint32_t num_qubits, std::minstd_rand& rng) {
      PauliString p(num_qubits);

      for (uint32_t j = 0; j < p.width; j++) {
        p.bit_string[j] = rng();
      }

      p.set_r(rng() % 2);

      // Need to check that at least one bit is nonzero so that p is not the identity
      for (uint32_t j = 0; j < num_qubits; j++) {
        if (p.get_xz(j)) {
          return p;
        }
      }

      return PauliString::rand(num_qubits, rng);
    }

    static PauliString basis(uint32_t num_qubits, const std::string& P, uint32_t q, bool r) {
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

    PauliString substring(const std::vector<uint32_t>& sites, bool remove_qubits=false) const {
      size_t n = remove_qubits ? sites.size() : num_qubits;
      PauliString p(n);

      if (remove_qubits) {
        for (size_t i = 0; i < sites.size(); i++) {
          p.set_x(i, get_x(sites[i]));
          p.set_z(i, get_z(sites[i]));
        }
      } else {
        for (const auto q : sites) {
          p.set_x(q, get_x(q));
          p.set_z(q, get_z(q));
        }
      }

      p.set_r(get_r());
      return p;
    }

    PauliString superstring(const std::vector<uint32_t>& sites, size_t new_num_qubits) const {
      if (sites.size() != num_qubits) {
        throw std::runtime_error(fmt::format("When constructing a superstring Pauli, provided sites must have same size as num_qubits. P = {}, sites = {}.", to_string_ops(), sites));
      }
      std::vector<Pauli> paulis(new_num_qubits, Pauli::I);
      for (size_t i = 0; i < num_qubits; i++) {
        uint32_t q = sites[i];

        paulis[q] = to_pauli(i);
      }

      return PauliString(paulis);
    }

    static int g(uint8_t xz1, uint8_t xz2) {
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

    static int get_multiplication_phase(const PauliString& p1, const PauliString& p2) {
      int s = 0;

      if (p1.get_r()) { 
        s += 2; 
      }

      if (p2.get_r()) { 
        s += 2; 
      }

      for (uint32_t j = 0; j < p1.num_qubits; j++) {
        s += PauliString::g(p1.get_xz(j), p2.get_xz(j));
      }

      return s;
    }

    PauliString operator*(const PauliString& other) const {
      if (num_qubits != other.num_qubits) {
        throw std::runtime_error(fmt::format("Multiplying PauliStrings with {} qubits and {} qubits do not match.", num_qubits, other.num_qubits));
      }

      int s = PauliString::get_multiplication_phase(*this, other);
      PauliString p(num_qubits);
      if (s % 4 == 0) {
        p.set_r(false);
      } else if (std::abs(s % 4) == 2) {
        p.set_r(true);
      }

      uint32_t width = other.width;
      for (uint32_t j = 0; j < width; j++) {
        p.bit_string[j] = bit_string[j] ^ other.bit_string[j];
      }

      return p;
    }

    PauliString& operator*=(const PauliString& other) {
      if (num_qubits != other.num_qubits) {
        throw std::runtime_error(fmt::format("Multiplying PauliStrings with {} qubits and {} qubits do not match.", num_qubits, other.num_qubits));
      }

      int s = PauliString::get_multiplication_phase(*this, other);
      if (s % 4 == 0) {
        set_r(false);
      } else if (std::abs(s % 4) == 2) {
        set_r(true);
      }

      uint32_t width = other.width;
      for (uint32_t j = 0; j < width; j++) {
        bit_string[j] ^= other.bit_string[j];
      }

      return *this;
    }

    PauliString operator-() { 
      set_r(!get_r());
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

      if (phase) {
        g = -g;
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

    std::string to_string() const {
      std::string s = "[ ";
      for (uint32_t i = 0; i < num_qubits; i++) {
        s += get_x(i) ? "1" : "0";
      }

      for (uint32_t i = 0; i < num_qubits; i++) {
        s += get_z(i) ? "1" : "0";
      }

      s += " | ";

      s += phase ? "1 ]" : "0 ]";

      return s;
    }

    std::string to_string_ops() const {
      std::string s = phase ? "-" : "+";

      for (uint32_t i = 0; i < num_qubits; i++) {
        s += to_op(i);
      }

      return s;
    }

    void evolve(const QuantumCircuit& qc) {
      if (!qc.is_clifford()) {
        throw std::runtime_error("Provided circuit is not Clifford.");
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
            h(gate->qbits[0]);
          } else if (name == "S") {
            s(gate->qbits[0]);
          } else if (name == "Sd") {
            sd(gate->qbits[0]);
          } else if (name == "X") {
            x(gate->qbits[0]);
          } else if (name == "Y") {
            y(gate->qbits[0]);
          } else if (name == "Z") {
            z(gate->qbits[0]);
          } else if (name == "CX") {
            cx(gate->qbits[0], gate->qbits[1]);
          } else if (name == "CZ") {
            cz(gate->qbits[0], gate->qbits[1]);
          } else {
            throw std::runtime_error(fmt::format("Invalid instruction \"{}\" provided to PauliString.evolve.", name));
          }
				},
				[this](Measurement m) { 
          throw std::runtime_error(fmt::format("Cannot mzr a single PauliString."));
				},
			}, inst);
		}

    void s(uint32_t a) {
      uint8_t xza = get_xz(a);
      bool xa = (xza >> 0u) & 1u;
      bool za = (xza >> 1u) & 1u;

      bool r = phase;

      set_r(r != (xa && za));
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

      bool r = phase;

      set_r(r != (xa && za));
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

    void cx(uint32_t a, uint32_t b) {
      uint8_t xza = get_xz(a);
      bool xa = (xza >> 0u) & 1u;
      bool za = (xza >> 1u) & 1u;

      uint8_t xzb = get_xz(b);
      bool xb = (xzb >> 0u) & 1u;
      bool zb = (xzb >> 1u) & 1u;

      bool r = phase;

      set_r(r != ((xa && zb) && ((xb != za) != true)));
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

    bool commutes_at(PauliString &p, uint32_t i) const {
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

    bool commutes(PauliString &p) const {
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
      std::vector<uint32_t> qubits(p.num_qubits);
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

    inline bool get_r() const { 
      return phase; 
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

    inline void set_r(bool v) { 
      phase = v; 
    }
};

template <class T>
void single_qubit_clifford_impl(T& qobj, size_t q, size_t r) {
  // r == 0 is identity, so do nothing in this case
  if (r == 1) {
    qobj.x(q);
  } else if (r == 2) {
    qobj.y(q);
  } else if (r == 3) {
    qobj.z(q);
  } else if (r == 4) {
    qobj.h(q);
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
  } else if (r == 5) {
    qobj.h(q);
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
    qobj.x(q);
  } else if (r == 6) {
    qobj.h(q);
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
    qobj.y(q);
  } else if (r == 7) {
    qobj.h(q);
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
    qobj.z(q);
  } else if (r == 8) {
    qobj.h(q);
    qobj.s(q);
  } else if (r == 9) {
    qobj.h(q);
    qobj.s(q);
    qobj.x(q);
  } else if (r == 10) {
    qobj.h(q);
    qobj.s(q);
    qobj.y(q);
  } else if (r == 11) {
    qobj.h(q);
    qobj.s(q);
    qobj.z(q);
  } else if (r == 12) {
    qobj.h(q);
  } else if (r == 13) {
    qobj.h(q);
    qobj.x(q);
  } else if (r == 14) {
    qobj.h(q);
    qobj.y(q);
  } else if (r == 15) {
    qobj.h(q);
    qobj.z(q);
  } else if (r == 16) {
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
  } else if (r == 17) {
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
    qobj.x(q);
  } else if (r == 18) {
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
    qobj.y(q);
  } else if (r == 19) {
    qobj.s(q);
    qobj.h(q);
    qobj.s(q);
    qobj.z(q);
  } else if (r == 20) {
    qobj.s(q);
  } else if (r == 21) {
    qobj.s(q);
    qobj.x(q);
  } else if (r == 22) {
    qobj.s(q);
    qobj.y(q);
  } else if (r == 23) {
    qobj.s(q);
    qobj.z(q);
  }
}

template<typename... Args>
void reduce_paulis(const PauliString& p1, const PauliString& p2, const std::vector<uint32_t>& qubits, Args&... args) {
  PauliString p1_ = p1;
  PauliString p2_ = p2;

  reduce_paulis_inplace(p1_, p2_, qubits, args...);
}

template<typename... Args>
void reduce_paulis_inplace(PauliString& p1, PauliString& p2, const std::vector<uint32_t>& qubits, Args&... args) {
  size_t num_qubits = p1.num_qubits;
  if (p2.num_qubits != num_qubits) {
    throw std::runtime_error(fmt::format("Cannot reduce tableau for provided PauliStrings {} and {}; mismatched number of qubits.", p1.to_string_ops(), p2.to_string_ops()));
  }

  std::vector<uint32_t> qubits_(num_qubits);
  std::iota(qubits_.begin(), qubits_.end(), 0);

  p1.reduce_inplace(false, std::make_pair(&args, qubits)..., std::make_pair(&p2, qubits_));

  PauliString z1p = PauliString::basis(num_qubits, "Z", 0, false);
  PauliString z1m = PauliString::basis(num_qubits, "Z", 0, true);

  if (p2 != z1p && p2 != z1m) {
    p2.reduce_inplace(true, std::make_pair(&args, qubits)..., std::make_pair(&p1, qubits_));
  }

  bool sa = p1.get_r();
  bool sb = p2.get_r();

  if (sa) {
    if (sb) {
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
    if (sb) {
      // apply x
      (args.x(qubits[0]), ...);
      p1.x(0);
      p2.x(0);
    }
  }
}

// Performs an iteration of the random clifford algorithm outlined in https://arxiv.org/pdf/2008.06011.pdf
template <typename... Args>
void random_clifford_iteration_impl(const std::vector<uint32_t>& qubits, std::minstd_rand& rng, Args&... args) {
  size_t num_qubits = qubits.size();

  // If only acting on one qubit, can easily lookup from a table
  if (num_qubits == 1) {
    size_t r = rng() % 24;
    (single_qubit_clifford_impl(args, {qubits[0]}, r), ...);
    return;
  }

  std::vector<uint32_t> qubits_(num_qubits);
  std::iota(qubits_.begin(), qubits_.end(), 0);

  PauliString p1 = PauliString::rand(num_qubits, rng);
  PauliString p2 = PauliString::rand(num_qubits, rng);
  while (p1.commutes(p2)) {
    p2 = PauliString::rand(num_qubits, rng);
  }

  reduce_paulis_inplace(p1, p2, qubits, args...);
}

template <typename... Args>
void random_clifford_impl(const std::vector<uint32_t>& qubits, std::minstd_rand& rng, Args&... args) {
  std::vector<uint32_t> qubits_(qubits.begin(), qubits.end());

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
    CliffordTable(std::optional<std::function<bool(const QuantumCircuit&)>> mask_opt = std::nullopt) {
      std::function<bool(const QuantumCircuit&)> mask;
      if (mask_opt) {
        mask = mask_opt.value();
      } else {
        mask = [](const QuantumCircuit& qc) -> bool { return true; };
      }

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

          basis.push_back({ X, Z });
          basis.push_back({ X, -Z });
          basis.push_back({ -X, Z });
          basis.push_back({ -X, -Z });
        }
      }

      std::vector<uint32_t> qubits{0, 1};
      for (const auto& [X, Z] : basis) {
        for (size_t r = 0; r < 24; r++) {
          QuantumCircuit qc(2);
          reduce_paulis(X, Z, qubits, qc);
          single_qubit_clifford_impl(qc, 0, r);
          if (mask(qc)) {
            circuits.push_back(std::make_tuple(X, Z, r));
          }
        }
      }

      std::cout << fmt::format("Finished generating table with {} elements\n", num_elements());
    }

    size_t num_elements() const {
      return 24 * circuits.size();
    }

    template <typename... Args>
    void apply_random(std::minstd_rand& rng, const std::vector<uint32_t>& qubits, Args&... args) {
      size_t r1 = rng() % circuits.size();

      auto [X, Z, r2] = circuits[r1];

      reduce_paulis(X, Z, qubits, args...);
      (single_qubit_clifford_impl(args, qubits[0], r2), ...);
    } 
};



