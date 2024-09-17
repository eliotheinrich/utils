#pragma once

#include <unsupported/Eigen/KroneckerProduct>

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


class PauliString {
  public:
    uint32_t num_qubits;
    bool phase;

    // Store bitstring as an array of 32-bit words
    // The bits are formatted as:
    // x0 z0 x1 z1 ... x15 z15
    // x16 z16 x17 z17 ... etc
    // This is slightly more efficient than the originally format originally described
    // by Aaronson and Gottesman (https://arxiv.org/abs/quant-ph/0406196) as it 
    // is more cache-friendly; most operations only act on a single word.
    std::vector<uint32_t> bit_string;
    uint32_t width;

    PauliString()=default;
    PauliString(uint32_t num_qubits) : num_qubits(num_qubits), phase(false) {
      width = (2u*num_qubits) / 32 + static_cast<bool>((2u*num_qubits) % 32);
      bit_string = std::vector<uint32_t>(width, 0);
    }

    static PauliString rand(uint32_t num_qubits, std::minstd_rand& r) {
      PauliString p(num_qubits);

      for (uint32_t j = 0; j < p.width; j++) {
        p.bit_string[j] = r();
      }

      p.set_r(r() % 2);

      // Need to check that at least one bit is nonzero so that p is not the identity
      for (uint32_t j = 0; j < num_qubits; j++) {
        if (p.xz(j)) {
          return p;
        }
      }

      return PauliString::rand(num_qubits, r);
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

    static PauliString basis(uint32_t num_qubits, const std::string& P, uint32_t q) {
      return PauliString::basis(num_qubits, P, q, false);
    }

    PauliString copy() const {
      PauliString p(num_qubits);
      std::copy(bit_string.begin(), bit_string.end(), p.bit_string.begin());
      p.set_r(r());
      return p;
    }

    inline bool operator[](size_t i) const {
      size_t word_ind = i / 32;
      size_t bit_ind = i % 32;
      return (bit_string[word_ind] >> bit_ind) & 1u;
    }

    bool operator==(const PauliString &rhs) const {
      if (num_qubits != rhs.num_qubits) {
        return false;
      }

      if (r() != rhs.r()) {
        return false;
      }

      for (uint32_t i = 0; i < num_qubits; i++) {
        if (x(i) != rhs.x(i)) {
          return false;
        }

        if (z(i) != rhs.z(i)) {
          return false;
        }
      }

      return true;
    }

    bool operator!=(const PauliString &rhs) const { 
      return !(this->operator==(rhs)); 
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

    std::string to_op(uint32_t i) const {
      bool xi = x(i); 
      bool zi = z(i);

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
        s += x(i) ? "1" : "0";
      }

      for (uint32_t i = 0; i < num_qubits; i++) {
        s += z(i) ? "1" : "0";
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
      uint8_t xza = xz(a);
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
      uint8_t xza = xz(a);
      bool xa = (xza >> 0u) & 1u;
      bool za = (xza >> 1u) & 1u;

      bool r = phase;

      set_r(r != (xa && za));
      set_x(a, za);
      set_z(a, xa);
    }

    void cx(uint32_t a, uint32_t b) {
      uint8_t xza = xz(a);
      bool xa = (xza >> 0u) & 1u;
      bool za = (xza >> 1u) & 1u;

      uint8_t xzb = xz(b);
      bool xb = (xzb >> 0u) & 1u;
      bool zb = (xzb >> 1u) & 1u;

      bool r = phase;

      set_r(r != ((xa && zb) && ((xb != za) != true)));
      set_x(b, xa != xb);
      set_z(a, za != zb);
    }

    void cz(uint32_t a, uint32_t b) {
      h(b);
      cx(a, b);
      h(b);
    }

    bool commutes_at(PauliString &p, uint32_t i) const {
      if ((x(i) == p.x(i)) && (z(i) == p.z(i))) { // operators are identical
        return true;
      } else if (!x(i) && !z(i)) { // this is identity
        return true;
      } else if (!p.x(i) && !p.z(i)) { // other is identity
        return true;
      } else {
        return false; 
      }
    }

    bool commutes(PauliString &p) const {
      if (num_qubits != p.num_qubits) {
        throw std::invalid_argument("number of p does not have the same number of qubits.");
      }

      uint32_t anticommuting_indices = 0u;
      for (uint32_t i = 0; i < num_qubits; i++) {
        if (!commutes_at(p, i)) {
          anticommuting_indices++;
        }
      }

      return anticommuting_indices % 2 == 0;
    }

    // Returns the circuit which maps this PauliString onto ZII... if z or XII.. otherwise
    QuantumCircuit reduce(bool z = true) const {
      PauliString p(*this);

      QuantumCircuit circuit;

      if (z) {
        p.h(0);
        circuit.add_gate("h", {0});
      }

      for (uint32_t i = 0; i < num_qubits; i++) {
        if (p.z(i)) {
          if (p.x(i)) {
            p.s(i);
            circuit.add_gate("s", {i});
          } else {
            p.h(i);
            circuit.add_gate("h", {i});
          }
        }
      }

      // Step two
      std::vector<uint32_t> nonzero_idx;
      for (uint32_t i = 0; i < num_qubits; i++) {
        if (p.x(i)) {
          nonzero_idx.push_back(i);
        }
      }
      while (nonzero_idx.size() > 1) {
        for (uint32_t j = 0; j < nonzero_idx.size()/2; j++) {
          uint32_t q1 = nonzero_idx[2*j];
          uint32_t q2 = nonzero_idx[2*j+1];
          p.cx(q1, q2);
          circuit.add_gate("cx", {q1, q2});
        }

        remove_even_indices(nonzero_idx);
      }

      // Step three
      uint32_t ql = nonzero_idx[0];
      if (ql != 0) {
        for (uint32_t i = 0; i < num_qubits; i++) {
          if (p.x(i)) {
            p.cx(0, ql);
            p.cx(ql, 0);
            p.cx(0, ql);

            circuit.add_gate("cx", {0, ql});
            circuit.add_gate("cx", {ql, 0});
            circuit.add_gate("cx", {0, ql});

            break;
          }
        }
      }

      if (p.r()) {
        // Apply Y gate to tableau
        p.h(0);
        p.s(0);
        p.s(0);
        p.h(0);
        p.s(0);
        p.s(0);

        circuit.add_gate("h", {0});
        circuit.add_gate("s", {0});
        circuit.add_gate("s", {0});
        circuit.add_gate("h", {0});
        circuit.add_gate("s", {0});
        circuit.add_gate("s", {0});
      }

      if (z) {
        // tableau is discarded after function exits, so no need to apply it here. Just add to circuit.
        circuit.add_gate("h", {0});
      }

      return circuit;
    }

    // Returns the circuit which maps this PauliString onto p
    QuantumCircuit transform(PauliString const &p) const {
      QuantumCircuit c1 = reduce();
      QuantumCircuit c2 = p.reduce().adjoint();

      c1.append(c2);

      return c1;
    }

    // It is slightly faster (~20-30%) to query both the x and z bits at a given site
    // at the same time, storing them in the first two bits of the return value.
    inline uint8_t xz(uint32_t i) const {
      uint32_t word = bit_string[i / 16u];
      uint32_t bit_ind = 2u*(i % 16u);

      return 0u | (((word >> bit_ind) & 3u) << 0u);
    }

    inline bool x(uint32_t i) const { 
      uint32_t word = bit_string[i / 16u];
      uint32_t bit_ind = 2u*(i % 16u);
      return (word >> bit_ind) & 1u;
    }

    inline bool z(uint32_t i) const { 
      uint32_t word = bit_string[i / 16u];
      uint32_t bit_ind = 2u*(i % 16u) + 1u;
      return (word >> bit_ind) & 1u; 
    }

    inline bool r() const { 
      return phase; 
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

