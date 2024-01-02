#pragma once

#include <string>
#include <vector>
#include <random>
#include <variant>
#include <algorithm>

#include "QuantumState.h"

#include <unsupported/Eigen/KroneckerProduct>
#include <Eigen/Dense>

namespace tableau_utils {

    struct sgate { uint32_t q; };
    struct sdgate { uint32_t q; };
    struct hgate { uint32_t q;};
    struct cxgate { uint32_t q1; uint32_t q2; };

    typedef std::variant<sgate, sdgate, hgate, cxgate> Gate;
    typedef std::vector<Gate> Circuit;

    template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
    template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

}

static tableau_utils::Circuit conjugate_circuit(const tableau_utils::Circuit &circuit) {
    tableau_utils::Circuit ncircuit;
    for (auto const &gate : circuit) {
        std::visit(tableau_utils::overloaded{
            [&ncircuit](tableau_utils::sgate s) { ncircuit.push_back(tableau_utils::sdgate{s.q}); },
            [&ncircuit](tableau_utils::sdgate s) { ncircuit.push_back(tableau_utils::sgate{s.q}); },
            [&ncircuit](auto s) { ncircuit.push_back(s); }
        }, gate);
    }

    std::reverse(ncircuit.begin(), ncircuit.end());

    return ncircuit;
}

template <class T>
static void apply_circuit(const tableau_utils::Circuit &circuit, T &state) {
    for (auto const &gate : circuit) {
        std::visit(tableau_utils::overloaded{
                [&state](tableau_utils::sgate s) {  
                    state.s_gate(s.q); 
                },
                [&state](tableau_utils::sdgate s) { 
                    state.sd_gate(s.q);
                },
                [&state](tableau_utils::hgate s) {  
                    state.h_gate(s.q); 
                },
                [&state](tableau_utils::cxgate s) { 
                    state.cx_gate(s.q1, s.q2); 
                }
        }, gate);
    }
}

template <typename T>
static void remove_even_indices(std::vector<T> &v) {
    uint32_t vlen = v.size();
    for (uint32_t i = 0; i < vlen; i++) {
        uint32_t j = vlen - i - 1;
        if ((j % 2)) v.erase(v.begin() + j);
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
                g << 0, -1j, 1j, 0;
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

        void s_gate(uint32_t a) {
            uint8_t xza = xz(a);
            bool xa = (xza >> 0u) & 1u;
            bool za = (xza >> 1u) & 1u;

            bool r = phase;

            set_r(r != (xa && za));
            set_z(a, xa != za);
        }

        void h_gate(uint32_t a) {
            uint8_t xza = xz(a);
            bool xa = (xza >> 0u) & 1u;
            bool za = (xza >> 1u) & 1u;

            bool r = phase;

            set_r(r != (xa && za));
            set_x(a, za);
            set_z(a, xa);
        }

        void cx_gate(uint32_t a, uint32_t b) {
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
        tableau_utils::Circuit reduce(bool z) const;

        // Returns the circuit which maps this PauliString onto p
        tableau_utils::Circuit transform(PauliString const &p) const;

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

        void sd_gate(uint32_t a) {
            s_gate(a);
            s_gate(a);
            s_gate(a);
        }
};

class Tableau {
    private:
        uint32_t num_qubits;
        bool track_destabilizers;

    public:
        std::vector<PauliString> rows;

        Tableau()=default;

        Tableau(uint32_t num_qubits)
         : num_qubits(num_qubits), track_destabilizers(true) {
            rows = std::vector<PauliString>(2*num_qubits + 1, PauliString(num_qubits));
            for (uint32_t i = 0; i < num_qubits; i++) {
                rows[i].set_x(i, true);
                rows[i + num_qubits].set_z(i, true);
            }
        }

        Tableau(uint32_t num_qubits, const std::vector<PauliString>& rows)
         : num_qubits(num_qubits), track_destabilizers(false), rows(rows) {}

        uint32_t num_rows() const { 
            if (track_destabilizers) { 
                return rows.size() - 1; 
            } else {
                return rows.size(); 
            }
        }

        Statevector to_statevector() const {
            Eigen::MatrixXcd dm = Eigen::MatrixXcd::Identity(1u << num_qubits, 1u << num_qubits);
            Eigen::MatrixXcd I = Eigen::MatrixXcd::Identity(1u << num_qubits, 1u << num_qubits);
            
            for (uint32_t i = num_qubits; i < 2*num_qubits; i++) {
                PauliString p = rows[i];
                Eigen::MatrixXcd g = p.to_matrix();
                dm = dm*((I + g)/2.0);
            }

            uint32_t N = 1u << num_qubits;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(dm);
            Eigen::VectorXcd vec = solver.eigenvectors().block(0,N-1,N,1).rowwise().reverse();
            
            return Statevector(vec);

        }

        bool operator==(Tableau& other) {
            if (num_qubits != other.num_qubits) {
                return false;
            }

            rref();
            other.rref();

            uint32_t r1 = track_destabilizers ? num_qubits : 0;
            for (uint32_t i = r1; i < num_rows(); i++) {
                if (r(i) != other.r(i)) {
                    return false;
                }

                for (uint32_t j = 0; j < num_qubits; j++) {
                    if (z(i, j) != other.z(i, j)) {
                        return false;
                    }

                    if (x(i, j) != other.x(i, j)) {
                        return false;
                    }
                }
            }

            return true;
        }

        // Put tableau into reduced row echelon form
        void rref(const std::vector<uint32_t>& sites) {
            uint32_t r1 = track_destabilizers ? num_qubits : 0;
            uint32_t r2 = num_rows();

            uint32_t pivot_row = 0;
            uint32_t row = r1;

            for (uint32_t k = 0; k < 2*sites.size(); k++) {
                uint32_t c = sites[k % sites.size()];
                bool z = k < sites.size();
                bool found_pivot = false;
                for (uint32_t i = row; i < r2; i++) {
                    if ((z && rows[i].z(c)) || (!z && rows[i].x(c))) {
                        pivot_row = i;
                        found_pivot = true;
                        break;
                    }
                }

                if (found_pivot) {
                    std::swap(rows[row], rows[pivot_row]);

                    for (uint32_t i = r1; i < r2; i++) {
                        if (i == row) {
                            continue;
                        }

                        if ((z && rows[i].z(c)) || (!z && rows[i].x(c))) {
                            rowsum(i, row);
                        }
                    }

                    row += 1;
                } else {
                    continue;
                }
            }
        }

        uint32_t rank(const std::vector<uint32_t>& sites) {
            rref(sites);

            uint32_t r1 = track_destabilizers ? num_qubits : 0;
            uint32_t r2 = num_rows();

            uint32_t r = 0;
            for (uint32_t i = r1; i < r2; i++) {
                for (uint32_t j = 0; j < sites.size(); j++) {
                    if (rows[i].x(sites[j]) || rows[i].z(sites[j])) {
                        r++;
                        break;
                    }
                }
            }

            return r;
        }

        void rref() {
            std::vector<uint32_t> qubits(num_qubits);
            std::iota(qubits.begin(), qubits.end(), 0);
            rref(qubits);
        }

        uint32_t rank() {
            std::vector<uint32_t> qubits(num_qubits);
            std::iota(qubits.begin(), qubits.end(), 0);
            return rank(qubits);
        }

        inline void validate_qubit(uint32_t a) const {
            if (!(a >= 0 && a < num_qubits)) {
                std::string error_message = "A gate was applied to qubit " + std::to_string(a) + 
                                            ", which is outside of the allowed range (0, " + std::to_string(num_qubits) + ").";
                throw std::invalid_argument(error_message);
            }
        }

        std::string to_string() const {
            std::string s = "";
            for (uint32_t i = 0; i < num_rows(); i++) {
                s += (i == 0) ? "[" : " ";
                s += rows[i].to_string();
                s += (i == num_rows() - 1) ? "]" : "\n";
            }
            return s;
        }

        std::string to_string_ops() const {
            std::string s = "";
            for (uint32_t i = 0; i < num_rows(); i++) {
                s += (i == 0) ? "[" : " ";
                s += "[" + rows[i].to_string_ops() + "]";
                s += (i == num_rows() - 1) ? "]" : "\n";
            }
            return s + "]";
        }

        int g(uint32_t xz1, uint8_t xz2) {
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

        void rowsum(uint32_t h, uint32_t i) {
            int s = 0;
            
            if (r(i)) { 
                s += 2; 
            }

            if (r(h)) { 
                s += 2; 
            }

            for (uint32_t j = 0; j < num_qubits; j++) {
                s += Tableau::g(rows[i].xz(j), rows[h].xz(j));
            }

            if (s % 4 == 0) {
                set_r(h, false);
            } else if (std::abs(s % 4) == 2) {
                set_r(h, true);
            }

            uint32_t width = rows[h].width;
            for (uint32_t j = 0; j < width; j++) {
                rows[h].bit_string[j] ^= rows[i].bit_string[j];
            }
        }

        void h_gate(uint32_t a) {
            validate_qubit(a);
            for (uint32_t i = 0; i < num_rows(); i++) {
                rows[i].h_gate(a);
            }
        }

        void s_gate(uint32_t a) {
            validate_qubit(a);
            for (uint32_t i = 0; i < num_rows(); i++) {
                rows[i].s_gate(a);
            }
        }

        void cx_gate(uint32_t a, uint32_t b) {
            validate_qubit(a);
            validate_qubit(b);
            for (uint32_t i = 0; i < num_rows(); i++) {
                rows[i].cx_gate(a, b);
            }
        }

        // Returns a pair containing (1) wether the outcome of a measurement on qubit a is deterministic
        // and (2) the index on which the CHP algorithm performs rowsum if the mzr is random
        std::pair<bool, uint32_t> mzr_deterministic(uint32_t a) {
            if (!track_destabilizers) {
                throw std::invalid_argument("Cannot check mzr_deterministic without track_destabilizers.");
            }

            for (uint32_t p = num_qubits; p < 2*num_qubits; p++) {
                // Suitable p identified; outcome is random
                if (x(p, a)) { 
                    return std::pair(false, p);
                }
            }

            // No p found; outcome is deterministic
            return std::pair(true, 0);
        }

        bool mzr(uint32_t a, std::minstd_rand& rng) {
            validate_qubit(a);
            if (!track_destabilizers) {
                throw std::invalid_argument("Cannot mzr without track_destabilizers.");
            }


            auto [deterministic, p] = mzr_deterministic(a);

            if (!deterministic) {
                bool outcome = rng() % 2;
                for (uint32_t i = 0; i < 2*num_qubits; i++) {
                    if (i != p && x(i, a)) {
                        rowsum(i, p);
                    }
                }

                std::swap(rows[p - num_qubits], rows[p]);
                rows[p] = PauliString(num_qubits);

                set_r(p, outcome);
                set_z(p, a, true);

                return outcome;
            } else { // deterministic
                rows[2*num_qubits] = PauliString(num_qubits);
                for (uint32_t i = 0; i < num_qubits; i++) {
                    rowsum(2*num_qubits, i + num_qubits);
                }

                return r(2*num_qubits);
            }
        }

        double sparsity() const {
            float nonzero = 0;
            for (uint32_t i = 0; i < num_rows(); i++) {
                for (uint32_t j = 0; j < num_qubits; j++) {
                    nonzero += rows[i].x(j);
                    nonzero += rows[i].z(j);
                }
            }

            return nonzero/(num_rows()*num_qubits*2);
        }


        inline bool x(uint32_t i, uint32_t j) const { 
            return rows[i].x(j); 
        }

        inline bool z(uint32_t i, uint32_t j) const { 
            return rows[i].z(j); 
        }

        inline bool r(uint32_t i) const { 
            return rows[i].r(); 
        }

        inline void set_x(uint32_t i, uint32_t j, bool v) { 
            rows[i].set_x(j, v); 
        }

        inline void set_z(uint32_t i, uint32_t j, bool v) { 
            rows[i].set_z(j, v); 
        }

        inline void set_r(uint32_t i, bool v) { 
            rows[i].set_r(v); 
        }

        void sd_gate(uint32_t a) {
            s_gate(a);
            s_gate(a);
            s_gate(a);
        }

        void x_gate(uint32_t a) {
            h_gate(a);
            z_gate(a);
            h_gate(a);
        }

        void y_gate(uint32_t a) {
            x_gate(a);
            z_gate(a);
        }

        void z_gate(uint32_t a) {
            s_gate(a);
            s_gate(a);
        }
};
