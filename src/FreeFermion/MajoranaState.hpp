#pragma once

#include "FreeFermionState.hpp"

// Following https://arxiv.org/pdf/1112.2184
class MajoranaState : public FreeFermionState {
  private:
    Eigen::MatrixXd M;

    void particles_at(const Qubits& sites) {
      M = Eigen::MatrixXd::Zero(2*L, 2*L);
      for (size_t i = 0; i < L; i++) {
        M(2*i, 2*i+1) =  1.0;
        M(2*i+1, 2*i) = -1.0;
      }

      for (const auto& q : sites) {
        M(2*q, 2*q+1) = -1.0;
        M(2*q+1, 2*q) =  1.0;
      }
    }

  public:
    MajoranaState(uint32_t L, std::optional<Qubits> sites) : FreeFermionState(L) {
      if (sites) {
        particles_at(sites.value());
      } else {
        particles_at({});
      }
    }

    MajoranaState(const MajoranaState& other)
      : FreeFermionState(other), M(other.M) { }

    std::vector<double> williamson_eigenvalues(const Eigen::MatrixXd& A) const {
      Eigen::ComplexEigenSolver<Eigen::MatrixXd> ces;
      ces.compute(A);

      Eigen::VectorXcd eigvals = ces.eigenvalues();

      std::vector<double> williamson_eigs;
      for (int i = 0; i < eigvals.size(); ++i) {
        std::complex<double> lambda = eigvals[i];
        if (std::abs(std::real(lambda)) < 1e-10) {
          williamson_eigs.push_back(-std::imag(lambda));
        }
      }

      std::sort(williamson_eigs.begin(), williamson_eigs.end());

      size_t L = williamson_eigs.size();
      return std::vector<double>(williamson_eigs.begin() + L/2, williamson_eigs.end());
    }

    Eigen::MatrixXd prepare_hamiltonian(const FreeFermionHamiltonian& H) const {
      Eigen::MatrixXd Hm = Eigen::MatrixXd::Zero(2*L, 2*L);
      for (const auto& term : H.terms) {
        Hm(2*term.i, 2*term.j+1) =  term.a;
        Hm(2*term.j, 2*term.i+1) =  term.a;
        Hm(2*term.i+1, 2*term.j) = -term.a;
        Hm(2*term.j+1, 2*term.i) = -term.a;
      }

      for (const auto& term : H.nc_terms) {
        Hm(2*term.i, 2*term.j+1) =  term.a;
        Hm(2*term.j, 2*term.i+1) = -term.a;
        Hm(2*term.i+1, 2*term.j) =  term.a;
        Hm(2*term.j+1, 2*term.i) = -term.a;
      }

      return Hm;
    }

    void evolve(const Eigen::MatrixXd& U) {
      M = U * M * U.adjoint();
    }

    virtual void evolve_hamiltonian(const FreeFermionHamiltonian& H, double t=1.0) override {
      Eigen::MatrixXd U = (t * prepare_hamiltonian(H)).exp();
      evolve(U);
    }

    void forced_majorana_measurement(size_t i, bool outcome) {
      double s = outcome ? -1 : 1;

      Eigen::MatrixXd K = Eigen::MatrixXd::Zero(2*L, 2*L);
      K(i, i + 1) = 1;
      K(i + 1, i) = -1;

      Eigen::MatrixXd Lm = M + s * M * K * M / (1 + s * M(i, i + 1));

      M = Lm;
      M.row(i)     = s * K.row(i);
      M.row(i + 1) = s * K.row(i + 1);
      M.col(i)     = s * K.col(i);
      M.col(i + 1) = s * K.col(i + 1);
    }

    // According to https://arxiv.org/pdf/2210.05681
    virtual void forced_projective_measurement(size_t i, bool outcome) override {
      forced_majorana_measurement(2*i, outcome);
    }

    virtual double occupation(size_t i) const override {
      return (1.0 - M(2*i, 2*i + 1)) / 2.0;
    }

    // TODO fix (broken)
    virtual double entanglement(const QubitSupport& support, uint32_t index) override {
      Qubits qubits = to_qubits(support);
      size_t N = qubits.size();
      if (N == 0) {
        return 0.0;
      }

      if (N > L/2) {
        auto _support = support_complement(support, L);
        return entanglement(_support, index);
      }

      std::vector<int> sites(2*N);
      for (size_t i = 0; i < N; i++) {
        uint32_t q = qubits[i];
        sites[2*i]   = 2*q;
        sites[2*i+1] = 2*q+1;
      }
      Eigen::VectorXi indices = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(sites.data(), sites.size());
      Eigen::MatrixXd MA = M(indices, indices);
      std::vector<double> eigenvalues = williamson_eigenvalues(MA);

      double S;
      if (index == 1) {
        S = 0.0;
        for (const auto eig : eigenvalues) {
          double lm = (1 - eig)/2;
          double lp = (1 + eig)/2;
          S -= lm * std::log(lm) + lp * std::log(lp);
        }
      } else if (index == 2) {
        S = N * std::log(2);
        for (const auto eig : eigenvalues) {
          S -= std::log(1 + eig*eig);
        }
      }

      return S;
    }

    virtual std::string to_string() const {
      std::stringstream s;
      s << M;
      return s.str();
    }
};

class ExtendedMajoranaState : public FreeFermionState {
  private:
    uint32_t L;
    using State = std::pair<double, MajoranaState>;
    std::vector<State> states;


  public:
    ExtendedMajoranaState(uint32_t L, std::optional<Qubits> sites) : FreeFermionState(L) {
      states = {State{1.0, MajoranaState(L, sites)}};
    }

    virtual void evolve_hamiltonian(const FreeFermionHamiltonian& H, double t=1.0) override {
      for (auto& [amplitude, state] : states) {
        state.evolve_hamiltonian(H, t);
      }
    }

    virtual void forced_projective_measurement(size_t i, bool outcome) override {
      for (auto& [amplitude, state] : states) {
        state.projective_measurement(i, outcome);
      }
    }

    virtual std::string to_string() const override {
      std::stringstream s;
      for (auto& [amplitude, state] : states) {
        s << fmt::format("{:.5f}: \n{}\n\n", amplitude, state.to_string());
      }
      return s.str();
    }

    virtual double occupation(size_t i) const override {
      std::cout << fmt::format("Calculating occupation({})\n", i);
      double n = 0.0;
      for (const auto& [amplitude, state] : states) {
        std::cout << fmt::format("Adding {:.5f} * {:.5f}\n", amplitude, state.occupation(i));
        n += amplitude * state.occupation(i);
      }
      
      return n;
    }

    void interaction(size_t i) {
      if (i > L - 1) {
        throw std::runtime_error("Can only apply a measurement on qubits i < num_qubits - 1.");
      }

      std::vector<State> new_states;

      for (const auto& [amplitude, state] : states) {
        MajoranaState state1 = state;
        state1.forced_projective_measurement(i, true);
        state1.forced_projective_measurement(i, true);
        MajoranaState state2 = state;
        state2.forced_projective_measurement(i, false);
        state2.forced_projective_measurement(i, false);

        new_states.push_back({amplitude, state1});
        new_states.push_back({amplitude, state2});
      }

      states = new_states;
    }

    virtual double entanglement(const QubitSupport& support, uint32_t index) override {
      throw std::runtime_error("entanglement on implemented for ExtendedMajoranaState");
    }

};
