#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <unsupported/Eigen/MatrixFunctions>

#include <Display.h>
#include <EntropyState.hpp>
#include <Samplers.h>
#include <Simulator.hpp>

#include <sstream>
#include <iostream>

#include <fmt/format.h>
#include <fmt/ranges.h>

class FreeFermionState : public EntropyState {
  private:
    bool particles_conserved;
    size_t L;

  public:
    Eigen::MatrixXcd amplitudes;
    FreeFermionState(size_t L, bool particles_conserved) : L(L), particles_conserved(particles_conserved), EntropyState(L) {
      particles_at({});
    }

    size_t system_size() const {
      return L;
    }

    void particles_at(const std::vector<size_t>& sites) {
      //std::cout << fmt::format("PARTICLES AT {}\n", sites);
      for (auto i : sites) {
        if (i > L) {
          throw std::invalid_argument(fmt::format("Invalid site. Must be within 0 < i < {}", L));
        }
      }

      if (particles_conserved) {
        amplitudes = Eigen::MatrixXcd::Zero(L, L);
        for (auto i : sites) {
          amplitudes(i, i) = 1.0;
        }
      } else {
        amplitudes = Eigen::MatrixXcd::Zero(2*L, L);
        std::vector<bool> included(L, false);
        for (auto i : sites) {
          included[i] = true;
        }

        for (size_t i = 0; i < L; i++) {
          if (included[i]) {
            amplitudes(i + L, i) = 1.0;
          } else {
            amplitudes(i, i) = 1.0;
          }
        }
      }
    }

    void single_particle() {
      particles_at({L/2});
    }

    void all_particles() {
      std::vector<size_t> sites(L);
      std::iota(sites.begin(), sites.end(), 0);
      particles_at(sites);
    }

    void checkerboard_particles() {
      std::vector<size_t> sites;
      for (size_t i = 0; i < L; i++) {
        sites.push_back(i);
      }
      particles_at(sites);
    }

    void swap(size_t i, size_t j) {
      //std::cout << "calling swap\n";
      amplitudes.row(i).swap(amplitudes.row(j));
      if (!particles_conserved) {
        amplitudes.row(i + L).swap(amplitudes.row(j + L));
      }
    }

		virtual double entropy(const std::vector<uint32_t> &sites, uint32_t index) override {
      size_t N = sites.size();
      if (N == 0) {
        return 0.0;
      }

      if (N > L/2) {
        std::vector<bool> contains(L, false);
        for (auto i : sites) {
          contains[i] = true;
        }

        std::vector<uint32_t> _sites;
        for (size_t i = 0; i < L; i++) {
          if (!contains[i]) {
            _sites.push_back(i);
          }
        }

        return entropy(_sites, index);
      }

      auto C = correlation_matrix();

      std::vector<int> _sites(sites.begin(), sites.end());
      if (!particles_conserved) {
        for (size_t i = 0; i < N; i++) {
          _sites.push_back(sites[i] + N);
        }
      }
      Eigen::VectorXi indices = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(_sites.data(), _sites.size());
      Eigen::MatrixXcd CA1 = C(indices, indices);

      Eigen::MatrixXcd CA2 = -CA1;

      for (size_t i = 0; i < CA2.rows(); i++) {
        CA2(i, i) = CA2.coeff(i, i) + 1.0;
      }

      if (index == 1) {
        Eigen::MatrixXcd Cn = Eigen::MatrixXcd::Zero(indices.size(), indices.size());
        if (std::abs(CA1.determinant()) > 1e-6) {
          Cn += CA1*CA1.log();
        }

        if (std::abs(CA2.determinant()) > 1e-6) {
          Cn += CA2*CA2.log();
        }

        return -Cn.trace().real();
      } else {
        Eigen::MatrixXcd Cn = (CA1.pow(index) + CA2.pow(index)).log();
        return Cn.trace().real()/static_cast<double>(1.0 - index);
      }
    }

    bool check_orthogonality() const {
      auto A = amplitudes.adjoint()*amplitudes;

      for (size_t i = 0; i < A.rows(); i++) {
        for (size_t j = 0; j < A.cols(); j++) {
          if (i == j) {
            auto a = A(i, i);
            if (std::abs(a - 1.0) > 1e-6 && std::abs(a) > 1e-6) {
              return false;
            }
          } else {
            auto a = A(i, j);
            if (std::abs(a) > 1e-6) {
              return false;
            }
          }
        }
      }

      return true;
    }

    void orthogonalize() {
      Eigen::JacobiSVD<Eigen::MatrixXcd> svd(amplitudes, Eigen::ComputeThinU);
      auto U = svd.matrixU();
      auto D = svd.singularValues();

      size_t j = 0;
      for (size_t i = 0; i < L; i++) {
        if (std::abs(D(i)) > 1e-6) {
          for (size_t k = 0; k < amplitudes.rows(); k++) {
            amplitudes(k, j) = U(k, i);
          }
          j++;
        }
      }

      for (size_t i = j; i < L; i++) {
        for (size_t k = 0; k < amplitudes.rows(); k++) {
          amplitudes(k, i) = 0.0;
        }
      }
    }

    Eigen::MatrixXcd prepare_hamiltonian(const Eigen::MatrixXcd& H) const {
      if (!particles_conserved && H.rows() == L && H.cols() == L) {
        Eigen::MatrixXcd zeros = Eigen::MatrixXcd::Zero(L, L);
        return prepare_hamiltonian(H, zeros);
      }

      size_t K = amplitudes.rows();

      if (H.rows() != K || H.cols() != K) {
        throw std::invalid_argument("Dimension mismatch of provided Hamiltonian.");
      }

      return H;
    }

    Eigen::MatrixXcd prepare_hamiltonian(const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B) const {
      if (A.rows() != L || A.cols() != L || B.rows() != L || B.cols() != L) {
        throw std::invalid_argument("Dimension mismatch of provided Hamiltonian.");
      }

      Eigen::MatrixXcd hamiltonian(2*L, 2*L);
      hamiltonian  << A, B,
                      B.adjoint(), -A.transpose();

      return hamiltonian;
    }

    void evolve(const Eigen::MatrixXcd& H, double t=1.0) {
      auto hamiltonian = prepare_hamiltonian(H);
      auto U = (std::complex<double>(0.0, -t)*hamiltonian).exp();
      amplitudes = U * amplitudes;
    }

    void evolve(const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B, double t=1.0) {
      if (particles_conserved) {
        throw std::invalid_argument("Cannot call evolve(A, B) for particle-conserving simulation.");
      }

      auto hamiltonian = prepare_hamiltonian(A, B);
      evolve(hamiltonian, t);
    }

    void weak_measurement(const Eigen::MatrixXcd& H, double beta=1.0) {
      auto hamiltonian = prepare_hamiltonian(H);
      auto U = (std::complex<double>(beta, 0.0)*hamiltonian).exp();
      amplitudes = U * amplitudes;
      orthogonalize();
    }

    void weak_measurement(const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B, double beta=1.0) {
      if (particles_conserved) {
        throw std::invalid_argument("Cannot call weak_measurement(A, B) for particle-conserving simulation.");
      }

      auto hamiltonian = prepare_hamiltonian(A, B);
      weak_measurement(hamiltonian, beta);
    }

    void forced_projective_measurement(size_t i, bool outcome) {
      if (i < 0 || i > L) {
        throw std::invalid_argument(fmt::format("Invalid qubit measured: {}, L = {}", i, L));
      }

      size_t k = outcome ? i + L : i;
      //std::cout << "from forced_projective_measurement, amplitudes = \n" << amplitudes << "\n";

      size_t i0;
      double d = 0.0;
      for (size_t j = 0; j < L; j++) {
        double dj = std::abs(amplitudes(k, j));
        //std::cout << fmt::format("a({}, {}) = {}\n", k, j, dj);
        if (dj > d) {
          d = dj;
          i0 = j;
        }
      }

      if (!(d > 0)) {
        //std::cout << amplitudes << "\n";
        //std::cout << fmt::format("called forced_projective_measurement({}, {}), c = {}\n", i, outcome, occupation());
        throw std::runtime_error("Found no positive amplitudes to determine i0.");
      }

      for (size_t j = 0; j < L; j++) {
        if (j == i0) {
          continue;
        }

        amplitudes(Eigen::indexing::all, j) = amplitudes(Eigen::indexing::all, j) - amplitudes(k, j)/amplitudes(k, i0) * amplitudes(Eigen::indexing::all, i0);
        amplitudes(i, j) = 0.0;
      }

      for (size_t j = 0; j < amplitudes.rows(); j++) {
        amplitudes(j, i0) = 0.0;
      }

      amplitudes(k, i0) = 1.0;

      orthogonalize();
    }

    bool projective_measurement(size_t i, double r) {
      double c = occupation(i);
      bool outcome = (r < c);
      //std::cout << fmt::format("calling projective measurement: {}, {}, {}\n", r, c, outcome);

      forced_projective_measurement(i, outcome);

      return outcome;
    }

    double num_particles() const {
      auto C = correlation_matrix();
      return C.trace().real();
    }

    Eigen::MatrixXcd correlation_matrix() const {
      return amplitudes * amplitudes.adjoint();
    }

    double occupation(size_t i) const {
      double d = 0.0;
      for (size_t k = 0; k < amplitudes.cols(); k++) {
        auto c = std::abs(amplitudes(i, k));
        d += c*c;
      }

      return d;
    }

    std::vector<double> occupation() const {
      auto C = correlation_matrix();
      //std::cout << "C = \n" << C << "\n";
      std::vector<double> n(L);

      int d = particles_conserved ? 0 : L;
      for (size_t i = 0; i < L; i++) {
        n[i] = std::abs(C(i + d, i + d));
      }

      return n;
    }

    std::string to_string() const {
      std::stringstream s;
      s << amplitudes;
      return s.str();
    }
};

class FreeFermionSimulator : public Simulator {
  private:
    EntropySampler sampler;

    bool sample_correlations;

  protected:
    size_t L;

  public:
    std::shared_ptr<FreeFermionState> state;
    FreeFermionSimulator(dataframe::Params& params, uint32_t num_threads) : Simulator(params), sampler(params) {
      L = dataframe::utils::get<int>(params, "system_size");
      sample_correlations = dataframe::utils::get<int>(params, "sample_correlations", 0);
    }

    void init_fermion_state(bool particles_conserved) {
      state = std::make_shared<FreeFermionState>(L, particles_conserved);
    }

    virtual Texture get_texture() const override {
      Texture texture(L, L);
      auto correlations = state->correlation_matrix();
      double m_r = 0.0;
      double m_i = 0.0;
      for (size_t x = 0; x < L; x++) {
        for (size_t y = 0; y < L; y++) {
          auto c = correlations(x, y);
          auto c_r = std::abs(c.real());
          auto c_i = std::abs(c.imag());

          if (c_r > m_r) {
            m_r = c_r;
          }
          
          if (c_i > m_i) {
            m_i = c_i;
          }
        }
      }

      for (size_t x = 0; x < L; x++) {
        for (size_t y = 0; y < L; y++) {
          auto c = correlations(x, y);
          Color color = {static_cast<float>(std::abs(c.real())/m_r), static_cast<float>(std::abs(c.imag())/m_i), 0.0, 1.0};
          texture.set(x, y, color);
        }
      }

      return texture;
    }

    void add_correlation_samples(dataframe::data_t& samples) const {
      auto C = state->correlation_matrix();
      std::vector<std::vector<double>> correlations(L, std::vector<double>(L));

      for (size_t r = 0; r < L; r++) {
        // Average over space
        for (size_t i = 0; i < L; i++) {
          double c = std::abs(C(i, (i+r)%L));
          correlations[r][i] = c*c;
        }
      }

      dataframe::utils::emplace(samples, "correlations", correlations);
    }

    virtual dataframe::data_t take_samples() override {
      dataframe::data_t samples;
      sampler.add_samples(samples, state);

      if (sample_correlations) {
        add_correlation_samples(samples);
      }

      return samples;
    }
};
