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

static inline bool is_hermitian(const Eigen::MatrixXcd& H) {
  return H.isApprox(H.adjoint());
}

static inline bool is_antisymmetric(const Eigen::MatrixXcd& A) {
  return A.isApprox(-A.transpose());
}

static inline bool is_unitary(const Eigen::MatrixXcd& U) {
  Eigen::MatrixXcd I = Eigen::MatrixXcd::Identity(U.rows(), U.cols());
  return (U.adjoint() * U).isApprox(I);
}

class FreeFermionState : public EntropyState {
  private:
    size_t L;

  public:
    Eigen::MatrixXcd amplitudes;
    FreeFermionState(size_t L) : L(L), EntropyState(L) {
      particles_at({});
    }

    size_t system_size() const {
      return L;
    }

    void particles_at(const std::vector<size_t>& sites) {
      for (auto i : sites) {
        if (i > L) {
          throw std::invalid_argument(fmt::format("Invalid site. Must be within 0 < i < {}", L));
        }
      }

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
      amplitudes.row(i).swap(amplitudes.row(j));
      amplitudes.row(i + L).swap(amplitudes.row(j + L));
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
      for (size_t i = 0; i < N; i++) {
        _sites.push_back(sites[i] + L);
      }
      Eigen::VectorXi indices = Eigen::Map<Eigen::VectorXi, Eigen::Unaligned>(_sites.data(), _sites.size());
      Eigen::MatrixXcd CA1 = C(indices, indices);
      Eigen::MatrixXcd CA2 = Eigen::MatrixXcd::Identity(CA1.rows(), CA1.cols()) - CA1;

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
        return Cn.trace().real()/(2.0*static_cast<double>(1.0 - index));
      }
    }

    bool is_identity(const Eigen::MatrixXcd& A) const {
      size_t r = A.rows();
      size_t c = A.cols();

      if (r != c) {
        throw std::runtime_error("Non-square matrix passed to is_identity.");
      }

      Eigen::MatrixXcd I = Eigen::MatrixXcd::Identity(r, c);

      return (A - I).norm() < 1e-4;
    }

    bool check_orthogonality() const {
      auto A = amplitudes.adjoint() * amplitudes;
      return is_identity(A);
    }

    void orthogonalize() {
      size_t r = amplitudes.rows();
      size_t c = amplitudes.cols();
      Eigen::MatrixXcd Q(r, c);
      for (int i = 0; i < c; i++) {
        Eigen::VectorXcd q = amplitudes.col(i);

        for (int j = 0; j < i; ++j) {
          q -= Q.col(j).adjoint() * amplitudes.col(i) * Q.col(j);
        }

        q.normalize();
        Q.col(i) = q;
      }

      amplitudes = Q;

      //Eigen::JacobiSVD<Eigen::MatrixXcd> svd(amplitudes, Eigen::ComputeThinU);
      //auto U = svd.matrixU();
      //auto D = svd.singularValues();

      //size_t j = 0;
      //for (size_t i = 0; i < L; i++) {
      //  if (std::abs(D(i)) > 1e-6) {
      //    for (size_t k = 0; k < amplitudes.rows(); k++) {
      //      amplitudes(k, j) = U(k, i);
      //    }
      //    j++;
      //  }
      //}

      //for (size_t i = j; i < L; i++) {
      //  for (size_t k = 0; k < amplitudes.rows(); k++) {
      //    amplitudes(k, i) = 0.0;
      //  }
      //}
    }

    Eigen::MatrixXcd prepare_hamiltonian(const Eigen::MatrixXcd& H) const {
      if (H.rows() == L && H.cols() == L) {
        Eigen::MatrixXcd zeros = Eigen::MatrixXcd::Zero(L, L);
        return prepare_hamiltonian(H, zeros);
      }

      if (!is_hermitian(H)) {
        throw std::runtime_error("Passed non-hermitian matrix to prepare_hamiltonian.");
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

      if (!is_hermitian(A) || !is_antisymmetric(B)) {
        throw std::runtime_error("Passed invalid matrices to prepare_hamiltonian.");
      }

      Eigen::MatrixXcd hamiltonian(2*L, 2*L);
      hamiltonian  << A, B,
                      B.adjoint(), -A.transpose();

      return hamiltonian;
    }

    void evolve_hamiltonian(const Eigen::MatrixXcd& H, double t=1.0) {
      auto hamiltonian = prepare_hamiltonian(H);
      auto U = (std::complex<double>(0.0, -t)*hamiltonian).exp();
      evolve(U);
    }

    void evolve(const Eigen::MatrixXcd& U) {
      if (!is_unitary(U)) {
        throw std::runtime_error("Non-unitary matrix passed to evolve.");
      }

      amplitudes = U * amplitudes;
    }

    void assert_ortho() const {
      bool ortho = check_orthogonality();
      if (!ortho) {
        throw std::runtime_error("Not orthogonal!");
      }
    }

    void evolve_hamiltonian(const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B, double t=1.0) {
      auto hamiltonian = prepare_hamiltonian(A, B);
      evolve_hamiltonian(hamiltonian, t);
    }

    void weak_measurement(const Eigen::MatrixXcd& U) {
      amplitudes = U * amplitudes;
      orthogonalize();
    }

    void weak_measurement_hamiltonian(const Eigen::MatrixXcd& H, double beta=1.0) {
      auto hamiltonian = prepare_hamiltonian(H);
      auto U = (std::complex<double>(beta, 0.0)*hamiltonian).exp();
      weak_measurement(U);
    }

    void weak_measurement_hamiltonian(const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B, double beta=1.0) {
      auto hamiltonian = prepare_hamiltonian(A, B);
      weak_measurement_hamiltonian(hamiltonian, beta);
    }

    void forced_projective_measurement(size_t i, bool outcome) {
      if (i < 0 || i > L) {
        throw std::invalid_argument(fmt::format("Invalid qubit measured: {}, L = {}", i, L));
      }

      size_t k = outcome ? (i + L) : i;
      size_t k_ = outcome ? i : (i + L);

      size_t i0;
      double d = 0.0;
      for (size_t j = 0; j < L; j++) {
        double dj = std::abs(amplitudes(k, j));
        if (dj > d) {
          d = dj;
          i0 = j;
        }
      }

      if (!(d > 0)) {
        std::cout << fmt::format("d = {:.5f}\n", d);
        throw std::runtime_error("Found no positive amplitudes to determine i0.");
      }

      for (size_t j = 0; j < L; j++) {
        if (j == i0) {
          continue;
        }

        amplitudes(Eigen::indexing::all, j) = amplitudes(Eigen::indexing::all, j) - amplitudes(k, j)/amplitudes(k, i0) * amplitudes(Eigen::indexing::all, i0);
        amplitudes(k_, j) = 0.0;
      }

      for (size_t j = 0; j < amplitudes.rows(); j++) {
        amplitudes(j, i0) = 0.0;
      }

      amplitudes(k, i0) = 1.0;

      orthogonalize();
    }

    bool projective_measurement(size_t i, double r) {
      double c = occupation(i);
      bool outcome = r < c;
      forced_projective_measurement(i, outcome);
      return outcome;
    }

    double num_particles() const {
      auto C = correlation_matrix();
      return C.trace().real();
    }

    double num_real_particles() const {
      auto C = correlation_matrix().block(L, L, L, L);
      return C.trace().real();
    }

    Eigen::MatrixXcd correlation_matrix() const {
      return amplitudes * amplitudes.adjoint();
    }

    double occupation(size_t i) const {
      double d = 0.0;

      for (size_t j = 0; j < L; j++) {
        auto c = std::abs(amplitudes(i + L, j));
        d += c*c;
      }

      return d;
    }

    std::vector<double> occupation() const {
      auto C = correlation_matrix();
      std::vector<double> n(L);

      for (size_t i = 0; i < L; i++) {
        n[i] = std::abs(C(i + L, i + L));
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
    FreeFermionSimulator(dataframe::ExperimentParams& params, uint32_t num_threads) : Simulator(params), sampler(params) {
      L = dataframe::utils::get<int>(params, "system_size");
      sample_correlations = dataframe::utils::get<int>(params, "sample_correlations", 0);
      state = std::make_shared<FreeFermionState>(L);
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

    void add_correlation_samples(dataframe::SampleMap& samples) const {
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

    virtual dataframe::SampleMap take_samples() override {
      dataframe::SampleMap samples;
      sampler.add_samples(samples, state);

      if (sample_correlations) {
        add_correlation_samples(samples);
      }

      dataframe::utils::emplace(samples, "num_particles", state->num_particles());
      dataframe::utils::emplace(samples, "num_real_particles", state->num_real_particles());

      return samples;
    }
};
