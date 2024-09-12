#define FMT_HEADER_ONLY
#include <fmt/format.h>

#include "QuantumState.h"
#include "BinaryPolynomial.h"
#include "Graph.hpp"
#include <iostream>
#include <cassert>

bool test_solve_linear_system() {
  BinaryMatrix M(4, 4);
  M.set(0,0,1);
  M.set(0,1,1);
  M.set(0,2,1);
  M.set(0,3,1);

  M.set(1,0,1);
  M.set(1,1,1);

  M.set(2,0,1);
  M.set(2,2,1);

  M.set(3,0,1);
  std::vector<bool> v{0, 0, 0, 1};
  auto x = M.solve_linear_system(v);
  std::vector<bool> correct{1, 1, 1, 1};

  if (x != correct) {
    return false;
  }

  return true;
}

bool test_binary_polynomial() {
  BinaryPolynomial poly(5);
  poly.add_term(1, 2);
  poly.add_term(1);
  auto inds = std::vector<size_t>{};
  poly.add_term(inds);

  return true;
}


void random_binary_matrix(std::shared_ptr<BinaryMatrixBase> A, int s) {
  thread_local std::minstd_rand rng;
  rng.seed(s);
  for (size_t i = 0; i < A->num_rows; i++) {
    for (size_t j = 0; j < A->num_cols; j++) {
      if (rng() % 2) {
        A->set(i, j, 1);
      }
    }
  }
}

bool test_binary_matrix() {
  size_t v = 3;
  std::shared_ptr<SparseBinaryMatrix> M1 = std::make_shared<SparseBinaryMatrix>(v, v);
  std::shared_ptr<BinaryMatrix> M2 = std::make_shared<BinaryMatrix>(v, v);

  for (int i = 0; i < 100; i++) {
    random_binary_matrix(M1, i);
    random_binary_matrix(M2, i);

    int r1 = M1->rank();
    int r2 = M2->rank();

    if (r1 != r2) {
      std::cout << "M1 = \n" << M1->to_string() << "\n\n";
      std::cout << "M2 = \n" << M2->to_string() << "\n\n";

      std::cout << "r1 = " << r1 << ", r2 = " << r2 << "\n\n";
      return false;
    }
  }

  return true;
}

bool test_generator_matrix() {
  std::vector<BinaryMatrix> test_cases;
  test_cases.push_back(BinaryMatrix(3, 5));
  //test_cases.push_back(std::make_shared<SparseBinaryMatrix>(3, 5));

  for (auto A : test_cases) {
    A.set(0, 0, 1);
    A.set(0, 1, 1);
    A.set(0, 2, 1);

    A.set(1, 1, 1);
    A.set(1, 3, 1);

    A.set(2, 0, 1);
    A.set(2, 4, 1);

  

    auto G = ParityCheckMatrix(A).to_generator_matrix();
    auto H = G.to_parity_check_matrix();

    if (!(G.congruent(H) && H.congruent(G))) {
      std::cout << "A = \n" << A.to_string() << std::endl;
      std::cout << "G = \n" << G.to_string() << std::endl;
      std::cout << "H = \n" << H.to_string() << std::endl;
      return false;
    }
  }

  return true;
}

bool test_random_regular_graph() {
  for (size_t n = 3; n < 10; n++) {
    for (size_t k = 1; k < n/2; k++) {
      if (n * k % 2) {
        // Algorithm not defined in this case
        continue;
      }

      Graph<int, int> g = Graph<int, int>::random_regular_graph(n, k);
      for (size_t j = 0; j < g.num_vertices; j++) {
        if (g.degree(j) != k) {
          std::cout << "Degree violation.\n";
          std::cout << g.to_string() << std::endl;
          return false;
        }

        if (g.contains_edge(j, j)) {
          std::cout << "Edge violation.\n";
          std::cout << g.to_string() << std::endl;
          return false;
        }
      }
    }
  }

  return true;
}

bool test_parity_check_reduction() {
  size_t num_runs = 100;
  thread_local std::random_device rd;
  std::minstd_rand rng(rd());
  for (size_t i = 0; i < num_runs; i++) {
    size_t num_cols = rng() % 20;
    size_t num_rows = num_cols + 5;
    ParityCheckMatrix P(0, num_cols);

    for (size_t j = 0; j < num_rows; j++) {
      Bitstring b = Bitstring::random(num_cols, rng);
      P.append_row(b);
    }

    uint32_t rank = P.rank();
    P.reduce();

    if (rank != P.num_rows) {
      std::cout << P.to_string() << std::endl;
      return false;
    }
  }

  return true;
}

bool test_leaf_removal() {
  ParityCheckMatrix H(3, 5);

  H.set(0, 0, 1);
  H.set(0, 1, 1);
  H.set(0, 3, 1);

  H.set(1, 1, 1);
  H.set(1, 2, 1);
  H.set(1, 3, 1);

  H.set(2, 2, 1);
  H.set(2, 3, 1);
  H.set(2, 4, 1);

  std::minstd_rand rng(5);
  auto result = H.leaf_removal_iteration(rng);

  // TODO rigorous test

  return true;
}

bool test_mps() {
  size_t num_qubits = 6;
  size_t bond_dimension = 12;

  thread_local std::random_device gen;
  std::minstd_rand rng(gen());

  MatrixProductState mps(num_qubits, bond_dimension);
  QuantumCircuit qc(num_qubits);
  qc.add_gate("h", {0});
  qc.add_gate("h", {1});
  
  qc.append(random_clifford(2, rng), {0, 1});
  qc.append(random_clifford(2, rng), {2, 3});
  qc.append(random_clifford(2, rng), {4, 5});

  qc.append(random_clifford(2, rng), {1, 2});
  qc.append(random_clifford(2, rng), {3, 4});

  qc.append(random_clifford(2, rng), {0, 1});
  qc.append(random_clifford(2, rng), {2, 3});
  qc.append(random_clifford(2, rng), {4, 5});
  mps.evolve(qc);

  double d = mps.stabilizer_renyi_entropy(2, 1000);
  std::cout << fmt::format("d = {}\n", d);

  Statevector s(mps);
  std::cout << s.to_string() << std::endl;


  return true;
}

int main() {
  assert(test_solve_linear_system());
  assert(test_binary_polynomial());
  assert(test_binary_matrix());
  assert(test_generator_matrix());
  assert(test_random_regular_graph());
  assert(test_parity_check_reduction());
  assert(test_leaf_removal());
  assert(test_mps());
}
