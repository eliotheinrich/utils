#include "tests.hpp"
#include "LinearCode.h"

ParityCheckMatrix hamming_code() {
  return BinaryMatrix({
    {1,0,1,0,1,0,1},
    {0,1,1,0,0,1,1},
    {0,0,0,1,1,1,1}
  });
}

BitString random_codeword(const GeneratorMatrix& G) {
  size_t num_generators = G.get_num_rows();
  size_t num_bits = G.get_num_cols();
  BitString u = BitString::random(num_generators);

  // Compute x = uG
  BitString x(num_bits);
  for (size_t j = 0; j < num_bits; ++j) {
    bool acc = 0;
    for (size_t i = 0; i < num_generators; ++i) {
      acc ^= (u[i] & G.get(i, j));
    }
    x[j] = acc;
  }

  return x;
}

bool test_generator_parity_roundtrip() {
  ParityCheckMatrix H = hamming_code();

  const size_t n = H.get_num_cols();
  const size_t m = H.get_num_rows();
  const size_t k = n - m;

  GeneratorMatrix G = H.to_generator_matrix();

  ASSERT(G.get_num_rows() == k && G.get_num_cols() == n);

  // 3. Verify H * G^T = 0 (row-by-row)
  for (size_t i = 0; i < k; ++i) {
    BitString g = G.row(i);
    BitString s = H.multiply(g);
    for (size_t j = 0; j < s.get_num_bits(); ++j) {
      ASSERT(!s[j], fmt::format("HG^T != 0 for generator row {}\n", i));
    }
  }

  // 4. Convert G -> H2
  ParityCheckMatrix H2 = G.to_parity_check_matrix();

  ASSERT(H2.get_num_cols() == n);

  // 5. Verify G * H2^T = 0 (row-by-row)
  for (size_t i = 0; i < H2.get_num_rows(); ++i) {
    BitString h = H2.row(i);
    BitString s = G.multiply(h);
    for (size_t j = 0; j < s.get_num_bits(); ++j) {
      ASSERT(!s[j], fmt::format("GH2^T != 0 for generator row {}\n", i));
    }
  }

  // 6. Sample random codewords and test syndromes
  for (size_t trial = 0; trial < 20; ++trial) {
    BitString x = random_codeword(G);

    BitString s1 = H.BinaryMatrixBase::multiply(x);
    for (size_t j = 0; j < s1.get_num_bits(); ++j) {
      ASSERT(!s1[j], "Random codeword has nonzero syndrome under H\n");
    }

    BitString s2 = H2.BinaryMatrixBase::multiply(x);
    for (size_t j = 0; j < s2.get_num_bits(); ++j) {
      ASSERT(!s2[j], "Random codeword has nonzero syndrome under H2\n");
    }
  }

  std::cout << fmt::format("H = \n{}\nH2 = \n{}\nG = \n{}\n", H.to_string(), H2.to_string(), G.to_string());

  ASSERT(H.congruent(G));

  return true;
}

bool test_tanner_graph() {
  size_t num_qubits = 9;
  size_t num_checks = 8; 
  ParityCheckMatrix matrix(num_checks, 2*num_qubits);
  matrix[0][1] = 1; matrix[0][2] = 1;
  matrix[1][0] = 1; matrix[1][1] = 1; matrix[1][3] = 1; matrix[1][4] = 1;
  matrix[2][4] = 1; matrix[2][5] = 1; matrix[2][7] = 1; matrix[2][8] = 1;
  matrix[3][6] = 1; matrix[3][7] = 1;
  matrix[4][0 + num_qubits] = 1; matrix[4][3 + num_qubits] = 1;
  matrix[5][3 + num_qubits] = 1; matrix[5][4 + num_qubits] = 1; matrix[5][6 + num_qubits] = 1; matrix[5][7 + num_qubits] = 1;
  matrix[6][1 + num_qubits] = 1; matrix[6][2 + num_qubits] = 1; matrix[6][4 + num_qubits] = 1; matrix[6][6 + num_qubits] = 1;
  matrix[7][5 + num_qubits] = 1; matrix[7][8 + num_qubits] = 1;

  std::cout << matrix.to_string() << "\n";

  return true;
}

bool test_belief_propagation() {
  ParityCheckMatrix H = hamming_code();
  GeneratorMatrix G = H.to_generator_matrix();
  size_t num_bits = H.get_num_cols();
  size_t num_checks = H.get_num_rows();

  double error_prob = 0.1;
  std::vector<double> prior_probs(num_bits, error_prob);

  for (size_t i = 0; i < 100; ++i) {
    BitString message = random_codeword(G);
    BitString initial_syndrome = H.multiply(message);
    for (size_t k = 0; k < initial_syndrome.get_num_bits(); ++k) {
      ASSERT(!initial_syndrome[k]);
    } 

    // Introduce some errors
    BitString transmitted = message;
    for (size_t j = 0; j < num_bits; ++j) {
      if (randf() < prior_probs[j]) {
        std::cout << fmt::format("Error at {}\n", j);

        transmitted[j] = !transmitted[j];
      }
    }

    BitString syndrome = H.BinaryMatrixBase::multiply(transmitted);
    std::cout << fmt::format("Transmitted = {}, syndrome = {}\n", transmitted, syndrome);

    // Do belief propagation decoding
    std::vector<double> probs;
    size_t num_iterations = 100;
    BitString decoded_errors = belief_propagation(H, syndrome, prior_probs, num_iterations, probs).value();

    // Check that decoded_bits matches transmitted
    BitString recovered_bits = transmitted;
    for (size_t j = 0; j < num_bits; ++j) {
      if (decoded_errors[j]) {
        recovered_bits[j] = !recovered_bits[j];
      }
    }
    
    BitString original_syndrome = H.BinaryMatrixBase::multiply(message);
    BitString recovered_syndrome = H.BinaryMatrixBase::multiply(recovered_bits);
    ASSERT(original_syndrome == recovered_syndrome, fmt::format("Failed to decode. message: {}, transmitted: {}, decoded errors: {}, original syndrome: {}, decoded syndrome: {}\n", message, transmitted, decoded_errors, original_syndrome, recovered_syndrome));
  }

  return true;
}

int main(int argc, char *argv[]) {
  std::map<std::string, TestResult> tests;
  std::set<std::string> test_names;

  bool run_all = (argc == 1);

  if (!run_all) {
    for (size_t i = 1; i < argc; i++) {
      test_names.insert(argv[i]);
    }
  }

  ADD_TEST(test_generator_parity_roundtrip);
  ADD_TEST(test_belief_propagation);
  ADD_TEST(test_tanner_graph);

  constexpr char green[] = "\033[1;32m";
  constexpr char black[] = "\033[0m";
  constexpr char red[] = "\033[1;31m";

  auto test_passed_str = [&](bool passed) {
    std::stringstream stream;
    if (passed) {
      stream << green << "PASSED" << black;
    } else {
      stream << red << "FAILED" << black;
    }
    
    return stream.str();
  };

  if (tests.size() == 0) {
    std::cout << "No tests to run.\n";
  } else {
    double total_duration = 0.0;
    for (const auto& [name, result] : tests) {
      auto [passed, duration] = result;
      std::cout << fmt::format("{:>40}: {} ({:.2f} seconds)\n", name, test_passed_str(passed), duration/1e6);
      total_duration += duration;
    }

    std::cout << fmt::format("Total duration: {:.2f} seconds\n", total_duration/1e6);
  }
}
