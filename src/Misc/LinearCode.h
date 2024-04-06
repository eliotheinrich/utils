#pragma once

#include "BinaryMatrix.hpp"

class ParityCheckMatrix;
class GeneratorMatrix;

class ParityCheckMatrix : public BinaryMatrix {
  public:
    ParityCheckMatrix()=default;
    ParityCheckMatrix(uint32_t num_rows, uint32_t num_cols);
    ParityCheckMatrix(const BinaryMatrix& other);

    GeneratorMatrix to_generator_matrix(bool inplace=false);

    size_t degree(size_t c) const;
    std::vector<std::vector<size_t>> leaf_removal(size_t num_steps, std::minstd_rand& rng) const;
    bool congruent(const GeneratorMatrix& G) const;
    void reduce();
};

class GeneratorMatrix : public BinaryMatrix {
  public:
    GeneratorMatrix()=default;
    GeneratorMatrix(uint32_t num_rows, uint32_t num_cols);
    GeneratorMatrix(const BinaryMatrix& other);

    ParityCheckMatrix to_parity_check_matrix(bool inplace=false);

    std::vector<uint32_t> generator_locality_samples(const std::vector<size_t>& sites);
    uint32_t generator_locality(const std::vector<size_t>& sites);

    bool congruent(const ParityCheckMatrix& H) const;
};
