#pragma once

#include "BinaryMatrix.hpp"
#include <optional>

class ParityCheckMatrix;
class GeneratorMatrix;

class ParityCheckMatrix : public BinaryMatrix {
  public:
    ParityCheckMatrix()=default;
    ParityCheckMatrix(size_t num_rows, size_t num_cols);
    ParityCheckMatrix(const BinaryMatrix& other);

    GeneratorMatrix to_generator_matrix(bool inplace=false);

    size_t degree(size_t c) const;
    std::vector<size_t> degree_distribution() const;
    std::pair<std::optional<size_t>, std::vector<size_t>> leaf_removal_iteration(std::minstd_rand& rng);

    bool congruent(const GeneratorMatrix& G) const;
    void reduce();

};

class GeneratorMatrix : public BinaryMatrix {
  public:
    GeneratorMatrix()=default;
    GeneratorMatrix(size_t num_rows, size_t num_cols);
    GeneratorMatrix(const BinaryMatrix& other);

    ParityCheckMatrix to_parity_check_matrix(bool inplace=false);

    uint32_t sym(const std::vector<size_t>& sites1, const std::vector<size_t>& sites2);
    std::vector<uint32_t> generator_locality_samples(const std::vector<size_t>& sites);
    uint32_t generator_locality(const std::vector<size_t>& sites);
    GeneratorMatrix truncate(const std::vector<size_t>& sites) const;

    bool congruent(const ParityCheckMatrix& H) const;
};
