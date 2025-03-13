#pragma once

#include <iostream>
#include <bitset>

namespace quantumstate_utils {
	static inline bool print_congruence(uint32_t z1, uint32_t z2, const std::vector<uint32_t>& pos, bool outcome) {
		std::bitset<8> b1(z1);
		std::bitset<8> b2(z2);
		if (outcome) {
			std::cout << b1 << " and " << b2 << " are congruent at positions ";
		} else {
			std::cout << b1 << " and " << b2 << " are not congruent at positions ";
		}

		for (auto p : pos) {
			std::cout << p << " ";
		}
		std::cout << "\n";

		return outcome;
	}

	static inline bool bits_congruent(uint32_t z1, uint32_t z2, const std::vector<uint32_t>& pos) {
		for (uint32_t j = 0; j < pos.size(); j++) {
			if (((z2 >> j) & 1) != ((z1 >> pos[j]) & 1)) {
				return false;
			}
		}

		return true;
	}

	static inline uint32_t reduce_bits(uint32_t a, const std::vector<uint32_t>& v) {
		uint32_t b = 0;

		for (size_t i = 0; i < v.size(); i++) {
			// Get the ith bit of a
			int a_bit = (a >> v[i]) & 1;

			// Set the ith bit of b based on a_bit
			b |= (a_bit << i);
		}

		return b;
	}

  static inline std::vector<bool> to_bits(uint32_t z, size_t num_bits) {
    std::vector<bool> bits(num_bits);
    for (size_t i = 0; i < num_bits; i++) {
      bits[i] = (z >> i) & 1u;
    }
    return bits;
  }

	inline uint32_t set_bit(uint32_t b, uint32_t j, uint32_t a, uint32_t i) {
		uint32_t x = (a >> i) & 1u;
		return (b & ~(1u << j)) | (x << j);
	}


	static inline std::string print_binary(uint32_t a, uint32_t width=5) {
		std::string s = "";
		for (uint32_t i = 0; i < width; i++) {
			s = std::to_string((a >> i) & 1u) + s;
		}

    std::reverse(s.begin(), s.end());
		return s;
	}

  static inline uint32_t reverse_bits(uint32_t n, int k) {
    uint32_t mask = (1 << k) - 1;
    uint32_t first_k_bits = n & mask;

    uint32_t reversed = 0;
    for (int i = 0; i < k; ++i) {
      if (first_k_bits & (1 << i)) {
        reversed |= (1 << (k - 1 - i));
      }
    }

    return (n & ~mask) | reversed;
  }
}

