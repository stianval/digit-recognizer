#include "weightstorage.h"

#include <filesystem>
#include <type_traits>
#include <vector>
#include <bit>
#include <fstream>
#include <random>
#include <cstring>
#include <iostream>
#include <exception>

namespace fs = std::filesystem;

static_assert(std::endian::native == std::endian::little, "Requires little endian");
static_assert(std::numeric_limits<float>::is_iec559, "Requires standard float type");

void fillRandomWeights(fvec_t & weights)
{
    std::random_device rnd;
    std::uniform_real_distribution<float> distribution;
    for (float & a : weights) {
        a = distribution(rnd);
    }
}

void saveWeights(const fs::path & path, const fvec_t & weights)
{
    std::vector<char> buf(weights.size() * sizeof(float));
    std::memcpy(buf.data(), weights.data(), buf.size());
    std::ofstream file(path, std::ios_base::out | std::ios_base::binary);
    file.write(buf.data(), buf.size());
}

void loadWeights(const fs::path & path, fvec_t & weights)
{
    std::vector<char> buf(weights.size() * sizeof(float));
    if (fs::file_size(path) != buf.size()) {
	std::cerr << "File size is wrong: " << path << std::endl;
	throw std::runtime_error("File size is wrong");
    }
    std::ifstream file(path, std::ios_base::in | std::ios_base::binary);
    file.read(buf.data(), buf.size());
    std::memcpy(weights.data(), buf.data(), buf.size());
}
