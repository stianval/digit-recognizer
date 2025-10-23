#ifndef WEIGHTSTORAGE_H
#define WEIGHTSTORAGE_H

#include "model.h"

#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

void fillRandomWeights(fvec_t & weights);
void saveWeights(const fs::path & path, const fvec_t & weights);
void loadWeights(const fs::path & path, fvec_t & weights);

#endif  // WEIGHTSTORAGE_H
