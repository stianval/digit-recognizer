#ifndef WEIGHTSTORAGE_H
#define WEIGHTSTORAGE_H

#include "model.h"

#include <vector>
#include <filesystem>

void fillRandomWeights(fvec_t & weights);
void saveWeights(const std::filesystem::path & path, const fvec_t & weights);
void loadWeights(const std::filesystem::path & path, fvec_t & weights);

#endif  // WEIGHTSTORAGE_H
