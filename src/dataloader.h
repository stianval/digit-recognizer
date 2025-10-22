#ifndef DATALOADER_H
#define DATALOADER_H

#include "model.h"
#include <vector>
#include <filesystem>
#include <exception>

class FileSizeError : public std::runtime_error
{
    using runtime_error::runtime_error;
};
class MagicError : public std::runtime_error
{
    using runtime_error::runtime_error;
};

class ImageBank {
public:
    ImageBank(fvec_t data, std::size_t n, std::size_t rows, std::size_t cols);
    const fspan_t at(std::size_t idx);

private:
    fvec_t data_;
public:  // data
    const std::size_t n;
    const std::size_t rows;
    const std::size_t cols;
};

const ImageBank loadImages(std::filesystem::path path);
const std::vector<char> loadLabels(std::filesystem::path path);

#endif  // DATALOADER_H
