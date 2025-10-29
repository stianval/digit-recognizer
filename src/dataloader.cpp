#include "dataloader.h"

#include <fstream>
#include <iostream>


namespace fs = std::filesystem;

namespace {

int readBigEndianInt(std::istream &stream) {
    static_assert(sizeof(int) == 4, "Requires sizeof int to be 4");
    char buf[4];
    stream.read(&buf[0], 4);
    return buf[0] << 24 | buf[1] << 16 | buf[2] << 8 | buf[3];
}

class FileSizeChecker
{
public:
    explicit FileSizeChecker(fs::path path)
        : path_(path)
        , left_(fs::file_size(path))
    {}

    void checkCanRead(std::size_t bytes, bool last);
private:
    fs::path path_;
    std::size_t left_;
};

void FileSizeChecker::checkCanRead(std::size_t bytes, bool last)
{
    if (left_ < bytes || (last && left_ != bytes)) {
	std::cerr << "File size is wrong: " << path_ << std::endl;
	throw FileSizeError("File size is wrong");
    }
    left_ -= bytes;
}

}

ImageBank::ImageBank(fvec_t data, std::size_t n, std::size_t rows, std::size_t cols)
    : data_(std::move(data))
    , n(n)
    , rows(rows)
    , cols(cols)
{ ; }

cfspan_t ImageBank::at(std::size_t idx) const
{
    auto imageSize = rows * cols;
    return cfspan_t(data_.data() + idx * imageSize, imageSize);
}

const ImageBank loadImages(fs::path path)
{
    char magic[4];
    FileSizeChecker fileSizeChecker(path);
    fileSizeChecker.checkCanRead(sizeof(magic), false);

    std::ifstream file(path, std::ios_base::in | std::ios_base::binary);
    file.read(&magic[0], sizeof(magic));
    if (magic[0] != 0 || magic[1] != 0 || magic[2] != 8 || magic[3] != 3) {
        std::cerr << "Wrong magic: " << path << std::endl;
        throw MagicError("Wrong magic");
    }

    fileSizeChecker.checkCanRead(3 * sizeof(int), false);
    int n = readBigEndianInt(file);
    int rows = readBigEndianInt(file);
    int cols = readBigEndianInt(file);

    std::size_t dataSize = n * rows * cols;
    fileSizeChecker.checkCanRead(dataSize, true);
    std::vector<char> fileData(dataSize);
    fvec_t data(dataSize);
    file.read(fileData.data(), dataSize);
    for (std::size_t i = 0; i < data.size(); ++i) {
        data[i] = fileData[i] / 255.0f;
    }
    return ImageBank(std::move(data), n, rows, cols);
}

const std::vector<char> loadLabels(fs::path path)
{
    char magic[4];
    FileSizeChecker fileSizeChecker(path);
    fileSizeChecker.checkCanRead(sizeof(magic), false);

    std::ifstream file(path, std::ios_base::in | std::ios_base::binary);
    file.read(&magic[0], sizeof(magic));
    if (magic[0] != 0 || magic[1] != 0 || magic[2] != 8 || magic[3] != 1) {
        std::cerr << "Wrong magic: " << path << std::endl;
        throw MagicError("Wrong magic");
    }

    fileSizeChecker.checkCanRead(sizeof(int), false);
    int n = readBigEndianInt(file);

    fileSizeChecker.checkCanRead(n, false);
    std::vector<char> labels(n);
    file.read(labels.data(), n);
    return labels;
}
