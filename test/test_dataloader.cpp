#include "../src/dataloader.h"
#include "test_common.h"

#include <filesystem>
#include <cassert>

namespace fs = std::filesystem;

fs::path g_binDir;

void caseHappyImageBank()
{
    fs::path imagesFilePath = g_binDir / "data/mock-images";
    ImageBank images = loadImages(imagesFilePath);
    ASSERT_EQ(images.n, std::size_t(4), "");
    ASSERT_EQ(images.rows, std::size_t(2), "");
    ASSERT_EQ(images.cols, std::size_t(3), "");
    bool passing = true;
    for (std::size_t i = 0; i < images.n; ++i) {
        cfspan_t image = images.at(i);
        for (std::size_t row = 0; row < images.rows; ++row) {
            for (std::size_t col = 0; col < images.cols; ++col) {
                int expectedFromFile = i * images.rows * images.cols + row * images.cols + col + 1;
                passing &= EXPECT_EQ(image[row * images.cols + col],
                        expectedFromFile / 255.0f,
                        std::format("[{}][{}][{}]", i, row, col));
            }
        }
    }
    ASSERT_EQ(passing, true, "");
}

void caseHappyLabel()
{
    fs::path labelsFilePath = g_binDir / "data/mock-labels";
    std::vector<char> labels = loadLabels(labelsFilePath);
    ASSERT_EQ(labels.size(), 4UZ, "");
    bool passing = true;
    for (auto i = 0UZ; i < labels.size(); ++i) {
        passing &= EXPECT_EQ(int(labels[i]), int(i + 1), std::format("i={}", i));
    }
    ASSERT_EQ(passing, true, "");
}

void caseImageMagicOnly()
{
    fs::path imagesFilePath = g_binDir / "data/image-magic-only";
    bool errorThrown = false;
    try {
        ImageBank images = loadImages(imagesFilePath);
    } catch (FileSizeError &) {
        errorThrown = true;
    }
    ASSERT_EQ(errorThrown, true, "");
}

void caseLabelMagicOnly()
{
    fs::path labelsFilePath = g_binDir / "data/label-magic-only";
    bool errorThrown = false;
    try {
        std::vector<char> labels = loadLabels(labelsFilePath);
    } catch (const FileSizeError &) {
        errorThrown = true;
    }
    ASSERT_EQ(errorThrown, true, "");
}

void caseTryLoadLabelsAsImages()
{
    fs::path imagesFilePath = g_binDir / "data/mock-labels";
    bool errorThrown = false;
    try {
        ImageBank images = loadImages(imagesFilePath);
    } catch (const MagicError &) {
        errorThrown = true;
    }
    ASSERT_EQ(errorThrown, true, "");
}

void caseTryLoadImagesAsLabels()
{
    fs::path labelsFilePath = g_binDir / "data/mock-images";
    bool errorThrown = false;
    try {
        std::vector<char> labels = loadLabels(labelsFilePath);
    } catch (const MagicError &) {
        errorThrown = true;
    }
    ASSERT_EQ(errorThrown, true, "");
}

void caseTryLoadEmptyAsImages()
{
    fs::path imagesFilePath = g_binDir / "data/empty";
    bool errorThrown = false;
    try {
        ImageBank images = loadImages(imagesFilePath);
    } catch (const FileSizeError &) {
        errorThrown = true;
    }
    ASSERT_EQ(errorThrown, true, "");
}

void caseTryLoadEmptyAsLabels()
{
    fs::path labelsFilePath = g_binDir / "data/empty";
    bool errorThrown = false;
    try {
        std::vector<char> labels = loadLabels(labelsFilePath);
    } catch (const FileSizeError &) {
        errorThrown = true;
    }
    ASSERT_EQ(errorThrown, true, "");
}

int main(int argc, char *argv[])
{
    assert(argc > 0);
    g_binDir = argv[0];
    g_binDir.remove_filename();

    caseHappyImageBank();
    caseHappyLabel();
    caseImageMagicOnly();
    caseLabelMagicOnly();
    caseTryLoadLabelsAsImages();
    caseTryLoadImagesAsLabels();
    caseTryLoadEmptyAsImages();
    caseTryLoadEmptyAsLabels();
    std::cout << "All tests passed!" << std::endl;
}
