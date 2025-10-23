#include "../src/weightstorage.h"
#include "test_common.h"

#include <ctime>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

fs::path createTempDir(std::string caseName)
{
    std::time_t time = std::time({});
    char timeString[std::size("yyyymmdd-hhmmss")];
    std::strftime(std::data(timeString), std::size(timeString),
                  "%Y%m%d-%H%M%S", std::gmtime(&time));
    std::string dirName = std::string("testrun_") + timeString + "_" + caseName;
    auto tempDir = fs::temp_directory_path() / dirName;
    fs::create_directory(tempDir);
    return tempDir;
}

void case1()
{
    // prepare
    fvec_t weights(1024);
    for (float weight : weights) {
        ASSERT_EQ(weight, 0.0f, "");
    }

    // when
    fillRandomWeights(weights);
    // then
    for (float weight : weights) {
        ASSERT_NEQ(weight, 0.0f, "");
    }

    // when
    auto tempDir = createTempDir("case1");
    auto weightsFile = tempDir / "weights.data";
    saveWeights(weightsFile, weights);
    // then
    ASSERT_EQ(fs::file_size(weightsFile), weights.size() * sizeof(float), "");

    // prepare
    fvec_t loadedWeights(1024);
    for (float weight : loadedWeights) {
        ASSERT_EQ(weight, 0.0f, "");
    }

    // when
    loadWeights(weightsFile, loadedWeights);
    ASSERT_EQ(weights.size(), loadedWeights.size(), "");
    // then
    for (std::size_t i = 0; i < weights.size(); ++i) {
        EXPECT_EQ(weights[i], loadedWeights[i], std::format("[{}]", i))
    }

    // cleanup
    fs::remove_all(tempDir);
}

int main()
{
    case1();
    std::cout << "All tests passed!" << std::endl;
}
