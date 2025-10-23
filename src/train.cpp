#include "model.h"
#include "weightstorage.h"

#include <format>
#include <iostream>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

const int g_defaultMiniStep = 10;

struct ProgArgs {
    fs::path weightsIn;
    fs::path weightsOut;
    fs::path trainingDir;
    int miniStep = 10;
};

void printHelp(const char * progName)
{
    std::cerr << std::format("Usage: {} <weights-in> <weights-out> [training-dir] [mini-step={}]", progName, g_defaultMiniStep);
    std::exit(EXIT_FAILURE);
}

int main(int argc, const char * argv[])
{
    ProgArgs args;
    if (argc < 3) {
        printHelp(argv[0]);
    }
    args.weightsIn = argv[1];
    args.weightsOut = argv[2];

    if (argc >= 4) {
        args.trainingDir = argv[3];
    }

    if (argc >= 5) {
        args.miniStep = std::stoi(argv[4]);
    }

    bool createRandomWeights = false;
    if (args.weightsIn == "-") {
        createRandomWeights = true;
    } else if (!fs::exists(args.weightsIn)) {
        std::cerr << std::format("'{}' does not exist\n", std::string(args.weightsIn));
        return EXIT_FAILURE;
    }

    if (fs::exists(args.weightsOut)) {
        std::cerr << std::format("'{}' already exists\n", std::string(args.weightsOut));
        return EXIT_FAILURE;
    }

    bool skipTraining = false;
    if (args.trainingDir.empty()) {
        skipTraining = true;
    } else if (!fs::exists(args.trainingDir)) {
        std::cerr << std::format("'{}' does not exist\n", std::string(args.trainingDir));
        return EXIT_FAILURE;
    } else if (!fs::is_directory(args.trainingDir)) {
        std::cerr << std::format("'{}' is not a directory\n", std::string(args.trainingDir));
        return EXIT_FAILURE;
    }

    EmptyModel emptyModel;
    ModelBuilder modelBuilder(emptyModel, 256);
    modelBuilder.addLayer(16);
    modelBuilder.addLayer(16);
    modelBuilder.addLayer(10);

    fvec_t weights(modelBuilder.size(), 0.0f);
    if (createRandomWeights) {
        fillRandomWeights(weights);
    } else {
        loadWeights(args.weightsIn, weights);
    }

    if (!skipTraining) {
    }

    saveWeights(args.weightsOut, weights);
}
