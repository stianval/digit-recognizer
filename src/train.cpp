#include "model.h"
#include "weightstorage.h"
#include "dataloader.h"

#include <format>
#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <algorithm>
#include <numeric>
#include <cassert>

namespace fs = std::filesystem;

const int g_defaultMiniStep = 10;

struct ProgArgs {
    fs::path weightsIn;
    fs::path weightsOut;
    fs::path imageFile;
    fs::path labelFile;
    int miniStep = 10;
};

void printHelp(const char * progName)
{
    std::cerr << std::format("Usage: {} <weights-in> <weights-out> [<image-file> <label-file> [mini-step={}]]", progName, g_defaultMiniStep);
    std::exit(EXIT_FAILURE);
}

cfspan_t getTarget(int digit)
{
    static const fvec_t targetVec = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        1.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    return cfspan_t(targetVec.begin() + 9 - digit, 10);
}

void performMinistep(Model & model, const ImageBank & imageBank, const std::vector<char> & labels, std::span<const std::size_t> order)
{
    fvec_t dw(model.size(), 0.0f);
    for (std::size_t index : order) {
        auto input = imageBank.at(index);
        fvec_t activations = model.calculateActivations(input);
        model.backPropagate(dw, activations, getTarget(labels[index]), input);
    }
    for (float & value : dw) {
        value /= order.size();
    }
    model.apply(dw);
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
        if (argc < 5) {
            printHelp(argv[0]);
        }
        args.imageFile = argv[3];
        args.labelFile = argv[4];
    }

    if (argc >= 6) {
        args.miniStep = std::stoi(argv[5]);
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
    if (args.imageFile.empty()) {
        skipTraining = true;
    } else if (!fs::exists(args.imageFile)) {
        std::cerr << std::format("'{}' does not exist\n", std::string(args.imageFile));
        return EXIT_FAILURE;
    } else if (!fs::exists(args.labelFile)) {
        std::cerr << std::format("'{}' does not exist\n", std::string(args.labelFile));
        return EXIT_FAILURE;
    }

    std::size_t modelInputSize = 28 * 28;

    EmptyModel emptyModel;
    ModelBuilder modelBuilder(emptyModel, modelInputSize);
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
        ImageBank imageBank = loadImages(args.imageFile);
        std::vector<char> labels = loadLabels(args.labelFile);

        assert(imageBank.cols * imageBank.rows == modelInputSize);
        Model & model = modelBuilder.finalize(weights);

        std::size_t n = imageBank.n;
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0UZ);
        std::shuffle(order.begin(), order.end(), std::random_device());
        int nMiniSteps = n / args.miniStep;
        for (int step = 0; step < nMiniSteps; ++step) {
            if (step % 10 == 0) {
                std::cout << std::format("Step {} / {}\n", step, nMiniSteps);
            }
            std::span<std::size_t> thisStepOrder(order.begin() + step * args.miniStep, args.miniStep);
            performMinistep(model, imageBank, labels, thisStepOrder);
        }
        weights = model.weights();
    }

    saveWeights(args.weightsOut, weights);
}
