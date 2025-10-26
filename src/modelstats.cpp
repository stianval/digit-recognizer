#include "model.h"
#include "weightstorage.h"
#include "dataloader.h"

#include <format>
#include <iostream>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

struct ProgArgs {
    fs::path weightsPath;
    fs::path imageFile;
    fs::path labelFile;
};

struct DigitStats {
    int tp = 0;  // true positive
    int fp = 0;  // false positive
    int tn = 0;  // true negative
    int fn = 0;  // false negative
};

void printHelp(const char * progName)
{
    std::cerr << std::format("Usage: {} <weights-file> <image-file> <label-file>", progName);
    std::exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
    if (argc < 4) {
        printHelp(argv[0]);
    }
    ProgArgs args;
    args.weightsPath = argv[1];
    args.imageFile = argv[2];
    args.labelFile = argv[3];

    if (!fs::exists(args.weightsPath)) {
        std::cerr << std::format("'{}' does not exist\n", std::string(args.weightsPath));
        return EXIT_FAILURE;
    }
    if (!fs::exists(args.imageFile)) {
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

    loadWeights(args.weightsPath, weights);
    ImageBank imageBank = loadImages(args.imageFile);
    std::vector<char> labels = loadLabels(args.labelFile);

    if (imageBank.n != labels.size()) {
        std::cerr << std::format("image bank size ({}) and labels size ({}) does not match\n", imageBank.n, labels.size());
        return EXIT_FAILURE;
    }
    if (imageBank.rows * imageBank.cols != modelInputSize) {
        std::cerr << std::format("image size ({}*{}={}) and model input size ({}) does not match\n", imageBank.rows, imageBank.cols, imageBank.rows * imageBank.cols, modelInputSize);
        return EXIT_FAILURE;
    }

    Model & model = modelBuilder.finalize(weights);

    int correct = 0;
    double totalCost = 0.0;
    double totalConfidenceOvershoot = 0.0;
    double highestConfidenceOvershoot = 0.0;
    DigitStats digitStats[10];
    std::size_t n = imageBank.n;
    for (std::size_t i = 0; i < n; ++i) {
        fvec_t activations = model.calculateActivations(imageBank.at(i));
        cfspan_t result = model.activationSpans(activations).back();
        int highestDigit = 0;
        int highestDigitConfidence = 0;
        for (int digit = 0; digit < 10; ++digit) {
            if (result[digit] > highestDigitConfidence) {
                highestDigit = digit;
                highestDigitConfidence = result[digit];
            }
            bool isTarget = digit == labels[i];
            double diff = result[digit] - isTarget;
            totalCost += diff * diff;
            digitStats[digit].tn++;
        }
        if (highestDigit == labels[i]) {
            correct++;
            auto & correctDigitStat = digitStats[highestDigit];
            correctDigitStat.tn--;
            correctDigitStat.tp++;
        } else {
            auto & predictedDigitStat = digitStats[highestDigit];
            predictedDigitStat.tn--;
            predictedDigitStat.fp++;
            auto & actualDigitStat = digitStats[int(labels[i])];
            actualDigitStat.tn--;
            actualDigitStat.fn++;
        }
        if (highestDigitConfidence > 1.0) {
            double confidenceOvershoot = highestDigitConfidence - 1.0;
            totalConfidenceOvershoot += confidenceOvershoot;
            if (highestConfidenceOvershoot < confidenceOvershoot) {
                highestConfidenceOvershoot = confidenceOvershoot;
            }
        }
    }
    std::cout << std::format("Correct: {}/{} ({:.2f}%)\n", correct, n, 100.0 * correct / n);
    std::cout << std::format("Avg cost: {:.2f}\n", totalCost / n);
    std::cout << std::format("Highest confidence overshoot: {}\n", highestConfidenceOvershoot);
    std::cout << std::format("Total confidence overshoot: {}\n", totalConfidenceOvershoot);
    for (int digit = 0; digit < 10; ++digit) {
        auto ds = digitStats[digit];
        std::cout << std::format("Digit {}: tp={:5}    fp={:5}     tn={:5}     fn={:5}\n",
            digit, ds.tp, ds.fp, ds.tn, ds.fn);
    }
}
