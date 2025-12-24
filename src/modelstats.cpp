#include "model.h"
#include "weightstorage.h"
#include "dataloader.h"

#include <format>
#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

struct ProgArgs {
    fs::path weightsPath;
    fs::path imageFile;
    fs::path labelFile;
    fs::path csvFile;
};

struct Stats {
    int correct = 0;
    double totalCost = 0.0;
    double totalConfidenceOvershoot = 0.0;
    double highestConfidenceOvershoot = 0.0;
    double totalConfidenceUndershoot = 0.0;
    double highestConfidenceUndershoot = 0.0;
};

struct DigitStats {
    int tp = 0;  // true positive
    int fp = 0;  // false positive
    int tn = 0;  // true negative
    int fn = 0;  // false negative
};

void printHelp(const char * progName)
{
    std::cerr << std::format("Usage: {} <weights-file> <image-file> <label-file> [csv-file]", progName);
    std::exit(EXIT_FAILURE);
}

void printCsvHeader(std::ostream & stream)
{
    stream << "file,correct %,avg. cost,highest over,total over,highest under,total under";
    for (int digit = 0; digit < 10; digit++) {
        stream << std::format("tp{0},fp{0},tn{0},fn{0},acc{0} %,pre{0} %, rec{0} %", digit);
    }
    stream << "\n";
}

void printStats(std::ostream & stream, bool csv, int n, const Stats & stats, const DigitStats (& digitStats)[10])
{
    using digit_format_t = std::format_string<
        int&, int&, int&, int&,
        float&, float&, float&
    >;
    digit_format_t digitFormat =
        "tp={:5}    fp={:5}     tn={:5}     fn={:5}    "
        "acc={:4.1f}%    pre={:4.1f}%   rec={:4.1f}%\n";
    if (csv) {
        stream << std::format("{:.3f},", 100.0f * stats.correct / n);
        stream << std::format("{:.4f},", stats.totalCost / n);
        stream << std::format("{:.3f},{:.3f},{:.3f},{:.3f},",
            stats.highestConfidenceOvershoot, stats.totalConfidenceOvershoot,
            stats.highestConfidenceUndershoot, stats.totalConfidenceUndershoot);
        digitFormat = "{},{},{},{},{:.3f},{:.3f},{:.3f}";
    } else {
        stream << std::format("Correct: {}/{} ({:.2f}%)\n", stats.correct, n, 100.0 * stats.correct / n);
        stream << std::format("Avg cost: {:.4f}\n", stats.totalCost / n);
        stream << std::format("Highest confidence overshoot: {}\n", stats.highestConfidenceOvershoot);
        stream << std::format("Total confidence overshoot: {}\n", stats.totalConfidenceOvershoot);
        stream << std::format("Highest confidence undershoot: {}\n", stats.highestConfidenceUndershoot);
        stream << std::format("Total confidence undershoot: {}\n", stats.totalConfidenceUndershoot);
    }
    for (int digit = 0; digit < 10; ++digit) {
        if (!csv) {
            stream << std::format("Digit {}: ", digit);
        }
        auto ds = digitStats[digit];
        int digitTotal = ds.tp + ds.fp + ds.tn + ds.fn;
        assert(digitTotal == n);
        float accuracyPct = float(ds.tp + ds.tn) / digitTotal * 100;
        float precisionPct = float(ds.tp) / (ds.tp + ds.fp) * 100;
        float recallPct = float(ds.tp) / (ds.tp + ds.fn) * 100;
        stream << std::format(digitFormat,
            ds.tp, ds.fp, ds.tn, ds.fn,
            accuracyPct, precisionPct, recallPct);
    }
    if (csv) {
        stream << "\n";
    }
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
    if (argc >= 4) {
        args.csvFile = argv[4];
    }

    std::ofstream csvFile;
    if (!args.csvFile.empty()) {
        if (fs::exists(args.csvFile)) {
            csvFile.open(args.csvFile, std::ios_base::app);
        } else {
            csvFile.open(args.csvFile);
            printCsvHeader(csvFile);
        }
    }
    std::ostream & ostream = csvFile ? csvFile : std::cout;

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

    Stats stats;
    DigitStats digitStats[10];
    std::size_t n = imageBank.n;
    for (std::size_t i = 0; i < n; ++i) {
        fvec_t activations = model.calculateActivations(imageBank.at(i));
        cfspan_t result = model.activationSpans(activations).back();
        int highestDigit = 0;
        float highestDigitConfidence = 0;
        for (int digit = 0; digit < 10; ++digit) {
            if (result[digit] > highestDigitConfidence) {
                highestDigit = digit;
                highestDigitConfidence = result[digit];
            }
            bool isTarget = digit == labels[i];
            double diff = result[digit] - isTarget;
            stats.totalCost += diff * diff;
            digitStats[digit].tn++;
        }
        if (highestDigit == labels[i]) {
            stats.correct++;
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
            stats.totalConfidenceOvershoot += confidenceOvershoot;
            if (stats.highestConfidenceOvershoot < confidenceOvershoot) {
                stats.highestConfidenceOvershoot = confidenceOvershoot;
            }
        } else {
            double confidenceUndershoot = 1.0 - highestDigitConfidence;
            stats.totalConfidenceUndershoot += confidenceUndershoot;
            if (stats.highestConfidenceUndershoot < confidenceUndershoot) {
                stats.highestConfidenceUndershoot = confidenceUndershoot;
            }
        }
    }
    if (csvFile) {
        ostream << args.weightsPath << ',';
    }
    printStats(ostream, bool(csvFile), n, stats, digitStats);
}
