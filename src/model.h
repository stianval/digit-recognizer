#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <span>

using fvec_t = std::vector<float>;
using fspan_t = std::span<float>;
using cfspan_t = std::span<const float>;

class Matrix {
public:
    float & at(std::size_t row, std::size_t col) const;
    std::size_t size() const;
    fvec_t affineMultiply(const fvec_t & vec) const;
    void affineMultiply(const cfspan_t input, const fspan_t output) const;
    void updateWeightDifferentials(fspan_t dw, cfspan_t dR_dz, cfspan_t input) const;
    void overwriteActivationsWith_dR_dz(fspan_t activations, fspan_t dR_dz_prev) const;

private:  // functions
    explicit Matrix(std::size_t rows, std::size_t cols);
    float * grabData(float * data);
    friend class ModelBuilder;
    friend class Model;

private:
    float* data_;
    const std::size_t cols_;
    const std::size_t rows_;
};

class Model {
public:
    Model(const Model & other);
    Model(Model && other) = delete;
    Model & operator=(const Model & other) = delete;
    Model & operator=( Model && other) = delete;
    std::size_t size() const;
    fvec_t runInference(fvec_t input) const;
    fvec_t calculateActivations(cfspan_t input) const;
    std::vector<fspan_t> activationSpans(fspan_t activations) const;
    void backPropagate(fvec_t & dw, fvec_t activations, cfspan_t target, cfspan_t input) const;
    void apply(const fvec_t & dw);
    const fvec_t & weights() const;

private:  // functions
    Model() = default;
    Model & finalize(fvec_t weights);

private:
    std::vector<Matrix> layers_;
    fvec_t weights_;
    std::size_t totalNeurons_{};
    friend class ModelBuilder;
    friend class EmptyModel;
};

class EmptyModel : private Model {
private:
    friend class ModelBuilder;
};

class ModelBuilder {
public:
    explicit ModelBuilder(EmptyModel & model, std::size_t expectedInputSize);
    void addLayer(std::size_t size);
    std::size_t size() const;
    Model & finalize(fvec_t weights);
private:
    Model & model_;
    std::size_t currentLayerSize_{};
    std::size_t totalWeights_{};
};

#endif  // MODEL_H

