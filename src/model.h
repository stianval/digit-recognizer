#ifndef MODEL_H
#define MODEL_H

#include <vector>

using fvec_t = std::vector<float>;

class Matrix {
public:
    float & at(std::size_t row, std::size_t col) const;
    fvec_t affineMultiply(const fvec_t & vec) const;

private:  // functions
    explicit Matrix(std::size_t rows, std::size_t cols);
    float * grabData(float * data);
    friend class ModelBuilder;

private:
    float* data_;
    std::size_t cols_;
    std::size_t rows_;
};

class Model {
public:
    fvec_t runInference(fvec_t input) const;
private:
    Model() = default;
    std::vector<Matrix> layers_;
    fvec_t weights_;
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

