#include "model.h"

#include <cassert>

float & Matrix::at(std::size_t row, std::size_t col) const
{
    return data_[row * cols_ + col];
}

fvec_t Matrix::affineMultiply(const fvec_t & vec) const
{
    assert(vec.size() + 1 == cols_);
    fvec_t result(rows_, 0.0);
    for (std::size_t row = 0; row < rows_; row++) {
        float & v = result[row];
        for (std::size_t col = 0; col < vec.size(); col++) {
            v += vec[col] * at(row, col);
        }
        v += at(row, cols_ - 1);
    }
    return result;
}

void Matrix::affineMultiply(const cfspan_t input, const fspan_t output) const
{
    assert(input.size() + 1 == cols_);
    assert(output.size() == rows_);
    for (std::size_t row = 0; row < rows_; row++) {
        float & v = output[row];
        for (std::size_t col = 0; col < input.size(); col++) {
            v += input[col] * at(row, col);
        }
        v += at(row, cols_ - 1);
    }
}

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : cols_(cols)
    , rows_(rows)
{
}

float * Matrix::grabData(float * data)
{
    data_ = data;
    return data + rows_ * cols_;
}

fvec_t Model::runInference(fvec_t input) const
{
    fvec_t & result = input;
    for (const Matrix & layer : layers_) {
        result = std::move(layer.affineMultiply(result));
        for (float & v : result) {
            // apply ReLU
            v = std::max(0.0f, v);
        }
    }
    return result;
}

fvec_t Model::calculateActivations(cfspan_t input) const
{
    fvec_t activations(totalNeurons_);
    float * outputStart = activations.data();
    for (const Matrix & layer : layers_) {
        std::span output(outputStart, layer.rows_);
        layer.affineMultiply(input, output);
        for (float & v : output) {
            // apply ReLU
            v = std::max(0.0f, v);
        }
        outputStart = output.data() + output.size();
        input = output;
    }
    return activations;
}

ModelBuilder::ModelBuilder(EmptyModel & model, std::size_t expectedInputSize)
    : model_(model)
    , currentLayerSize_(expectedInputSize)
{ ; }

void ModelBuilder::addLayer(std::size_t size)
{
    model_.layers_.push_back(Matrix(size, currentLayerSize_ + 1));
    currentLayerSize_ = size;
    model_.totalNeurons_ += size;
    const Matrix & m = model_.layers_.back();
    totalWeights_ += m.rows_ * m.cols_;
}

std::size_t ModelBuilder::size() const
{
    return totalWeights_;
}

Model & ModelBuilder::finalize(fvec_t weights)
{
    model_.weights_ = std::move(weights);
    float * nextData = model_.weights_.data();
    float * const start = nextData;
    for (Matrix & layer : model_.layers_) {
        nextData = layer.grabData(nextData);
    }
    assert(std::size_t(nextData - start) == totalWeights_);
    return model_;
}

