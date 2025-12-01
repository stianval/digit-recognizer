#include "model.h"

#include <cassert>
#include <random>

float & Matrix::at(std::size_t row, std::size_t col) const
{
    return data_[row * cols_ + col];
}

std::size_t Matrix::size() const
{
    return rows_ * cols_;
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

void Matrix::updateWeightDifferentials(fspan_t dw, cfspan_t dR_dz, cfspan_t input) const
{
    assert(dw.size() == rows_ * cols_);
    assert(dR_dz.size() == rows_);
    assert(input.size() == cols_  - 1);
    for (std::size_t row = 0; row < rows_; ++row) {
        for (std::size_t col = 0; col < cols_ - 1; ++col) {
            std::size_t i = row * cols_ + col;
            dw[i] += input[col] * dR_dz[row];
        }
        std::size_t i = row * cols_ + cols_ - 1;
        dw[i] += dR_dz[row];
    }
}

void Matrix::overwriteActivationsWith_dR_dz(fspan_t activations, fspan_t dR_dz_prev) const
{
    assert(dR_dz_prev.size() == rows_);
    assert(activations.size() == cols_  - 1);
    for (std::size_t col = 0; col < cols_ - 1; ++col) {
        if (activations[col] == 0.0f)  // test if da/dz is 0
            continue;
        activations[col] = 0.0f;
        for (std::size_t row = 0; row < rows_; ++row) {
             activations[col] += at(row, col) * dR_dz_prev[row];
        }
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

Model::Model(const Model & other)
    : layers_(other.layers_)
    , totalNeurons_(other.totalNeurons_)
{
    finalize(other.weights_);
}

std::size_t Model::size() const
{
    return weights_.size();
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

std::vector<fspan_t> Model::activationSpans(fspan_t activations) const
{
    std::vector<fspan_t> result;
    result.reserve(layers_.size());
    float * start = activations.data();
    for (const Matrix & layer : layers_) {
        result.push_back(fspan_t(start, layer.rows_));
        start += layer.rows_;
    }
    return result;
}

void Model::backPropagate(fvec_t & dw, fvec_t activations, cfspan_t target, cfspan_t input) const
{
    // dR/dw = dz/dw da/dz dR/da   -- w is a weight or a bias
    fspan_t dR_dz(activations.end() - target.size(), target.size());
    fspan_t curr_dw(dw.end(), 0);
    for (std::size_t i = 0; i < target.size(); ++i) {
        bool da_dz = (dR_dz[i] != 0.0f);   // ReLU'(z)
        dR_dz[i] = da_dz * 2 * (dR_dz[i] - target[i]);
    }
    for (auto layerIt = layers_.rbegin(); layerIt != layers_.rend() - 1; ++layerIt) {
        auto & layer = *layerIt;
        assert(dR_dz.size() == layer.rows_);
        curr_dw = fspan_t(curr_dw.begin() - layer.size(), layer.size());
        fspan_t curr_activations(dR_dz.begin() - (layer.cols_ - 1), layer.cols_ - 1);
        layer.updateWeightDifferentials(curr_dw, dR_dz, curr_activations);

        layer.overwriteActivationsWith_dR_dz(curr_activations, dR_dz);
        dR_dz = curr_activations;
    }
    auto & layer = layers_.front();
    curr_dw = fspan_t(curr_dw.begin() - layer.size(), layer.size());
    layer.updateWeightDifferentials(curr_dw, dR_dz, input);
}

void Model::apply(const fvec_t & dw)
{
    assert(dw.size() == weights_.size());
    for (std::size_t i = 0; i < dw.size(); ++i) {
        weights_[i] -= dw[i];
    }
}

const fvec_t & Model::weights() const
{
    return weights_;
}

Model & Model::finalize(fvec_t weights)
{
    weights_ = std::move(weights);
    float * nextData = weights_.data();
    float * const start = nextData;
    for (Matrix & layer : layers_) {
        nextData = layer.grabData(nextData);
    }
    assert(std::size_t(nextData - start) == weights_.size());
    return *this;
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
    assert(weights.size() == totalWeights_);
    return model_.finalize(std::move(weights));
}

fvec_t ModelBuilder::prepareKaimingHeWeights()
{
    std::random_device rnd;
    fvec_t weights(totalWeights_, 0.0f);
    float * layerWeights = weights.data();
    for (Matrix & layer : model_.layers_) {
        std::normal_distribution<float> distribution(0.0f, std::sqrt(2.0f / (layer.cols_ - 1)));
        for (std::size_t row = 0; row < layer.rows_; ++row) {
            for (std::size_t col = 0; col < layer.cols_ - 1; ++col) {
                // Note: skip bias column this loop, biases are initialized to 0
                layerWeights[row * layer.cols_ + col] = distribution(rnd);
            }
        }
        layerWeights += layer.cols_ * layer.rows_;
    }
    assert(layerWeights == weights.data() + totalWeights_);
    return weights;
}
