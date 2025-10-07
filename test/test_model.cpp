#include "../src/model.h"
#include "test_common.h"

#include <cmath>
#include <format>
#include <iostream>

const fvec_t g_weights = {
    1.0f, 1.0f, 1.0f, -1.0f, 0.1f, 0.1f, 1.0f, 
    0.9f, 1.0f, 1.0f, -1.0f, 0.1f, 0.2f, 1.0f, 
    0.6f, 1.0f, 1.0f, -1.0f, 0.1f, 0.5f, 0.0f, 
    0.5f, 1.0f, 1.0f, -1.0f, 0.1f, 0.6f, 0.0f, 
    0.2f, 1.0f, 1.0f, -1.0f, 0.1f, 0.9f, -1.0f, 
    0.1f, 1.0f, 1.0f, -1.0f, 0.1f, 1.0f, -1.0f, 

    0.1f, 1.0f, 1.0f, -1.0f, 0.1f, 1.0f, 0.0f, 
    0.2f, 1.0f, 1.0f, -1.0f, 0.1f, 0.9f, 0.0f, 
    0.5f, 1.0f, 1.0f, -1.0f, 0.1f, 0.6f, -1.0f, 
    0.6f, 1.0f, 1.0f, -1.0f, 0.1f, 0.5f, -1.0f, 
    0.9f, 1.0f, 1.0f, -1.0f, 0.1f, 0.2f, -2.0f, 
    1.0f, 1.0f, 1.0f, -1.0f, 0.1f, 0.1f, -2.0f, 
    0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, -0.5f, 
};

const std::size_t g_firstMatrixSize = 42;

const fvec_t g_input = {
    -0.1f, 0.2f, 0.4f, 0.5f, -0.6f, 0.9f,
};

// common part:
// 0.2*1.0 + 0.4*1.0 + 0.5*-1.0 + -0.6*0.1 = 0.04

const fvec_t g_first_multiplication = {
    1.03f, 1.13f, 0.43f, 0.53f, -0.17f, -0.11f,
};

const fvec_t g_first_relu = {
    1.03f, 1.13f, 0.43f, 0.53f, -0.0f, -0.0f,
};

// common part for the six first rows:
// 1.13*1.0 + 0.43*1.0 + 0.53*-1.0 + 0.0*0.1 = 1.03

const fvec_t g_second_multiplication = {
    1.133f, 1.236f, 0.545f, 0.648f, -0.043f, 0.06f, 0.1f,
};

const fvec_t g_second_relu = {
    1.133f, 1.236f, 0.545f, 0.648f, -0.0f, 0.06f, 0.1f,
};

void case1()
{
    EmptyModel emptyModel;
    ModelBuilder modelBuilder(emptyModel, 6);
    modelBuilder.addLayer(6);
    ASSERT_EQ(modelBuilder.size(), g_firstMatrixSize, "");
    Model & model = modelBuilder.finalize(fvec_t(g_weights.data(), g_weights.data() + g_firstMatrixSize));
    fvec_t result = model.runInference(g_input);
    ASSERT_EQ(result.size(), g_first_relu.size(), "");
    bool passing = true;
    for (std::size_t i = 0; i < result.size(); ++i) {
        passing |= EXPECT_FUZZ_EQ(result[i], g_first_relu[i], std::format("[{}]", i), 1e-7f);
    }
    ASSERT_EQ(passing, true, "");
}


void case2()
{
    EmptyModel emptyModel;
    ModelBuilder modelBuilder(emptyModel, 6);
    modelBuilder.addLayer(6);
    modelBuilder.addLayer(7);
    ASSERT_EQ(modelBuilder.size(), g_weights.size(), "");
    Model & model = modelBuilder.finalize(g_weights);
    fvec_t result = model.runInference(g_input);
    ASSERT_EQ(result.size(), g_second_relu.size(), "");
    bool passing = true;
    for (std::size_t i = 0; i < result.size(); ++i) {
        passing |= EXPECT_FUZZ_EQ(result[i], g_second_relu[i], std::format("[{}]", i), 1e-6f);
    }
    ASSERT_EQ(passing, true, "");
}

int main()
{
    case1();
    case2();
    std::cout << "All tests passed!" << std::endl;
}

