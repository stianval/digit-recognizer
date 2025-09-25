#include "model.h"
#include <format>
#include <iostream>
#include <cstdlib>

int main(int argc, const char * argv[])
{
    if (argc < 5) {
        std::cout << std::format("Usage: {} <model-in> <model-out> <training-dir> <mini-step>", argv[0]);
        return EXIT_FAILURE;
    }
}
