#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <format>
#include <iostream>
#include <type_traits>

template <typename T>
bool assert_eq(T a, T b, const char * aStr, const char * bStr,
               const char * function, int line, const std::string & what,
               bool stop = true, T epsilon = T{})
{
    if (a == b)
        return true;
    if constexpr (std::is_signed_v<T>) {
        if (epsilon != T{} && std::abs(b - a) < epsilon)
            return true;
    }
    std::cerr <<
        std::format("{}:{} {}: Values not equal: {}\n"
                "\t{:16}: {:16}\n"
                "\t{:16}: {:16}\n",
                __FILE__, line, function, what,
                aStr, a, bStr, b);
    if (stop)
        std::abort();
    return false;
}

template <typename T>
bool assert_neq(T a, T b, const char * aStr, const char * bStr,
                const char * function, int line, const std::string & what,
                bool stop = true)
{
    if (a != b)
        return true;
    std::cerr <<
        std::format("{}:{} {}: Values should be diffferent, but are equal: {}\n"
                "\t{:16}: {:16}\n"
                "\t{:16}: {:16}\n",
                __FILE__, line, function, what,
                aStr, a, bStr, b);
    if (stop)
        std::abort();
    return false;
}

#define ASSERT_EQ(a, b, what) assert_eq(a, b, #a, #b, __FUNCTION__, __LINE__, what);
#define EXPECT_EQ(a, b, what) assert_eq(a, b, #a, #b, __FUNCTION__, __LINE__, what, false);
#define EXPECT_FUZZ_EQ(a, b, what, epsilon) assert_eq(a, b, #a, #b, __FUNCTION__, __LINE__, what, false, epsilon);

#define ASSERT_NEQ(a, b, what) assert_neq(a, b, #a, #b, __FUNCTION__, __LINE__, what);
#define EXPECT_NEQ(a, b, what) assert_neq(a, b, #a, #b, __FUNCTION__, __LINE__, what, false);

#endif  // TEST_COMMON_H
