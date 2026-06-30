#pragma once

#include <iomanip>
#include <sstream>
#include <string>

namespace betavllm::test
{
inline constexpr const char* kGreen = "\033[32m";
inline constexpr const char* kRed = "\033[31m";
inline constexpr const char* kReset = "\033[0m";

inline std::string color(bool ok, const std::string& text)
{
    return std::string(ok ? kGreen : kRed) + text + kReset;
}

inline std::string passFail(bool ok)
{
    return color(ok, ok ? "PASS" : "FAIL");
}

inline std::string speedup(double value)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3) << value << "x";
    return color(value > 1.0, stream.str());
}
}
