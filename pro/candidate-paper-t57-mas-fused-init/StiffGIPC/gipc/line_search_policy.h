#pragma once
#include <algorithm>
#include <cmath>

namespace gipc
{
constexpr int line_search_backtrack_limit = 64;
inline bool valid_line_search_input(double alpha, double cfl_alpha, double energy)
{
    return std::isfinite(alpha) && alpha>0 && std::isfinite(cfl_alpha)
        && cfl_alpha>0 && std::isfinite(energy);
}
inline double next_line_search_alpha(double alpha, double cfl_alpha)
{
    return std::min(alpha*0.5,cfl_alpha);
}
}
