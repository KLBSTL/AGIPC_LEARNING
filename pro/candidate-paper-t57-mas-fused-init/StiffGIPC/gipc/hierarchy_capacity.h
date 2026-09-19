#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace gipc
{
// Semantic backport of upstream a5bf7e5, adapted to the raw-pointer workspace.
inline std::size_t hierarchy_capacity(std::size_t vertices, std::size_t mapped,
                                      std::size_t levels, std::size_t bank)
{
    const auto base=std::max(vertices,mapped);
    if(!base || !levels) return 0;
    const auto maximum=std::numeric_limits<std::size_t>::max()/sizeof(unsigned int);
    if(!bank || base>maximum || bank-1>maximum-base)
        throw std::overflow_error("MAS hierarchy alignment overflow");
    const auto padded=(base+bank-1)/bank*bank;
    if(padded>maximum/levels)
        throw std::overflow_error("MAS hierarchy capacity overflow");
    return padded*levels;
}
}
