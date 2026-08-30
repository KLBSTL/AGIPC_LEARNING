#pragma once

namespace gipc
{
enum class SpmvMode
{
    Legacy,
    SRBK
};

inline const char* to_string(SpmvMode mode)
{
    return mode == SpmvMode::Legacy ? "legacy" : "srbk";
}
}  // namespace gipc
