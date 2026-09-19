#pragma once

namespace gipc
{
enum class SpmvMode
{
    Legacy,
    SRBK,
    Hybrid8,
    Hybrid16
};

inline const char* to_string(SpmvMode mode)
{
    switch(mode)
    {
    case SpmvMode::Legacy:
        return "legacy";
    case SpmvMode::SRBK:
        return "srbk";
    case SpmvMode::Hybrid8:
        return "hybrid8";
    case SpmvMode::Hybrid16:
        return "hybrid16";
    }
    return "unknown";
}
}  // namespace gipc
