#pragma once
#include <gipc/utils/json.h>

namespace agipc
{
enum class StepStatisticsMode { Automatic, Compact, History };
void configure_step_statistics_mode(StepStatisticsMode mode);
gipc::Json galerkin_step_summary();
gipc::Json step_statistics_self_test();
}
