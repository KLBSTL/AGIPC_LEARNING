#pragma once
#include <gipc/utils/json.h>
namespace agipc {
void configure_direct_mas_apply(bool enabled);
gipc::Json mas_apply_self_test();
}
