#pragma once
#include <gipc/utils/json.h>
namespace agipc {
void configure_coarse_pcg_batch(int batch);
gipc::Json device_pcg_self_test();
}
