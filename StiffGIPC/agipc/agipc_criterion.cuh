#pragma once
#include <gipc/utils/json.h>
#include <cstddef>
#include <vector_types.h>

class tetrahedra_obj;
class device_TetraData;
class GIPCTripletMatrix;

namespace agipc
{
struct MappingDeviceView
{
    const int* fine_to_coarse = nullptr;
    const int* coarse_block_bases = nullptr;
    const int* affine_flags = nullptr;
    const double3* rest_positions = nullptr;
    int fine_nodes = 0;
    int coarse_nodes = 0;
    int translational_nodes = 0;
    int affine_nodes = 0;
    int coarse_block_nodes = 0;
    bool ready = false;
};

// Criterion-only instrumentation; never changes the baseline solve direction.
void initialize_criterion(const tetrahedra_obj& mesh, double threshold, int max_levels);
void begin_criterion_step(device_TetraData& mesh);
gipc::Json update_criterion(device_TetraData& mesh);
gipc::Json update_mapping();
MappingDeviceView mapping_device_view();
void configure_galerkin(int fine_correction_max_iterations);
gipc::Json update_galerkin_shadow(const GIPCTripletMatrix& fine_matrix,
                                  const double* fine_rhs,
                                  std::size_t fine_rhs_dofs);
gipc::Json criterion_summary();
gipc::Json galerkin_summary();
gipc::Json criterion_self_test();
gipc::Json mapping_self_test();
gipc::Json galerkin_self_test();
}
