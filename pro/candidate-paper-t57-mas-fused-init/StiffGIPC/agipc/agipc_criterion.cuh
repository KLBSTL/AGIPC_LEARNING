#pragma once
#include <gipc/utils/json.h>
#include <cstddef>
#include <string>
#include <vector>
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
    const int* basis_masks = nullptr;
    const double3* rest_positions = nullptr;
    int fine_nodes = 0;
    int coarse_nodes = 0;
    int translational_nodes = 0;
    int affine_nodes = 0;
    int coarse_block_nodes = 0;
    bool paper_affine_basis = true;
    bool ready = false;
};

// Criterion-only instrumentation; never changes the baseline solve direction.
void initialize_criterion(const tetrahedra_obj& mesh,
                          double threshold,
                          int max_levels,
                          bool paper_affine_basis = true);
void begin_criterion_step(device_TetraData& mesh);
gipc::Json update_criterion(device_TetraData& mesh);
gipc::Json update_mapping();
gipc::Json capture_criterion_snapshot(const std::string& directory);
MappingDeviceView mapping_device_view();
void configure_galerkin(int fine_correction_max_iterations,
                        bool adoption_enabled,
                        std::string fallback_diagnostics_directory = {},
                        std::string coarse_diagnostics_directory = {},
                        bool use_coarse_mas32 = false,
                        bool use_factorized_mas32 = false,
                        std::string mas_validation = "gpu",bool mas_reuse_enabled = false);
bool galerkin_adoption_enabled();
void configure_instrumentation(bool collect_timing, bool collect_full_diagnostics);
void configure_tiny_dense_limit(int max_dofs);
gipc::Json tiny_dense_self_test();
bool collect_timing_enabled();
bool collect_full_diagnostics_enabled();
void configure_direction_freeze(std::string directory, std::size_t after_update,
                                bool continue_motion=false, std::vector<int> frames={}, int late_after_newton=0);
void set_direction_capture_context(int frame, int active_pairs, int active_ground_pairs,
                                   const int4* collision_pairs=nullptr);
gipc::Json direction_capture_summary();
bool direction_freeze_complete();
std::string freeze_accepted_direction(const GIPCTripletMatrix& fine_matrix,
                               const double* fine_rhs, std::size_t fine_dofs);
gipc::Json update_galerkin_shadow(const GIPCTripletMatrix& fine_matrix,
                                  const double* fine_rhs,
                                  std::size_t fine_rhs_dofs);
gipc::Json adopt_galerkin_candidate(double* destination, std::size_t destination_dofs);
void configure_guarded_fine_correction(bool enabled);
bool guarded_fine_correction_enabled();
void configure_guarded_fine_warm_start(bool enabled);
bool guarded_fine_warm_start_enabled();
double galerkin_prolongated_residual_squared();
const double* galerkin_prolongated(std::size_t dofs);
void record_guarded_fine_correction(gipc::Json result, const double* selected, std::size_t dofs);
void record_linear_solve_timing(gipc::Json timing);
void record_fallback_direction_quality(const GIPCTripletMatrix& fine_matrix,
                                       const double* fine_rhs,
                                       const double* fine_direction,
                                       std::size_t fine_dofs);
gipc::Json criterion_summary();
gipc::Json galerkin_summary();
gipc::Json criterion_self_test();
gipc::Json line_search_probe_self_test();
gipc::Json bvh_bounds_self_test();
gipc::Json collision_pair_capacity_self_test();
gipc::Json mapping_self_test();
gipc::Json galerkin_self_test();
gipc::Json replay_coarse_snapshot(const std::string& sample_directory);
gipc::Json benchmark_production_coarse(const GIPCTripletMatrix& matrix,
                                       const double* device_rhs,int repetitions);
}
