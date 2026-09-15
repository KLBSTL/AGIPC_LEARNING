#pragma once

#include <gipc/spmv_mode.h>
#include <string>

namespace gipc
{
enum class SolverMode
{
    StiffGIPC,
    AGIPC,
    AGIPCSymHessian,
    AGIPCPaper
};

struct RuntimeOptions
{
    std::string scene       = "interactive";
    SolverMode  solver      = SolverMode::StiffGIPC;
    std::string tet_mesh;
    std::string cloth_mesh;
    std::string framework      = "abd-cemas-srbk";
    std::string preconditioner = "cemas16";
    std::string body_mode      = "hybrid-abd";
    std::string newton_stop    = "solver-default";
    SpmvMode    spmv        = SpmvMode::SRBK;
    int         frames      = 0;
    int         figure12_bunny_count = 2;
    double      figure12_collision_buffer_scale = 1.0;
    double      figure12_linear_system_buffer_scale = 1.0;
    double      young_modulus = 1e7;
    double      dt            = 0.01;
    bool        headless      = false;
    std::string metrics_path;
    std::string fem_final_state_path;
    std::string fem_checkpoint_path;
    int         fem_checkpoint_start_frame = 1;
    int         fem_checkpoint_stride = 1;
    double      agipc_threshold = 5e-5;
    std::string agipc_mapping   = "warp-hash";
    int         agipc_max_levels = 16;
    int         agipc_fine_correction_iterations = 10;
    std::string agipc_coarse_preconditioner = "block-jacobi";
    std::string agipc_mas_validation = "gpu";
    bool        agipc_mas_reuse = false;
    bool        agipc_diagnostics = false;
    std::string agipc_direction_freeze_path;
    int         agipc_direction_freeze_after_update = 0;
    bool        agipc_self_test   = false;
    bool        spmv_self_test    = false;
    bool        mas32_self_test   = false;
    std::string agipc_coarse_replay_path;
    std::string frozen_linear_diagnostics_path;
};

struct ParseResult
{
    RuntimeOptions options;
    bool           ok        = true;
    bool           show_help = false;
    int            exit_code = 0;
    std::string    error;
    std::string    help;
};

ParseResult parse_runtime_options(int argc, char** argv);
std::string runtime_options_help();
}  // namespace gipc
