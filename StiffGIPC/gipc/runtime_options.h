#pragma once

#include <gipc/spmv_mode.h>
#include <string>

namespace gipc
{
enum class SolverMode
{
    StiffGIPC,
    AGIPC
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
    SpmvMode    spmv        = SpmvMode::SRBK;
    int         frames      = 0;
    int         figure12_bunny_count = 2;
    double      figure12_collision_buffer_scale = 1.0;
    double      figure12_linear_system_buffer_scale = 1.0;
    double      young_modulus = 1e7;
    double      dt            = 0.01;
    bool        headless      = false;
    std::string metrics_path;
    double      agipc_threshold = 5e-5;
    std::string agipc_mapping   = "warp-hash";
    int         agipc_max_levels = 8;
    int         agipc_fine_correction_iterations = 10;
    bool        agipc_diagnostics = false;
    bool        agipc_self_test   = false;
    bool        spmv_self_test    = false;
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
