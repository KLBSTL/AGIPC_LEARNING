#include <gipc/runtime_options.h>

#include <stdexcept>
#include <cmath>

namespace gipc
{
namespace
{
ParseResult invalid(RuntimeOptions options, std::string message)
{
    ParseResult result;
    result.options   = std::move(options);
    result.ok        = false;
    result.exit_code = 2;
    result.error     = std::move(message);
    result.help      = runtime_options_help();
    return result;
}
}  // namespace

std::string runtime_options_help()
{
    return "Usage: gipc [options]\n"
           "  --scene stiff-bunny-drop|paper-fig12-coupling-scaled|paper-fig15-cloth-abd-scaled\n"
           "  --solver stiffgipc|agipc-core|agipc-symhessian|agipc-paper (last two pending)\n"
           "  --tet-mesh <MSH_PATH>\n"
           "  --cloth-mesh <OBJ_PATH>\n"
           "  --framework gipc|srbk|cemas-srbk|abd-cemas-srbk\n"
           "  --preconditioner block-diagonal|gpu-mas|cemas16|cemas32\n"
           "  --spmv legacy|srbk|hybrid8|hybrid16\n"
           "  --body-mode fem|hybrid-abd\n"
           "  --newton-stop solver-default|paper-current\n"
           "  --figure12-bunny-count 1|2\n"
           "  --figure12-collision-buffer-scale <VALUE>\n"
           "  --figure12-linear-system-buffer-scale <VALUE>\n"
           "  --spmv-self-test\n"
           "  --mas32-self-test\n"
           "  --agipc-coarse-replay <SAMPLE_DIRECTORY>\n"
           "  --frozen-linear-diagnostics <JSON_PATH>\n"
           "  --frames <N>\n"
           "  --young-modulus <VALUE>\n"
           "  --dt <VALUE>\n"
           "  --headless\n"
           "  --metrics-path <JSON_PATH>\n"
           "  --fem-final-state-path <CSV_PATH>\n"
           "  --fem-checkpoint-path <BINARY_PATH>\n"
           "  --fem-checkpoint-start-frame <N>\n"
           "  --fem-checkpoint-stride <N>\n"
           "  --agipc-threshold <VALUE>\n"
           "  --agipc-mapping matching|warp-hash\n"
           "  --agipc-max-levels <N>\n"
           "  --agipc-fine-correction-iterations <N>\n"
           "  --agipc-coarse-preconditioner block-jacobi|mas32 (experimental)\n"
           "  --agipc-mas-validation gpu|cpu|crosscheck\n"
           "  --agipc-mas-reuse (experimental exact-structure cache)\n"
           "  --agipc-diagnostics\n"
           "  --agipc-direction-freeze-path <DIRECTORY> (capture one accepted direction and stop)\n"
           "  --agipc-direction-freeze-after-update <N>\n"
           "  --agipc-self-test (criterion GPU gate)\n"
           "  --agipc-post-cg-max <N> (alias; zero allowed for ablation)\n"
           "  --help\n";
}

ParseResult parse_runtime_options(int argc, char** argv)
{
    RuntimeOptions options;
    bool           frames_were_explicit = false;
    bool           framework_was_explicit = false;
    bool           preconditioner_was_explicit = false;
    bool           spmv_was_explicit = false;
    bool           body_mode_was_explicit = false;
    bool           young_modulus_was_explicit = false;

    auto require_value = [&](int& index, const std::string& flag) -> const char* {
        if(index + 1 >= argc)
            throw std::invalid_argument(flag + " requires a value");
        return argv[++index];
    };

    try
    {
        for(int i = 1; i < argc; ++i)
        {
            const std::string argument = argv[i];
            if(argument == "--help" || argument == "-h")
            {
                ParseResult result;
                result.options   = options;
                result.show_help = true;
                result.help      = runtime_options_help();
                return result;
            }
            if(argument == "--scene")
                options.scene = require_value(i, argument);
            else if(argument == "--solver")
            {
                const std::string value = require_value(i, argument);
                if(value == "stiffgipc")
                    options.solver = SolverMode::StiffGIPC;
                else if(value == "agipc" || value == "agipc-core")
                    options.solver = SolverMode::AGIPC;
                else if(value == "agipc-symhessian")
                    options.solver = SolverMode::AGIPCSymHessian;
                else if(value == "agipc-paper")
                    options.solver = SolverMode::AGIPCPaper;
                else
                    return invalid(options, "solver must be stiffgipc, agipc-core, agipc-symhessian or agipc-paper");
            }
            else if(argument == "--tet-mesh")
                options.tet_mesh = require_value(i, argument);
            else if(argument == "--cloth-mesh")
                options.cloth_mesh = require_value(i, argument);
            else if(argument == "--framework")
            {
                options.framework       = require_value(i, argument);
                framework_was_explicit = true;
            }
            else if(argument == "--preconditioner")
            {
                options.preconditioner       = require_value(i, argument);
                preconditioner_was_explicit = true;
            }
            else if(argument == "--spmv")
            {
                const std::string value = require_value(i, argument);
                if(value == "legacy")
                    options.spmv = SpmvMode::Legacy;
                else if(value == "srbk")
                    options.spmv = SpmvMode::SRBK;
                else if(value == "hybrid8")
                    options.spmv = SpmvMode::Hybrid8;
                else if(value == "hybrid16")
                    options.spmv = SpmvMode::Hybrid16;
                else
                    return invalid(
                        options,
                        "spmv must be legacy, srbk, hybrid8, or hybrid16");
                spmv_was_explicit = true;
            }
            else if(argument == "--body-mode")
            {
                options.body_mode       = require_value(i, argument);
                body_mode_was_explicit = true;
            }
            else if(argument == "--newton-stop")
                options.newton_stop = require_value(i, argument);
            else if(argument == "--spmv-self-test")
                options.spmv_self_test = true;
            else if(argument == "--mas32-self-test")
                options.mas32_self_test = true;
            else if(argument == "--agipc-coarse-replay")
                options.agipc_coarse_replay_path = require_value(i, argument);
            else if(argument == "--frozen-linear-diagnostics")
                options.frozen_linear_diagnostics_path = require_value(i, argument);
            else if(argument == "--figure12-bunny-count")
                options.figure12_bunny_count = std::stoi(require_value(i, argument));
            else if(argument == "--figure12-collision-buffer-scale")
                options.figure12_collision_buffer_scale =
                    std::stod(require_value(i, argument));
            else if(argument == "--figure12-linear-system-buffer-scale")
                options.figure12_linear_system_buffer_scale =
                    std::stod(require_value(i, argument));
            else if(argument == "--frames")
            {
                options.frames       = std::stoi(require_value(i, argument));
                frames_were_explicit = true;
            }
            else if(argument == "--young-modulus")
            {
                options.young_modulus = std::stod(require_value(i, argument));
                young_modulus_was_explicit = true;
            }
            else if(argument == "--dt")
                options.dt = std::stod(require_value(i, argument));
            else if(argument == "--headless")
                options.headless = true;
            else if(argument == "--metrics-path")
                options.metrics_path = require_value(i, argument);
            else if(argument == "--fem-final-state-path")
                options.fem_final_state_path = require_value(i, argument);
            else if(argument == "--fem-checkpoint-path")
                options.fem_checkpoint_path = require_value(i, argument);
            else if(argument == "--fem-checkpoint-start-frame")
                options.fem_checkpoint_start_frame = std::stoi(require_value(i, argument));
            else if(argument == "--fem-checkpoint-stride")
                options.fem_checkpoint_stride = std::stoi(require_value(i, argument));
            else if(argument == "--agipc-threshold")
                options.agipc_threshold = std::stod(require_value(i, argument));
            else if(argument == "--agipc-mapping")
                options.agipc_mapping = require_value(i, argument);
            else if(argument == "--agipc-max-levels")
                options.agipc_max_levels = std::stoi(require_value(i, argument));
            else if(argument == "--agipc-coarse-preconditioner")
                options.agipc_coarse_preconditioner = require_value(i, argument);
            else if(argument == "--agipc-mas-validation")
                options.agipc_mas_validation = require_value(i, argument);
            else if(argument == "--agipc-mas-reuse")
                options.agipc_mas_reuse = true;
            else if(argument == "--agipc-fine-correction-iterations" || argument == "--agipc-post-cg-max")
                options.agipc_fine_correction_iterations =
                    std::stoi(require_value(i, argument));
            else if(argument == "--agipc-diagnostics")
                options.agipc_diagnostics = true;
            else if(argument == "--agipc-direction-freeze-path")
                options.agipc_direction_freeze_path = require_value(i, argument);
            else if(argument == "--agipc-direction-freeze-after-update")
                options.agipc_direction_freeze_after_update = std::stoi(require_value(i, argument));
            else if(argument == "--agipc-self-test")
                options.agipc_self_test = true;
            else
                return invalid(options, "unknown argument: " + argument);
        }
    }
    catch(const std::exception& exception)
    {
        return invalid(options, exception.what());
    }

    if(options.scene != "interactive" && options.scene != "stiff-bunny-drop"
       && options.scene != "paper-fig12-coupling-scaled"
       && options.scene != "paper-fig15-cloth-abd-scaled")
        return invalid(options,
                       "scene must be interactive, stiff-bunny-drop, "
                       "paper-fig12-coupling-scaled, or paper-fig15-cloth-abd-scaled");
    const bool paper_mixed_scene = options.scene == "paper-fig12-coupling-scaled"
                                   || options.scene == "paper-fig15-cloth-abd-scaled";
    if(options.agipc_direction_freeze_after_update < 0)
        return invalid(options,"direction freeze update must be nonnegative");
    if(options.agipc_direction_freeze_after_update > 0 && options.agipc_direction_freeze_path.empty())
        return invalid(options,"direction freeze update requires a capture directory");
    if(!options.agipc_direction_freeze_path.empty()
       && (options.solver != SolverMode::AGIPC || !options.headless
           || !options.frozen_linear_diagnostics_path.empty()))
        return invalid(options,"direction freeze requires headless agipc-core without frozen-linear diagnostics");
    if(!options.fem_checkpoint_path.empty()
       && (!options.headless || !options.agipc_direction_freeze_path.empty()
           || !options.frozen_linear_diagnostics_path.empty()))
        return invalid(options,"FEM checkpoints require a complete headless run without frozen diagnostics");
    if(options.fem_checkpoint_start_frame < 1 || options.fem_checkpoint_stride < 1)
        return invalid(options,"FEM checkpoint start frame and stride must be positive");
    if(options.agipc_coarse_preconditioner != "block-jacobi"
       && options.agipc_coarse_preconditioner != "mas32")
        return invalid(options, "agipc-coarse-preconditioner must be block-jacobi or mas32");
    if(options.agipc_coarse_preconditioner == "mas32" && options.solver == SolverMode::StiffGIPC)
        return invalid(options, "experimental coarse mas32 requires --solver agipc-core");
    if(options.agipc_mas_validation!="gpu" && options.agipc_mas_validation!="cpu" && options.agipc_mas_validation!="crosscheck")
        return invalid(options,"agipc-mas-validation must be gpu, cpu, or crosscheck");
    if(options.agipc_mas_reuse && options.agipc_coarse_preconditioner!="mas32")
        return invalid(options,"agipc-mas-reuse requires coarse mas32");
    if(options.scene == "paper-fig15-cloth-abd-scaled" && !young_modulus_was_explicit)
        options.young_modulus = 1e6;
    if(!options.cloth_mesh.empty()
       && !paper_mixed_scene)
        return invalid(options,
                       "cloth-mesh is only valid for the paper cloth scenes");
    if(options.preconditioner != "block-diagonal"
       && options.preconditioner != "gpu-mas"
       && options.preconditioner != "cemas16"
       && options.preconditioner != "cemas32")
        return invalid(options,
                       "preconditioner must be block-diagonal, gpu-mas, cemas16, or cemas32");
    if(options.preconditioner == "cemas32")
        return invalid(options,
                       "cemas32 is unavailable: BANKSIZE and sorted assets are fixed to 16");
    if(preconditioner_was_explicit
       && !paper_mixed_scene)
        return invalid(options,
                       "preconditioner is currently implemented only for paper scenes");
    if(options.body_mode != "fem" && options.body_mode != "hybrid-abd")
        return invalid(options, "body-mode must be fem or hybrid-abd");
    if(body_mode_was_explicit
       && !paper_mixed_scene)
        return invalid(options,
                       "body-mode is currently implemented only for paper scenes");
    if(options.figure12_bunny_count < 1 || options.figure12_bunny_count > 2)
        return invalid(options, "figure12-bunny-count must be 1 or 2");
    if(options.figure12_collision_buffer_scale <= 0.0
       || options.figure12_collision_buffer_scale > 1.0)
        return invalid(options,
                       "figure12-collision-buffer-scale must be in (0, 1]");
    if(options.figure12_linear_system_buffer_scale <= 0.0
       || options.figure12_linear_system_buffer_scale > 1.0)
        return invalid(options,
                       "figure12-linear-system-buffer-scale must be in (0, 1]");
    if(framework_was_explicit)
    {
        if(options.framework != "gipc" && options.framework != "srbk"
           && options.framework != "cemas-srbk"
           && options.framework != "abd-cemas-srbk")
            return invalid(options,
                           "framework must be gipc, srbk, cemas-srbk, or abd-cemas-srbk");
        if(!paper_mixed_scene)
            return invalid(options,
                           "framework presets are currently implemented only for paper scenes");
        const bool traditional_framework = options.framework == "gipc"
                                           || options.framework == "srbk";
        const SpmvMode desired_spmv = options.framework == "gipc"
                                          ? SpmvMode::Legacy
                                          : SpmvMode::SRBK;
        const std::string desired_preconditioner = traditional_framework
                                                       ? "gpu-mas"
                                                       : "cemas16";
        const std::string desired_body_mode = options.framework == "abd-cemas-srbk"
                                                  ? "hybrid-abd"
                                                  : "fem";
        if((spmv_was_explicit && options.spmv != desired_spmv)
           || (preconditioner_was_explicit
               && options.preconditioner != desired_preconditioner)
           || (body_mode_was_explicit && options.body_mode != desired_body_mode))
            return invalid(options,
                           "framework conflicts with an explicitly selected feature gate");
        options.spmv           = desired_spmv;
        options.preconditioner = desired_preconditioner;
        options.body_mode      = desired_body_mode;
    }
    else if(options.spmv == SpmvMode::Legacy && options.preconditioner == "gpu-mas"
            && options.body_mode == "fem")
    {
        options.framework = "gipc";
    }
    else if(options.spmv == SpmvMode::SRBK && options.preconditioner == "gpu-mas"
            && options.body_mode == "fem")
    {
        options.framework = "srbk";
    }
    else if(options.spmv == SpmvMode::SRBK && options.preconditioner == "cemas16"
            && options.body_mode == "fem")
    {
        options.framework = "cemas-srbk";
    }
    else if(options.spmv == SpmvMode::SRBK && options.preconditioner == "cemas16"
            && options.body_mode == "hybrid-abd")
    {
        options.framework = "abd-cemas-srbk";
    }
    else
    {
        options.framework = "custom";
    }
    if(options.scene == "paper-fig15-cloth-abd-scaled"
       && options.body_mode != "hybrid-abd")
        return invalid(options,
                       "paper-fig15-cloth-abd-scaled requires the hybrid-abd body mode");
    if(!options.frozen_linear_diagnostics_path.empty()
       && (options.scene != "paper-fig12-coupling-scaled"
           || options.preconditioner != "gpu-mas" || !options.headless))
        return invalid(options,
                       "frozen linear diagnostics requires the headless Figure 12 scene with gpu-mas");
    if(frames_were_explicit && options.frames <= 0)
        return invalid(options, "frames must be positive");
    if(options.young_modulus <= 0.0 || options.dt <= 0.0 || options.agipc_threshold <= 0.0
       || !std::isfinite(options.agipc_threshold))
        return invalid(options, "Young modulus, dt, and AGIPC threshold must be positive");
    if(options.agipc_mapping != "matching" && options.agipc_mapping != "warp-hash")
        return invalid(options, "AGIPC mapping must be matching or warp-hash");
    if(options.newton_stop != "solver-default" && options.newton_stop != "paper-current")
        return invalid(options, "Newton stop must be solver-default or paper-current");
    if(options.agipc_max_levels < 1 || options.agipc_max_levels > 16)
        return invalid(options, "AGIPC max levels must be in 1..16");
    if(options.agipc_fine_correction_iterations < 0)
        return invalid(options, "AGIPC fine correction iterations must be nonnegative");

    ParseResult result;
    result.options = std::move(options);
    result.help    = runtime_options_help();
    return result;
}
}  // namespace gipc
