#include <gipc/runtime_options.h>

#include <stdexcept>

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
           "  --scene stiff-bunny-drop|paper-fig12-coupling-scaled\n"
           "  --solver stiffgipc|agipc\n"
           "  --tet-mesh <MSH_PATH>\n"
           "  --cloth-mesh <OBJ_PATH>\n"
           "  --framework gipc|srbk|cemas-srbk|abd-cemas-srbk\n"
           "  --preconditioner block-diagonal|gpu-mas|cemas16|cemas32\n"
           "  --spmv legacy|srbk\n"
           "  --body-mode fem|hybrid-abd\n"
           "  --figure12-bunny-count 1|2\n"
           "  --figure12-collision-buffer-scale <VALUE>\n"
           "  --figure12-linear-system-buffer-scale <VALUE>\n"
           "  --spmv-self-test\n"
           "  --frames <N>\n"
           "  --young-modulus <VALUE>\n"
           "  --dt <VALUE>\n"
           "  --headless\n"
           "  --metrics-path <JSON_PATH>\n"
           "  --agipc-threshold <VALUE>\n"
           "  --agipc-mapping matching|warp-hash\n"
           "  --agipc-max-levels <N>\n"
           "  --agipc-fine-correction-iterations <N>\n"
           "  --agipc-diagnostics\n"
           "  --agipc-self-test\n"
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
                else if(value == "agipc")
                    options.solver = SolverMode::AGIPC;
                else
                    return invalid(options, "solver must be stiffgipc or agipc");
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
                else
                    return invalid(options, "spmv must be legacy or srbk");
                spmv_was_explicit = true;
            }
            else if(argument == "--body-mode")
            {
                options.body_mode       = require_value(i, argument);
                body_mode_was_explicit = true;
            }
            else if(argument == "--spmv-self-test")
                options.spmv_self_test = true;
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
                options.young_modulus = std::stod(require_value(i, argument));
            else if(argument == "--dt")
                options.dt = std::stod(require_value(i, argument));
            else if(argument == "--headless")
                options.headless = true;
            else if(argument == "--metrics-path")
                options.metrics_path = require_value(i, argument);
            else if(argument == "--agipc-threshold")
                options.agipc_threshold = std::stod(require_value(i, argument));
            else if(argument == "--agipc-mapping")
                options.agipc_mapping = require_value(i, argument);
            else if(argument == "--agipc-max-levels")
                options.agipc_max_levels = std::stoi(require_value(i, argument));
            else if(argument == "--agipc-fine-correction-iterations")
                options.agipc_fine_correction_iterations =
                    std::stoi(require_value(i, argument));
            else if(argument == "--agipc-diagnostics")
                options.agipc_diagnostics = true;
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
       && options.scene != "paper-fig12-coupling-scaled")
        return invalid(options,
                       "scene must be interactive, stiff-bunny-drop, or "
                       "paper-fig12-coupling-scaled");
    if(!options.cloth_mesh.empty()
       && options.scene != "paper-fig12-coupling-scaled")
        return invalid(options,
                       "cloth-mesh is only valid for paper-fig12-coupling-scaled");
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
       && options.scene != "paper-fig12-coupling-scaled")
        return invalid(options,
                       "preconditioner is currently implemented only for paper-fig12-coupling-scaled");
    if(options.body_mode != "fem" && options.body_mode != "hybrid-abd")
        return invalid(options, "body-mode must be fem or hybrid-abd");
    if(body_mode_was_explicit
       && options.scene != "paper-fig12-coupling-scaled")
        return invalid(options,
                       "body-mode is currently implemented only for paper-fig12-coupling-scaled");
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
        if(options.scene != "paper-fig12-coupling-scaled")
            return invalid(options,
                           "framework presets are currently implemented only for paper-fig12-coupling-scaled");
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
    if(frames_were_explicit && options.frames <= 0)
        return invalid(options, "frames must be positive");
    if(options.young_modulus <= 0.0 || options.dt <= 0.0 || options.agipc_threshold <= 0.0)
        return invalid(options, "Young modulus, dt, and AGIPC threshold must be positive");
    if(options.agipc_mapping != "matching" && options.agipc_mapping != "warp-hash")
        return invalid(options, "AGIPC mapping must be matching or warp-hash");
    if(options.agipc_max_levels < 1 || options.agipc_max_levels > 16)
        return invalid(options, "AGIPC max levels must be in 1..16");
    if(options.agipc_fine_correction_iterations <= 0)
        return invalid(options, "AGIPC fine correction iterations must be positive");

    ParseResult result;
    result.options = std::move(options);
    result.help    = runtime_options_help();
    return result;
}
}  // namespace gipc
