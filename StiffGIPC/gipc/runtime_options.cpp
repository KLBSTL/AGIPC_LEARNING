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
           "  --scene stiff-bunny-drop\n"
           "  --solver stiffgipc|agipc\n"
           "  --tet-mesh <MSH_PATH>\n"
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

    if(options.scene != "interactive" && options.scene != "stiff-bunny-drop")
        return invalid(options, "scene must be interactive or stiff-bunny-drop");
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
