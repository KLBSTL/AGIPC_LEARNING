#include <gipc/runtime_options.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace
{
void require(bool condition, const char* message)
{
    if(!condition)
    {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

gipc::ParseResult parse(std::vector<std::string> arguments)
{
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for(auto& argument : arguments)
        argv.push_back(argument.data());
    return gipc::parse_runtime_options(static_cast<int>(argv.size()), argv.data());
}
}  // namespace

int main()
{
    {
        const auto result = parse({"gipc"});
        require(result.ok, "default options must parse");
        require(result.options.scene == "interactive", "default scene must remain interactive");
        require(result.options.solver == gipc::SolverMode::StiffGIPC,
                "default solver must remain StiffGIPC");
        require(result.options.frames == 0, "interactive mode must not impose a frame limit");
        require(result.options.agipc_threshold == 5e-5,
                "paper reproduction threshold must default to 5e-5");
        require(result.options.agipc_max_levels == 8, "AGIPC must default to eight levels");
        require(result.options.agipc_fine_correction_iterations == 10,
                "AGIPC must default to ten fine corrections");
    }

    {
        const auto result = parse({"gipc",
                                   "--scene",
                                   "stiff-bunny-drop",
                                   "--solver",
                                   "agipc",
                                   "--tet-mesh",
                                   "bunny2.msh",
                                   "--frames",
                                   "30",
                                   "--young-modulus",
                                   "1e7",
                                   "--dt",
                                   "0.01",
                                   "--headless",
                                   "--metrics-path",
                                   "metrics.json",
                                   "--agipc-threshold",
                                   "5e-5",
                                   "--agipc-mapping",
                                   "warp-hash",
                                   "--agipc-max-levels",
                                   "8",
                                   "--agipc-fine-correction-iterations",
                                   "10",
                                   "--agipc-diagnostics"});
        require(result.ok, "paper reproduction options must parse");
        require(result.options.solver == gipc::SolverMode::AGIPC, "AGIPC solver must be selected");
        require(result.options.frames == 30, "frame count must be retained");
        require(result.options.young_modulus == 1e7, "Young modulus must be retained");
        require(result.options.dt == 0.01, "time step must be retained");
        require(result.options.headless, "headless mode must be retained");
        require(result.options.agipc_mapping == "warp-hash", "mapping mode must be retained");
        require(result.options.agipc_diagnostics, "diagnostics flag must be retained");
    }

    {
        const auto result = parse({"gipc", "--agipc-mapping", "unknown"});
        require(!result.ok && result.exit_code == 2,
                "unknown AGIPC mapping must be rejected with exit code 2");
    }

    {
        const auto result = parse({"gipc", "--frames", "0"});
        require(!result.ok && result.exit_code == 2,
                "an explicit zero frame count must be rejected");
    }

    {
        const auto result = parse({"gipc", "--help"});
        require(result.ok && result.show_help, "help must exit before GPU startup");
        require(result.help.find("--scene stiff-bunny-drop") != std::string::npos,
                "help must describe the bunny scene");
        require(result.help.find("--agipc-self-test") != std::string::npos,
                "help must expose the algebraic self-test");
    }

    return 0;
}
