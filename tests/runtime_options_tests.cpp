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
        require(result.options.spmv == gipc::SpmvMode::SRBK,
                "the integrated c499 path must keep SRBK as its default SpMV");
        require(result.options.preconditioner == "cemas16",
                "the integrated c499 path must keep CEMAS16 as its default preconditioner");
        require(result.options.body_mode == "hybrid-abd",
                "the integrated Figure 12 path must keep hybrid ABD as its default body mode");
        require(result.options.framework == "abd-cemas-srbk",
                "the default feature combination must report its effective framework");
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
        const auto result = parse({"gipc",
                                   "--scene",
                                   "paper-fig12-coupling-scaled",
                                   "--framework",
                                   "cemas-srbk",
                                   "--frames",
                                   "1",
                                   "--headless"});
        require(result.ok, "the supported CEMAS+SRBK preset must parse");
        require(result.options.spmv == gipc::SpmvMode::SRBK,
                "CEMAS+SRBK must select SRBK SpMV");
        require(result.options.preconditioner == "cemas16",
                "CEMAS+SRBK must select the verified CEMAS16 path");
        require(result.options.body_mode == "fem",
                "CEMAS+SRBK must retain full FEM bodies");
    }

    {
        const auto result = parse({"gipc",
                                   "--scene",
                                   "paper-fig12-coupling-scaled",
                                   "--framework",
                                   "abd-cemas-srbk",
                                   "--frames",
                                   "1",
                                   "--headless"});
        require(result.ok, "the supported ABD+CEMAS+SRBK preset must parse");
        require(result.options.body_mode == "hybrid-abd",
                "ABD+CEMAS+SRBK must select the hybrid body representation");
    }

    {
        const auto result = parse({"gipc",
                                   "--scene",
                                   "paper-fig12-coupling-scaled",
                                   "--spmv",
                                   "legacy",
                                   "--preconditioner",
                                   "block-diagonal",
                                   "--body-mode",
                                   "fem"});
        require(result.ok, "orthogonal supported feature gates must parse");
        require(result.options.framework == "custom",
                "a non-paper feature combination must be labelled custom");
        require(result.options.preconditioner == "block-diagonal",
                "block diagonal must be retained");
        require(result.options.body_mode == "fem", "FEM body mode must be retained");
    }

    {
        const auto result = parse({"gipc",
                                   "--scene",
                                   "paper-fig12-coupling-scaled",
                                   "--framework",
                                   "cemas-srbk",
                                   "--body-mode",
                                   "hybrid-abd"});
        require(!result.ok && result.exit_code == 2,
                "framework and explicit feature conflicts must return exit code 2");
    }

    {
        const auto result = parse({"gipc",
                                   "--scene",
                                   "paper-fig12-coupling-scaled",
                                   "--preconditioner",
                                   "gpu-mas",
                                   "--body-mode",
                                   "fem"});
        require(result.ok, "the recovered traditional GPU MAS must parse");
        require(result.options.framework == "srbk",
                "GPU MAS with the default SRBK SpMV must report the SRBK framework");
    }

    {
        const auto result = parse({"gipc", "--preconditioner", "cemas32"});
        require(!result.ok && result.exit_code == 2,
                "CEMAS32 must remain unavailable until its sorted assets are implemented");
    }

    {
        const auto result = parse({"gipc",
                                   "--scene",
                                   "stiff-bunny-drop",
                                   "--preconditioner",
                                   "block-diagonal"});
        require(!result.ok && result.exit_code == 2,
                "a preconditioner gate must not be accepted by a scene that still overrides it");
    }

    {
        const auto result = parse({"gipc",
                                   "--scene",
                                   "paper-fig12-coupling-scaled",
                                   "--framework",
                                   "gipc"});
        require(result.ok, "the GIPC framework must parse after GPU MAS recovery");
        require(result.options.spmv == gipc::SpmvMode::Legacy,
                "GIPC must select legacy SpMV");
        require(result.options.preconditioner == "gpu-mas",
                "GIPC must select traditional GPU MAS");
        require(result.options.body_mode == "fem", "GIPC must use full FEM bodies");
    }

    {
        const auto result = parse({"gipc",
                                   "--scene",
                                   "paper-fig12-coupling-scaled",
                                   "--framework",
                                   "srbk"});
        require(result.ok, "the SRBK framework must parse after GPU MAS recovery");
        require(result.options.spmv == gipc::SpmvMode::SRBK,
                "SRBK must select the warp-reduced SpMV");
        require(result.options.preconditioner == "gpu-mas",
                "SRBK must retain the traditional GPU MAS preconditioner");
        require(result.options.body_mode == "fem", "SRBK must use full FEM bodies");
    }

    {
        const auto result = parse({"gipc",
                                   "--scene",
                                   "paper-fig12-coupling-scaled",
                                   "--spmv",
                                   "legacy",
                                   "--frames",
                                   "1",
                                   "--headless"});
        require(result.ok, "the Figure 12 benchmark must accept legacy SpMV");
        require(result.options.spmv == gipc::SpmvMode::Legacy,
                "legacy SpMV must be retained by the parser");
    }

    {
        const auto result = parse({"gipc",
                                   "--scene",
                                   "paper-fig12-coupling-scaled",
                                   "--spmv",
                                   "hybrid8",
                                   "--preconditioner",
                                   "gpu-mas",
                                   "--body-mode",
                                   "fem"});
        require(result.ok, "the experimental threshold-8 hybrid SpMV must parse");
        require(std::string(gipc::to_string(result.options.spmv)) == "hybrid8",
                "the threshold-8 hybrid SpMV must be retained by the parser");
        require(result.options.framework == "custom",
                "an experimental hybrid SpMV must not impersonate a paper framework");
    }

    {
        const auto result = parse({"gipc",
                                   "--scene",
                                   "paper-fig12-coupling-scaled",
                                   "--spmv",
                                   "hybrid16",
                                   "--preconditioner",
                                   "gpu-mas",
                                   "--body-mode",
                                   "fem"});
        require(result.ok, "the experimental threshold-16 hybrid SpMV must parse");
        require(std::string(gipc::to_string(result.options.spmv)) == "hybrid16",
                "the threshold-16 hybrid SpMV must be retained by the parser");
    }

    {
        const auto result = parse({"gipc", "--spmv", "unknown"});
        require(!result.ok && result.exit_code == 2,
                "unknown SpMV modes must be rejected with exit code 2");
    }

    {
        const auto result = parse({"gipc", "--spmv-self-test"});
        require(result.ok && result.options.spmv_self_test,
                "the fixed-Hessian SpMV equivalence self-test must parse");
    }

    {
        const auto result = parse({"gipc", "--mas32-self-test"});
        require(result.ok && result.options.mas32_self_test,
                "the MAS32 hierarchy self-test must parse");
    }

    {
        const auto result = parse({"gipc",
                                   "--scene",
                                   "paper-fig12-coupling-scaled",
                                   "--framework",
                                   "gipc",
                                   "--headless",
                                   "--frozen-linear-diagnostics",
                                   "frozen.json"});
        require(result.ok
                    && result.options.frozen_linear_diagnostics_path == "frozen.json",
                "Figure 12 GPU MAS32 must accept frozen linear diagnostics");
    }

    {
        const auto result = parse({"gipc",
                                   "--scene",
                                   "paper-fig12-coupling-scaled",
                                   "--cloth-mesh",
                                   "cloth_129x129.obj",
                                   "--frames",
                                   "1",
                                   "--headless"});
        require(result.ok,
                "the reduced Figure 12 coupling scene must accept a P2 cloth mesh override");
        require(result.options.scene == "paper-fig12-coupling-scaled",
                "the Figure 12 scene name must be retained");
    }

    {
        const auto result = parse({"gipc",
                                   "--scene",
                                   "stiff-bunny-drop",
                                   "--cloth-mesh",
                                   "cloth_129x129.obj"});
        require(!result.ok && result.exit_code == 2,
                "cloth-mesh must not be silently ignored by non-cloth scenes");
    }

    {
        const auto result = parse({"gipc", "--help"});
        require(result.ok && result.show_help, "help must exit before GPU startup");
        require(result.help.find("--scene stiff-bunny-drop") != std::string::npos,
                "help must describe the bunny scene");
        require(result.help.find("--agipc-self-test") != std::string::npos,
                "help must expose the algebraic self-test");
        require(result.help.find("--cloth-mesh") != std::string::npos,
                "help must expose the Figure 12 cloth mesh override");
        require(result.help.find("--spmv legacy|srbk|hybrid8|hybrid16")
                    != std::string::npos,
                "help must expose the baseline and experimental SpMV implementations");
        require(result.help.find("--spmv-self-test") != std::string::npos,
                "help must expose the fixed-Hessian equivalence gate");
        require(result.help.find("--mas32-self-test") != std::string::npos,
                "help must expose the MAS32 hierarchy equivalence gate");
        require(result.help.find("--frozen-linear-diagnostics") != std::string::npos,
                "help must expose the frozen linear-system diagnostic gate");
        require(result.help.find("--framework gipc|srbk|cemas-srbk|abd-cemas-srbk")
                    != std::string::npos,
                "help must expose the benchmark framework contract");
        require(result.help.find("--preconditioner block-diagonal|gpu-mas|cemas16|cemas32")
                    != std::string::npos,
                "help must expose the preconditioner contract");
        require(result.help.find("--body-mode fem|hybrid-abd") != std::string::npos,
                "help must expose the body representation contract");
    }

    return 0;
}
