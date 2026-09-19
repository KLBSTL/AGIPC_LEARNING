#include <gipc/runtime_options.h>
#include <gipc/hierarchy_capacity.h>
#include <gipc/line_search_policy.h>
#include <limits>

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
    require(!parse({"gipc"}).options.agipc_guarded_fine_warm_start,"warm start must be opt in");
    require(!parse({"gipc","--solver","agipc-core","--agipc-guarded-fine-warm-start"}).ok,"warm start requires guarded correction");
    require(parse({"gipc","--solver","agipc-core","--agipc-guarded-fine-correction","--agipc-guarded-fine-warm-start"}).ok,"guarded warm start must parse");
    require(!parse({"gipc"}).options.agipc_guarded_fine_correction,"additional fine work must be opt in");
    require(parse({"gipc","--solver","agipc-core","--agipc-guarded-fine-correction"}).ok,"explicit guarded correction must parse");
    require(!parse({"gipc","--agipc-guarded-fine-correction"}).ok,"baseline solver must reject guarded correction");
    require(!parse({"gipc","--solver","agipc-core","--agipc-guarded-fine-correction","--frozen-linear-diagnostics","fixture"}).ok,"frozen diagnostics cannot silently adopt guarded motion");
    require(parse({"gipc"}).options.agipc_coarse_pcg_batch==0,"device PCG must be opt in");
    for(const char* batch:{"0","1","4","8","16"})require(parse({"gipc","--solver","agipc-core","--agipc-coarse-pcg-batch",batch}).ok,"bounded PCG candidate must parse");
    require(!parse({"gipc","--agipc-coarse-pcg-batch","4"}).ok,"base cannot select device PCG");
    require(!parse({"gipc","--solver","agipc-core","--agipc-coarse-pcg-batch","32"}).ok,"unbounded PCG batch rejected");
    require(parse({"gipc"}).options.agipc_mas_apply=="copy","unmeasured direct apply must be opt in");
    require(parse({"gipc","--solver","agipc-core","--agipc-mas-apply","direct"}).ok,"direct MAS candidate must parse");
    require(!parse({"gipc","--agipc-mas-apply","direct"}).ok,"base mode cannot silently select direct MAS");
    require(!parse({"gipc","--agipc-mas-apply","invalid"}).ok,"unknown MAS apply mode must fail");
    require(parse({"gipc","--solver","agipc-core","--agipc-coarse-preconditioner","mas32-factor"}).ok,
            "factorized local MAS must parse for agipc-core");
    require(!parse({"gipc","--agipc-coarse-preconditioner","mas32-factor"}).ok,
            "base solver must reject factorized local MAS");
    require(parse({"gipc"}).options.agipc_step_statistics=="auto","production statistics must default to automatic compact mode");
    for(const char* mode:{"auto","compact","history"}) {
        const auto result=parse({"gipc","--solver","agipc-core","--agipc-step-statistics",mode});
        require(result.ok && result.options.agipc_step_statistics==mode,"statistics ablation mode must parse");
    }
    require(!parse({"gipc","--agipc-step-statistics","invalid"}).ok,"unknown statistics mode must fail");
    require(parse({"gipc"}).options.agipc_mas_validation_backend=="packed","packed validation must remain default");
    require(parse({"gipc"}).options.agipc_mas_validation=="off","MAS diagnostics must be opt-in for production");
    require(parse({"gipc","--solver","agipc-core","--agipc-mas-validation","gpu"}).ok,"explicit GPU MAS diagnostics must parse");
    require(!parse({"gipc","--solver","agipc-core","--agipc-mas-validation","invalid"}).ok,"unknown MAS diagnostics mode must fail");
    require(parse({"gipc","--solver","agipc-core","--agipc-mas-validation-backend","gemm"}).ok,"explicit gemm validation must parse");
    require(!parse({"gipc","--agipc-mas-validation-backend","gemm"}).ok,"baseline numerical mode must not silently select gemm");
    require(parse({"gipc","--solver","agipc-core","--agipc-mas-validation-backend","tiled"}).ok,"explicit tiled validation must parse");
    require(!parse({"gipc","--agipc-mas-validation-backend","tiled"}).ok,"baseline must reject tiled validation");
    require(!parse({"gipc","--solver","agipc-core","--agipc-mas-validation-backend","invalid"}).ok,"unknown validation backend must fail");
    require(parse({"gipc"}).options.agipc_tiny_dense_max_dofs==0,"tiny dense must remain disabled by default");
    for(const char* limit:{"24","48","96"}) {
        const auto result=parse({"gipc","--headless","--scene","paper-fig15-cloth-abd-scaled","--solver","agipc-core","--agipc-tiny-dense-max-dofs",limit});
        require(result.ok,"explicit tiny dense candidate boundary must parse");
        require(result.options.agipc_tiny_dense_max_dofs==std::stoi(limit),"tiny dense boundary must be preserved");
    }
    require(!parse({"gipc","--agipc-tiny-dense-max-dofs","48"}).ok,"baseline solver must reject candidate numerical change");
    require(!parse({"gipc","--solver","agipc-core","--agipc-tiny-dense-max-dofs","25"}).ok,"unmeasured boundary must be rejected");
    require(parse({"gipc","--collect-timing"}).options.collect_timing,
            "stage timing must be explicitly selectable");
    require(!parse({"gipc"}).options.collect_timing,
            "production defaults must omit stage timing");
    require(!parse({"gipc","--agipc-diagnostics"}).options.collect_timing,
            "diagnostics and timing must remain independent");
    require(gipc::valid_line_search_input(1,1,0),"zero energy is a valid initial state");
    require(!gipc::valid_line_search_input(0,1,0),"zero step must fail explicitly");
    require(!gipc::valid_line_search_input(1,1,std::numeric_limits<double>::quiet_NaN()),
            "nonfinite initial energy must fail");
    require(gipc::next_line_search_alpha(1,0.1)==0.1,"backtracking must honor CCD bound");
    require(gipc::next_line_search_alpha(std::numeric_limits<double>::denorm_min(),1)==0,
            "underflow must reach a detectable failure step");
    require(gipc::hierarchy_capacity(38386,38386,4,32)==38400*4,
            "MAS capacity must include level alignment");
    require(gipc::hierarchy_capacity(1001,1024,6,32)==1024*6,
            "MAS capacity must include mapped domain");
    require(parse({"gipc","--solver","agipc-core","--agipc-post-cg-max","0"}).ok,
            "core mode and post0 ablation must parse");
    require(parse({"gipc","--solver","agipc-paper"}).ok,"paper mode must parse");
    require(parse({"gipc","--solver","agipc-symhessian",
                   "--agipc-coarse-preconditioner","mas32-factor",
                   "--agipc-coarse-pcg-batch","4","--agipc-mas-apply","direct"}).ok,
            "symmetric-Hessian AGIPC mode must accept the core paper solver controls");
    require(!parse({"gipc","--agipc-threshold","nan"}).ok,"NaN threshold must fail");
    require(parse({"gipc"}).options.agipc_affine_basis=="paper12",
            "strict paper basis must be the default");
    require(parse({"gipc","--solver","agipc-core","--agipc-affine-basis","rank-aware"}).ok,
            "rank-aware affine basis must remain an explicit ablation");
    require(!parse({"gipc","--solver","agipc-core","--agipc-affine-basis","unknown"}).ok,
            "unknown affine basis must fail");
    {
        const auto result = parse({"gipc"});
        require(result.ok, "default options must parse");
        require(result.options.scene == "interactive", "default scene must remain interactive");
        require(result.options.solver == gipc::SolverMode::StiffGIPC,
                "default solver must remain StiffGIPC");
        require(result.options.frames == 0, "interactive mode must not impose a frame limit");
        require(result.options.agipc_threshold == 5e-5,
                "paper reproduction threshold must default to 5e-5");
        require(result.options.agipc_max_levels == 16, "AGIPC must preserve its sixteen-level runtime default");
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

    {
        const auto valid=parse({"gipc","--headless","--solver","agipc-core",
            "--scene","paper-fig15-cloth-abd-scaled","--frames","60",
            "--agipc-direction-freeze-path","capture","--agipc-direction-freeze-continue",
            "--agipc-direction-freeze-frames","27,35,45,60","--fem-checkpoint-path","positions.bin"});
        require(valid.ok && valid.options.agipc_direction_freeze_frames.size()==4,
                "capture-and-continue permits complete contact motion and checkpoints");
        for(const auto* frames: {"0,35","35,27","27,27","27,","27junk","27,61"})
        {
            const auto invalid=parse({"gipc","--headless","--solver","agipc-core",
                "--scene","paper-fig15-cloth-abd-scaled","--frames","60",
                "--agipc-direction-freeze-path","capture","--agipc-direction-freeze-continue",
                "--agipc-direction-freeze-frames",frames});
            require(!invalid.ok,"malformed or out-of-run capture frames must be rejected");
        }
        const auto short_run=parse({"gipc","--headless","--solver","agipc-core",
            "--frames","59","--agipc-direction-freeze-path","capture",
            "--agipc-direction-freeze-continue","--agipc-direction-freeze-frames","27"});
        require(!short_run.ok,"capture-and-continue must not authorize a short motion run");
    }
    {
        const auto valid=parse({"gipc","--headless","--scene","paper-fig15-cloth-abd-scaled",
            "--solver","agipc-core","--preconditioner","cemas16","--agipc-fine-replay","state"});
        require(valid.ok,"native fine frozen replay must parse");
        const auto incompatible=parse({"gipc","--headless","--scene","paper-fig15-cloth-abd-scaled",
            "--solver","agipc-core","--preconditioner","cemas32","--agipc-fine-replay","state"});
        require(!incompatible.ok,"fine replay must reject incompatible preconditioner");
        const auto missing_continue=parse({"gipc","--agipc-direction-freeze-late-after-newton","20"});
        require(!missing_continue.ok,"late sampling requires capture-and-continue");
        const auto late=parse({"gipc","--headless","--scene","paper-fig15-cloth-abd-scaled",
            "--solver","agipc-core","--frames","60","--agipc-direction-freeze-path","state",
            "--agipc-direction-freeze-continue","--agipc-direction-freeze-frames","27,35,45,60",
            "--agipc-direction-freeze-late-after-newton","20"});
        require(late.ok && late.options.agipc_direction_freeze_late_after_newton==20,
                "bounded early/late complete contact capture must parse");
    }
    return 0;
}
