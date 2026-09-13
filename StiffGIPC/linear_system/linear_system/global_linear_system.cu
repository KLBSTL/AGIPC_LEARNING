#include <linear_system/linear_system/global_linear_system.h>
#include <linear_system/linear_system/i_linear_system_solver.h>
#include <linear_system/linear_system/i_preconditioner.h>
#include <cuda_tools/cuda_tools.h>
#include <gipc/utils/timer.h>
#include <linear_system/solver/pcg_solver.h>
#include <linear_system/preconditioner/traditional_mas32_preconditioner.h>
#include <agipc/agipc_criterion.cuh>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

namespace
{
struct BuildStageTiming
{
    double fine_assembly_ms=0;
    double adaptive_pipeline_ms=0;
    double total_ms=0;
};

thread_local BuildStageTiming last_build_timing;

template <typename T>
void write_binary(const std::filesystem::path& path, const std::vector<T>& values)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
}
}  // namespace

namespace gipc
{
bool GlobalLinearSystem::build_linear_system()
{
    last_build_timing={};
    auto hessian_provider_count  = m_subsystems.size();
    auto gradient_provider_count = m_inner_subsystems.size();

    // right hand side can only be provided by both LinearSubsystem
    m_rhs_count_per_subsystem.resize(gradient_provider_count);
    m_rhs_offset_per_subsystem.resize(gradient_provider_count);

    for(auto& subsystem : m_subsystems)
        subsystem->report_subsystem_info();

    for(auto& gp : m_inner_subsystems)
    {
        auto i                       = gp->gid();
        m_rhs_count_per_subsystem[i] = gp->right_hand_side_dof();
    }

    std::exclusive_scan(m_rhs_count_per_subsystem.begin(),
                        m_rhs_count_per_subsystem.end(),
                        m_rhs_offset_per_subsystem.begin(),
                        0);

    for(auto& gp : m_inner_subsystems)
    {
        auto i = gp->gid();
        gp->dof_offset(m_rhs_offset_per_subsystem[i]);
    }

    auto total_rhs_count =
        m_rhs_offset_per_subsystem.back() + m_rhs_count_per_subsystem.back();


    if(gipc_global_triplet->global_triplet_offset == 0 || total_rhs_count == 0)
    {
        std::cout << "The global linear system is empty, skip *assembling, *solving and *solution distributing phase."
                  << std::endl;
        return false;
    }

    const bool measure_stages=agipc::galerkin_adoption_enabled();
    cudaEvent_t build_start=nullptr,fine_end=nullptr,adaptive_end=nullptr,build_end=nullptr;
    if(measure_stages)
    {
        CUDA_SAFE_CALL(cudaEventCreate(&build_start));
        CUDA_SAFE_CALL(cudaEventCreate(&fine_end));
        CUDA_SAFE_CALL(cudaEventCreate(&adaptive_end));
        CUDA_SAFE_CALL(cudaEventCreate(&build_end));
        CUDA_SAFE_CALL(cudaEventRecord(build_start));
    }


    m_b.resize(total_rhs_count);
    m_x.resize(total_rhs_count);

    auto rhs_view = m_b.view();

    for(auto& subsystem : m_subsystems)
        subsystem->do_assemble(rhs_view);

    int start_preconditioner_id = 0;
    if(m_local_preconditioners.size() && m_local_preconditioners[0]->preconditioner_id == 0)
    {
        m_local_preconditioners[0]->assemble();
        start_preconditioner_id++;
    }
    convert_new();

    if(measure_stages) CUDA_SAFE_CALL(cudaEventRecord(fine_end));

    // Assemble the adaptive candidate after the fine matrix has been reduced to unique BCOO.
    agipc::update_galerkin_shadow(*gipc_global_triplet,
                                  m_b.buffer_view().data(),
                                  total_rhs_count);

    if(measure_stages) CUDA_SAFE_CALL(cudaEventRecord(adaptive_end));

    // An accepted Galerkin candidate does not use the fine preconditioners.
    // Defer them on the adoption path and assemble them in solve_linear_system
    // only when the candidate falls back to the original fine solver.
    const bool frozen_diagnostics_pending=
        !m_frozen_linear_diagnostics_path.empty()
        && !m_frozen_linear_diagnostics_complete;
    if(!measure_stages || frozen_diagnostics_pending)
    {
        if(m_global_preconditioner)
            m_global_preconditioner->do_assemble(*gipc_global_triplet);

        for(int i=start_preconditioner_id;i<m_local_preconditioners.size();i++)
            m_local_preconditioners[i]->assemble();
    }

    if(measure_stages)
    {
        CUDA_SAFE_CALL(cudaEventRecord(build_end));
        CUDA_SAFE_CALL(cudaEventSynchronize(build_end));
        float fine_ms=0,adaptive_ms=0,total_ms=0;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&fine_ms,build_start,fine_end));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&adaptive_ms,fine_end,adaptive_end));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&total_ms,build_start,build_end));
        last_build_timing={fine_ms,adaptive_ms,total_ms};
        CUDA_SAFE_CALL(cudaEventDestroy(build_start));
        CUDA_SAFE_CALL(cudaEventDestroy(fine_end));
        CUDA_SAFE_CALL(cudaEventDestroy(adaptive_end));
        CUDA_SAFE_CALL(cudaEventDestroy(build_end));
    }

    return true;
}

void GlobalLinearSystem::distribute_solution()
{
    auto x_view = std::as_const(m_x).view();

    for(auto& subsystem : m_inner_subsystems)
        subsystem->do_retrieve_solution(x_view);

    wait_device();
}

DiagonalSubsystem& GlobalLinearSystem::_create_subsystem(U<DiagonalSubsystem>&& subsystem)
{
    auto ptr = subsystem.get();
    ptr->gid(m_inner_subsystems.size());
    m_inner_subsystems.push_back(ptr);  // push to gradient providers

    ptr->hid(m_subsystems.size());
    ptr->system(*this);
    m_subsystems.emplace_back(std::move(subsystem));  // push to hessian providers

    return *ptr;
}



IterativeSolver& GlobalLinearSystem::_create_solver(U<IterativeSolver>&& solver)
{
    m_solver = std::move(solver);
    m_solver->system(*this);
    return *m_solver;
}

GlobalLinearSystem::~GlobalLinearSystem() {}

LocalPreconditioner& GlobalLinearSystem::_create_preconditioner(U<LocalPreconditioner>&& preconditioner)
{
    preconditioner->system(*this);
    return *m_local_preconditioners.emplace_back(std::move(preconditioner));
}

GlobalPreconditioner& GlobalLinearSystem::_create_preconditioner(U<GlobalPreconditioner>&& preconditioner)
{
    CT_ASSERT(m_global_preconditioner == nullptr, "Global preconditioner already exists.");
    preconditioner->system(*this);
    m_global_preconditioner = std::move(preconditioner);
    return *m_global_preconditioner;
}

gipc::SizeT GlobalLinearSystem::solve_linear_system()
{
    bool success = build_linear_system();
    if(!success)
        return 0;
    CT_ASSERT(m_solver, "Solver is null, call create_solver() to setup a solver.");
    if(!m_frozen_linear_diagnostics_path.empty()
       && !m_frozen_linear_diagnostics_complete)
    {
        run_frozen_linear_diagnostics();
        m_frozen_linear_diagnostics_complete = true;
        m_x.buffer_view().fill(0);
        return 0;
    }
    const bool measure_stages=agipc::galerkin_adoption_enabled();
    cudaEvent_t decision_start=nullptr,decision_end=nullptr,preconditioner_end=nullptr,
                solver_end=nullptr,distribute_end=nullptr;
    if(measure_stages)
    {
        CUDA_SAFE_CALL(cudaEventCreate(&decision_start));
        CUDA_SAFE_CALL(cudaEventCreate(&decision_end));
        CUDA_SAFE_CALL(cudaEventCreate(&preconditioner_end));
        CUDA_SAFE_CALL(cudaEventCreate(&solver_end));
        CUDA_SAFE_CALL(cudaEventCreate(&distribute_end));
        CUDA_SAFE_CALL(cudaEventRecord(decision_start));
    }
    const auto adoption=agipc::adopt_galerkin_candidate(m_x.data(),m_x.size());
    const bool adopted=adoption.value("adopted",false);
    if(measure_stages) CUDA_SAFE_CALL(cudaEventRecord(decision_end));

    if(measure_stages && !adopted)
    {
        if(m_global_preconditioner)
            m_global_preconditioner->do_assemble(*gipc_global_triplet);

        int start_preconditioner_id=0;
        if(m_local_preconditioners.size()
           && m_local_preconditioners[0]->preconditioner_id==0)
            start_preconditioner_id=1;
        for(int i=start_preconditioner_id;i<m_local_preconditioners.size();i++)
            m_local_preconditioners[i]->assemble();
    }
    if(measure_stages) CUDA_SAFE_CALL(cudaEventRecord(preconditioner_end));

    auto iter=adopted ? adoption.value("iterations",0) : m_solver->solve(m_x,m_b);
    if(measure_stages) CUDA_SAFE_CALL(cudaEventRecord(solver_end));
    distribute_solution();
    if(measure_stages)
    {
        CUDA_SAFE_CALL(cudaEventRecord(distribute_end));
        CUDA_SAFE_CALL(cudaEventSynchronize(distribute_end));
        float decision_ms=0,preconditioner_ms=0,solver_ms=0,distribute_ms=0,solve_ms=0;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&decision_ms,decision_start,decision_end));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&preconditioner_ms,decision_end,preconditioner_end));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&solver_ms,preconditioner_end,solver_end));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&distribute_ms,solver_end,distribute_end));
        CUDA_SAFE_CALL(cudaEventElapsedTime(&solve_ms,decision_start,distribute_end));
        agipc::record_linear_solve_timing({
            {"fine_assembly",last_build_timing.fine_assembly_ms},
            {"adaptive_pipeline",last_build_timing.adaptive_pipeline_ms},
            {"fine_preconditioner",preconditioner_ms},
            {"candidate_decision",decision_ms},
            {"fallback_fine_solve",adopted?0.0:static_cast<double>(solver_ms)},
            {"distribute",distribute_ms},{"build_total",last_build_timing.total_ms},
            {"solve_total",last_build_timing.total_ms+solve_ms},
            {"used_galerkin_candidate",adopted},{"fine_solver_iterations",adopted?0:iter}});
        CUDA_SAFE_CALL(cudaEventDestroy(decision_start));
        CUDA_SAFE_CALL(cudaEventDestroy(decision_end));
        CUDA_SAFE_CALL(cudaEventDestroy(preconditioner_end));
        CUDA_SAFE_CALL(cudaEventDestroy(solver_end));
        CUDA_SAFE_CALL(cudaEventDestroy(distribute_end));
    }
    if(measure_stages && !adopted)
        agipc::record_fallback_direction_quality(*gipc_global_triplet,
                                                 m_b.data(),m_x.data(),m_x.size());
    if(adopted)
        agipc::freeze_accepted_direction(*gipc_global_triplet,m_b.data(),m_x.size());
    return iter;
}

Json GlobalLinearSystem::as_json() const
{
    Json j;
    j["solver"]     = typeid(*m_solver).name();
    j["subsystems"] = Json::array();
    for(auto& s : m_subsystems)
    {
        j["subsystems"].push_back(s->as_json());
    }
    j["preconditioners"] = Json::array();
    for(auto& p : m_local_preconditioners)
    {
        j["preconditioners"].push_back(p->as_json());
    }
    return j;
}

void GlobalLinearSystem::apply_preconditioner(cudatool::DenseVectorView<Float>  z,
                                              cudatool::CDenseVectorView<Float> r)
{
    // first apply global preconditioner
    if(m_global_preconditioner)
        m_global_preconditioner->do_apply(r, z);
    else  // if no global preconditioner, use identity
        z.buffer_view().copy_from(r.buffer_view());

    // then apply local preconditioners
    // it's user's choice to rewrite or reuse the global preconditioner
    for(auto& p : m_local_preconditioners)
        p->do_apply(r, z);
}



void GlobalLinearSystem::convert_new()
{
    m_converter.convert(*gipc_global_triplet,
                        0,
                        gipc_global_triplet->global_triplet_offset,
                        gipc_global_triplet->global_triplet_offset);
//#ifndef SymGH
//    m_converter.ge2sym(*gipc_global_triplet);
//#endif
}



void GlobalLinearSystem::spmv(Float                         a,
                              cudatool::CDenseVectorView<Float> x,
                              Float                         b,
                              cudatool::DenseVectorView<Float>  y)
{

    if(m_spmv_mode == SpmvMode::Legacy)
    {
        m_spmv.legacy_sym_spmv(a,
                               gipc_global_triplet->block_values(),
                               gipc_global_triplet->block_row_indices(),
                               gipc_global_triplet->block_col_indices(),
                               gipc_global_triplet->h_unique_key_number,
                               x,
                               b,
                               y);
    }
    else if(m_spmv_mode == SpmvMode::SRBK)
    {
        m_spmv.warp_reduce_sym_spmv(a,
                                    gipc_global_triplet->block_values(),
                                    gipc_global_triplet->block_row_indices(),
                                    gipc_global_triplet->block_col_indices(),
                                    gipc_global_triplet->h_unique_key_number,
                                    x,
                                    b,
                                    y);
    }
    else
    {
        const int threshold = m_spmv_mode == SpmvMode::Hybrid8 ? 8 : 16;
        m_spmv.hybrid_sym_spmv(a,
                               gipc_global_triplet->block_values(),
                               gipc_global_triplet->block_row_indices(),
                               gipc_global_triplet->block_col_indices(),
                               gipc_global_triplet->h_unique_key_number,
                               x,
                               b,
                               y,
                               threshold);
    }
}

void GlobalLinearSystem::run_frozen_linear_diagnostics()
{
    const auto diagnostic_mode = m_spmv_mode;
    const bool diagnostic_is_hybrid = diagnostic_mode == SpmvMode::Hybrid8
                                      || diagnostic_mode == SpmvMode::Hybrid16;
    const std::filesystem::path json_path = m_frozen_linear_diagnostics_path;
    const auto output_dir = json_path.has_parent_path() ? json_path.parent_path()
                                                        : std::filesystem::current_path();
    std::filesystem::create_directories(output_dir);

    const int triplet_count = gipc_global_triplet->h_unique_key_number;
    std::vector<Eigen::Matrix3d> matrix_values(triplet_count);
    std::vector<int> matrix_rows(triplet_count);
    std::vector<int> matrix_cols(triplet_count);
    std::vector<Float> rhs;
    CUDA_SAFE_CALL(cudaMemcpy(matrix_values.data(),
                              gipc_global_triplet->block_values(),
                              matrix_values.size() * sizeof(Eigen::Matrix3d),
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(matrix_rows.data(),
                              gipc_global_triplet->block_row_indices(),
                              matrix_rows.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(matrix_cols.data(),
                              gipc_global_triplet->block_col_indices(),
                              matrix_cols.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));
    m_b.copy_to(rhs);

    const auto values_path = output_dir / "frozen_A_values.bin";
    const auto rows_path   = output_dir / "frozen_A_rows.bin";
    const auto cols_path   = output_dir / "frozen_A_cols.bin";
    const auto rhs_path    = output_dir / "frozen_b.bin";
    write_binary(values_path, matrix_values);
    write_binary(rows_path, matrix_rows);
    write_binary(cols_path, matrix_cols);
    write_binary(rhs_path, rhs);

    std::vector<Float> host_x(rhs.size());
    for(size_t i = 0; i < host_x.size(); ++i)
        host_x[i] = std::sin(0.013 * static_cast<double>(i + 1))
                    + 0.25 * std::cos(0.007 * static_cast<double>(i + 3));
    cudatool::DeviceDenseVector<Float> diagnostic_x;
    cudatool::DeviceDenseVector<Float> legacy_y(rhs.size());
    cudatool::DeviceDenseVector<Float> srbk_y(rhs.size());
    cudatool::DeviceDenseVector<Float> selected_y(rhs.size());
    diagnostic_x.copy_from(host_x);
    m_spmv.legacy_sym_spmv(1.0,
                           gipc_global_triplet->block_values(),
                           gipc_global_triplet->block_row_indices(),
                           gipc_global_triplet->block_col_indices(),
                           triplet_count,
                           diagnostic_x.cview(),
                           0.0,
                           legacy_y.view());
    m_spmv.warp_reduce_sym_spmv(1.0,
                                gipc_global_triplet->block_values(),
                                gipc_global_triplet->block_row_indices(),
                                gipc_global_triplet->block_col_indices(),
                                triplet_count,
                                diagnostic_x.cview(),
                                0.0,
                                srbk_y.view());
    if(diagnostic_is_hybrid)
    {
        const int threshold = diagnostic_mode == SpmvMode::Hybrid8 ? 8 : 16;
        m_spmv.hybrid_sym_spmv(1.0,
                               gipc_global_triplet->block_values(),
                               gipc_global_triplet->block_row_indices(),
                               gipc_global_triplet->block_col_indices(),
                               triplet_count,
                               diagnostic_x.cview(),
                               0.0,
                               selected_y.view(),
                               threshold);
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    std::vector<Float> host_legacy_y;
    std::vector<Float> host_srbk_y;
    std::vector<Float> host_selected_y;
    legacy_y.copy_to(host_legacy_y);
    srbk_y.copy_to(host_srbk_y);
    if(diagnostic_is_hybrid)
        selected_y.copy_to(host_selected_y);
    long double diff_sq = 0.0;
    long double legacy_sq = 0.0;
    long double selected_diff_sq = 0.0;
    double max_abs_error = 0.0;
    for(size_t i = 0; i < host_legacy_y.size(); ++i)
    {
        const double difference = host_legacy_y[i] - host_srbk_y[i];
        diff_sq += static_cast<long double>(difference) * difference;
        legacy_sq += static_cast<long double>(host_legacy_y[i]) * host_legacy_y[i];
        max_abs_error = std::max(max_abs_error, std::abs(difference));
        if(diagnostic_is_hybrid)
        {
            const double selected_difference =
                host_legacy_y[i] - host_selected_y[i];
            selected_diff_sq += static_cast<long double>(selected_difference)
                                * selected_difference;
        }
    }
    const double spmv_relative_error =
        std::sqrt(static_cast<double>(diff_sq / std::max(legacy_sq, 1e-300L)));
    const double selected_spmv_relative_error =
        diagnostic_is_hybrid
            ? std::sqrt(static_cast<double>(selected_diff_sq
                                            / std::max(legacy_sq, 1e-300L)))
            : 0.0;

    cudatool::DeviceDenseVector<Float> preconditioned_rhs(rhs.size());
    apply_preconditioner(preconditioned_rhs.view(), m_b.cview());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    std::vector<Float> host_z;
    preconditioned_rhs.copy_to(host_z);
    long double rhs_t_z = 0.0;
    for(size_t i = 0; i < rhs.size(); ++i)
        rhs_t_z += static_cast<long double>(rhs[i]) * host_z[i];

    Json mas32_numerical;
    bool has_mas32_numerical = false;
    for(const auto& preconditioner : m_local_preconditioners)
    {
        if(const auto* mas32 =
               dynamic_cast<const TraditionalMAS32_Preconditioner*>(preconditioner.get()))
        {
            mas32_numerical = mas32->numerical_diagnostics(m_b.cview());
            has_mas32_numerical = true;
            break;
        }
    }

    auto* pcg = dynamic_cast<PCGSolver*>(m_solver.get());
    CT_ASSERT(pcg, "frozen linear diagnostics requires PCGSolver");
    const auto original_mode = diagnostic_mode;
    m_spmv_mode = SpmvMode::Legacy;
    Json legacy_trace = pcg->trace(m_x.view(), m_b.cview(), 100);
    m_spmv_mode = SpmvMode::SRBK;
    Json srbk_trace = pcg->trace(m_x.view(), m_b.cview(), 100);
    Json selected_trace;
    if(diagnostic_is_hybrid)
    {
        m_spmv_mode = original_mode;
        selected_trace = pcg->trace(m_x.view(), m_b.cview(), 100);
    }
    m_spmv_mode = original_mode;

    int first_residual_divergence = -1;
    const auto trace_count = std::min(legacy_trace["iterations"].size(),
                                      srbk_trace["iterations"].size());
    for(size_t i = 0; i < trace_count; ++i)
    {
        const double legacy_residual =
            legacy_trace["iterations"][i]["residual_norm"].get<double>();
        const double srbk_residual =
            srbk_trace["iterations"][i]["residual_norm"].get<double>();
        const double relative_difference =
            std::abs(legacy_residual - srbk_residual)
            / std::max(std::abs(legacy_residual), 1e-300);
        if(relative_difference > 1e-3)
        {
            first_residual_divergence = static_cast<int>(i) + 1;
            break;
        }
    }

    auto trace_is_spd = [](const Json& trace) {
        if(trace["initial_rtz"].get<double>() <= 0.0)
            return false;
        for(const auto& iteration : trace["iterations"])
            if(iteration.contains("spd") && !iteration["spd"].get<bool>())
                return false;
        return true;
    };

    Json report;
    report["test"] = "figure12_first_frame_first_newton_frozen_linear_system";
    report["frozen_in_memory"] = true;
    report["matrix_block_rows"] = gipc_global_triplet->block_rows();
    report["matrix_block_cols"] = gipc_global_triplet->block_cols();
    report["matrix_triplets"] = triplet_count;
    report["scalar_dofs"] = rhs.size();
    report["artifacts"] = {{"A_values", values_path.filename().string()},
                             {"A_rows", rows_path.filename().string()},
                             {"A_cols", cols_path.filename().string()},
                             {"b", rhs_path.filename().string()}};
    report["spmv"] = {{"relative_error", spmv_relative_error},
                        {"max_abs_error", max_abs_error},
                        {"passed", spmv_relative_error <= 1e-10}};
    if(diagnostic_is_hybrid)
        report["selected_spmv"] = {
            {"mode", to_string(diagnostic_mode)},
            {"relative_error", selected_spmv_relative_error},
            {"passed", selected_spmv_relative_error <= 1e-10}};
    report["preconditioner"] = {{"rhs_t_z", static_cast<double>(rhs_t_z)},
                                  {"rhs_t_z_positive", rhs_t_z > 0.0},
                                  {"same_instance_reused", true}};
    if(has_mas32_numerical)
        report["preconditioner"]["mas32_numerical"] =
            std::move(mas32_numerical);
    report["legacy_pcg"] = std::move(legacy_trace);
    report["srbk_pcg"] = std::move(srbk_trace);
    if(diagnostic_is_hybrid)
        report["selected_pcg"] = std::move(selected_trace);
    report["first_residual_divergence_iteration"] = first_residual_divergence;
    report["gates"] = {{"spmv_equivalent", spmv_relative_error <= 1e-10},
                         {"rhs_preconditioner_spd", rhs_t_z > 0.0},
                         {"legacy_trace_spd", trace_is_spd(report["legacy_pcg"])},
                         {"srbk_trace_spd", trace_is_spd(report["srbk_pcg"])}};
    if(diagnostic_is_hybrid)
    {
        report["gates"]["selected_spmv_equivalent"] =
            selected_spmv_relative_error <= 1e-10;
        report["gates"]["selected_trace_spd"] =
            trace_is_spd(report["selected_pcg"]);
    }
    report["passed"] = report["gates"]["spmv_equivalent"].get<bool>()
                       && report["gates"]["rhs_preconditioner_spd"].get<bool>()
                       && report["gates"]["legacy_trace_spd"].get<bool>()
                       && report["gates"]["srbk_trace_spd"].get<bool>();
    if(diagnostic_is_hybrid)
        report["passed"] = report["passed"].get<bool>()
                           && report["gates"]["selected_spmv_equivalent"].get<bool>()
                           && report["gates"]["selected_trace_spd"].get<bool>();

    std::ofstream output(json_path, std::ios::trunc);
    output << report.dump(2) << '\n';
    std::cout << "Frozen linear diagnostics written to " << json_path << std::endl;
}
}  // namespace gipc
