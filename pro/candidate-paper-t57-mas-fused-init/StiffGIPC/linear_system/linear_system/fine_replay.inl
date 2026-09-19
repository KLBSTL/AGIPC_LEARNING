#include <linear_system/preconditioner/abd_preconditioner.h>
#include <algorithm>
#include <limits>
namespace {
template<class T> std::vector<T> fine_replay_read(const std::filesystem::path& path) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    if(!in)throw std::runtime_error("missing fine replay input: "+path.string());
    const auto bytes=in.tellg();if(bytes<0 || bytes%sizeof(T))throw std::runtime_error("invalid replay binary size");
    std::vector<T> data(static_cast<std::size_t>(bytes)/sizeof(T));in.seekg(0);
    if(bytes>0)in.read(reinterpret_cast<char*>(data.data()),bytes);
    if(!in)throw std::runtime_error("replay read failed");return data;
}
__global__ void fine_replay_residual(const double* b,const double* ax,double* r,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)r[i]=b[i]-ax[i];
}
__global__ void fine_replay_update(double* x,double* r,const double* p,const double* ap,double alpha,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){x[i]+=alpha*p[i];r[i]-=alpha*ap[i];}
}
__global__ void fine_replay_direction(double* p,const double* z,double beta,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)p[i]=z[i]+beta*p[i];
}
double fine_replay_dot(const double* a,const double* b,int n) {
    return thrust::inner_product(thrust::device_ptr<const double>(a),
        thrust::device_ptr<const double>(a)+n,thrust::device_ptr<const double>(b),0.0);
}
}
namespace gipc {
Json GlobalLinearSystem::replay_frozen_fine(const std::string& directory,const std::string& output)
{
    const std::filesystem::path dir=directory,json_path=output;
    if(output.empty())throw std::runtime_error("fine replay requires metrics-path");
    const auto outdir=json_path.has_parent_path()?json_path.parent_path():std::filesystem::current_path();
    std::filesystem::create_directories(outdir);
    const Json meta=Json::parse(std::ifstream(dir/"metadata.json"));
    const int blocks=meta.at("fine_block_nodes"),n=meta.at("fine_dofs"),nnz=meta.at("fine_unique_blocks");
    if(blocks<1 || n!=3*blocks || nnz<1)throw std::runtime_error("invalid fine replay dimensions");
    auto values=fine_replay_read<Eigen::Matrix3d>(dir/"fine_A_values.f64x9.bin");
    auto rows=fine_replay_read<int>(dir/"fine_A_rows.i32.bin"),cols=fine_replay_read<int>(dir/"fine_A_cols.i32.bin");
    auto rhs=fine_replay_read<double>(dir/"fine_rhs.f64.bin");
    const auto prolonged=fine_replay_read<double>(dir/"prolongated.f64.bin");
    const auto saved_candidate=fine_replay_read<double>(dir/"agipc_candidate.f64.bin");
    if(values.size()!=nnz || rows.size()!=nnz || cols.size()!=nnz || rhs.size()!=n
       || prolonged.size()!=n || saved_candidate.size()!=n)throw std::runtime_error("replay dimension mismatch");
    for(int i=0;i<nnz;++i)if(rows[i]<0 || rows[i]>=blocks || cols[i]<0 || cols[i]>=blocks || !values[i].allFinite())
        throw std::runtime_error("invalid fine matrix entry");
    for(double value:rhs)if(!std::isfinite(value))throw std::runtime_error("nonfinite RHS");
    SizeT offset=0;
    for(auto& subsystem:m_subsystems)subsystem->report_subsystem_info();
    for(auto& subsystem:m_inner_subsystems){subsystem->dof_offset(offset);offset+=subsystem->right_hand_side_dof();}
    if(offset!=n)throw std::runtime_error("native fine subsystem dimensions differ");
    auto& matrix=*gipc_global_triplet;
    matrix.reshape(blocks,blocks);matrix.resize_collision_hash_size(nnz);
    matrix.m_block_values.copy_from_host(values);matrix.m_block_row_indices.copy_from_host(rows);matrix.m_block_col_indices.copy_from_host(cols);
    matrix.h_unique_key_number=nnz;
    m_b.copy_from(rhs);m_x.resize(n);
    const auto inverse=fine_replay_read<Matrix12x12>(dir/"native_abd_inverse.f64x144.bin");
    bool restored=false;
    for(auto& preconditioner:m_local_preconditioners)
        if(auto* abd=dynamic_cast<ABDPreconditioner*>(preconditioner.get())){abd->replay_inverse(inverse);restored=true;}
    if(!restored)throw std::runtime_error("native ABD preconditioner missing");
    // ABD inverse was assembled before matrix conversion in the original
    // build. Restore it exactly; converted matrix segment offsets are not a
    // substitute for that pre-conversion state.
    const auto segments=meta.at("native_matrix_segments");
    matrix.h_abd_abd_contact_start_id=segments.at("abd_abd_start");
    matrix.h_abd_fem_contact_start_id=segments.at("abd_fem_start");
    matrix.h_fem_abd_contact_start_id=segments.at("fem_abd_start");
    matrix.h_fem_fem_contact_start_id=segments.at("fem_fem_start");
    matrix.abd_abd_contact_num=segments.at("abd_abd_count");matrix.abd_fem_contact_num=segments.at("abd_fem_count");
    matrix.fem_abd_contact_num=segments.at("fem_abd_count");matrix.fem_fem_contact_num=segments.at("fem_fem_count");
    auto* pcg=dynamic_cast<PCGSolver*>(m_solver.get());if(!pcg)throw std::runtime_error("native PCG missing");
    cudatool::DeviceDenseVector<double> r(n),z(n),p(n),ap(n),ax(n);
    const double rhs2=fine_replay_dot(m_b.data(),m_b.data(),n);
    if(!std::isfinite(rhs2) || rhs2<=0)
        throw std::runtime_error("native fine replay requires nonzero finite RHS; original zero-RHS PCG is unguarded");
    auto true_metrics=[&]() {
        spmv(1.0,m_x.cview(),0.0,ax.view());
        fine_replay_residual<<<(n+255)/256,256>>>(m_b.data(),ax.data(),r.data(),n);
        const double rr=fine_replay_dot(r.data(),r.data(),n);
        const double pg=fine_replay_dot(m_x.data(),m_b.data(),n),php=fine_replay_dot(m_x.data(),ax.data(),n);
        std::vector<double> host;m_x.copy_to(host);
        const bool finite=std::all_of(host.begin(),host.end(),[](double v){return std::isfinite(v);});
        return Json{{"finite_direction",finite},{"true_relative_residual",std::sqrt(rr/std::max(rhs2,1e-300))},
            {"p_dot_rhs",pg},{"p_dot_Hp",php},{"predicted_quadratic_decrease",pg-0.5*php}};
    };
    auto save_solution=[&](const std::string& name) {
        std::vector<double> host;m_x.copy_to(host);write_binary(outdir/name,host);
        if(std::filesystem::file_size(outdir/name)!=sizeof(double)*host.size())throw std::runtime_error("solution write failed");
    };
    auto strict_pcg=[&](int cap,bool correction) {
        if(correction)m_x.copy_from(prolonged);else m_x.buffer_view().fill(0);
        spmv(1.0,m_x.cview(),0.0,ax.view());
        fine_replay_residual<<<(n+255)/256,256>>>(m_b.data(),ax.data(),r.data(),n);
        double rr=fine_replay_dot(r.data(),r.data(),n);const double initial=rr;
        int iterations=0;std::string reason="residual_tolerance";
        if(rr>1e-6*rhs2) {
            apply_preconditioner(z.view(),r.cview());p.buffer_view().copy_from(z.buffer_view());
            double rz=fine_replay_dot(r.data(),z.data(),n);
            reason="iteration_cap";
            while(iterations<cap && rr>1e-6*rhs2) {
                spmv(1.0,p.cview(),0.0,ap.view());const double pap=fine_replay_dot(p.data(),ap.data(),n);
                if(!std::isfinite(rz)||rz<=0||!std::isfinite(pap)||pap<=0){reason="invalid_rz_or_curvature";break;}
                const double alpha=rz/pap;if(!std::isfinite(alpha)){reason="nonfinite_alpha";break;}
                fine_replay_update<<<(n+255)/256,256>>>(m_x.data(),r.data(),p.data(),ap.data(),alpha,n);++iterations;
                rr=fine_replay_dot(r.data(),r.data(),n);
                if(!std::isfinite(rr)){reason="nonfinite_residual";break;}
                if(rr<=1e-6*rhs2){reason="residual_tolerance";break;}
                apply_preconditioner(z.view(),r.cview());const double next=fine_replay_dot(r.data(),z.data(),n);
                const double beta=next/rz;
                if(!std::isfinite(next)||next<=0||!std::isfinite(beta)){reason="invalid_next_rz_or_beta";break;}
                fine_replay_direction<<<(n+255)/256,256>>>(p.data(),z.data(),beta,n);rz=next;
            }
        }
        return Json{{"iterations",iterations},{"max_iterations",cap},{"stop_reason",reason},
            {"initial_relative_residual",std::sqrt(initial/std::max(rhs2,1e-300))},
            {"recursive_relative_residual",std::sqrt(rr/std::max(rhs2,1e-300))},
            {"criterion","recursive Euclidean squared <=1e-6*rhs_squared; independent true residual after solve"}};
    };
    auto timed=[&](auto operation,Json& entry) {
        CUDA_SAFE_CALL(cudaDeviceSynchronize());const auto begin=std::chrono::steady_clock::now();
        operation();CUDA_SAFE_CALL(cudaDeviceSynchronize());
        entry["solve_wall_ms"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    };
    Json repeats=Json::array();bool passed=true;
    for(int repeat=0;repeat<3;++repeat) {
        CUDA_SAFE_CALL(cudaDeviceSynchronize());const auto start=std::chrono::steady_clock::now();
        for(auto& preconditioner:m_local_preconditioners)if(preconditioner->preconditioner_id!=0)preconditioner->assemble();
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        const double setup=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        Json native;
        timed([&](){native["iterations"]=m_solver->solve(m_x,m_b);},native);
        native["metrics"]=true_metrics();native["criterion"]="original abs(previous r_dot_z)<=native_tol*initial_r_dot_z; original check timing retained";
        native["native_tol"]=pcg->config().global_tol_rate;
        const auto native_cap=static_cast<SizeT>(pcg->config().max_iter_ratio*n);
        native["max_iterations"]=native_cap;
        native["iteration_cap_reached"]=native.at("iterations").get<SizeT>()>=native_cap;
        native["passed"]=native["metrics"].value("finite_direction",false) && !native["iteration_cap_reached"].get<bool>();
        passed&=native["passed"].get<bool>();
        const std::string prefix="r"+std::to_string(repeat)+"_";
        save_solution(prefix+"native.f64.bin");
        Json strict;timed([&](){strict=strict_pcg(4096,false);},strict);
        strict["metrics"]=true_metrics();save_solution(prefix+"strict.f64.bin");
        const bool valid=strict["metrics"].value("finite_direction",false)
            && strict["metrics"].value("true_relative_residual",1.0)<=1e-3*(1+2e-6)
            && strict.value("stop_reason",std::string{})=="residual_tolerance";
        strict["passed"]=valid;passed&=valid;
        Json correction;timed([&](){correction=strict_pcg(10,true);},correction);
        correction["raw_metrics"]=true_metrics();save_solution(prefix+"post10_cemas_raw.f64.bin");
        const double initial=correction.at("initial_relative_residual"),final=correction["raw_metrics"].value("true_relative_residual",std::numeric_limits<double>::infinity());
        const bool guard=!correction["raw_metrics"].value("finite_direction",false)||!std::isfinite(final)||final>initial*(1+1e-6);
        if(guard)m_x.copy_from(prolonged);
        correction["residual_guard_restored_prolongated"]=guard;
        correction["selected_metrics"]=true_metrics();save_solution(prefix+"post10_cemas_selected.f64.bin");
        const auto stop=correction.value("stop_reason",std::string{});
        correction["would_pass_original_direction_gate"]=(stop=="residual_tolerance" || stop=="iteration_cap")
            && correction["selected_metrics"].value("finite_direction",false)
            && correction["selected_metrics"].value("p_dot_rhs",0.0)>0
            && correction["selected_metrics"].value("true_relative_residual",std::numeric_limits<double>::infinity())<=initial*(1+1e-6);
        m_fine_preconditioner_ready=true;
        m_x.copy_from(saved_candidate);
        cudatool::DeviceDenseVector<double> guard_start(n);guard_start.copy_from(prolonged);
        const double prolonged_norm=meta.at("post_correction").at("initial_residual_norm");
        const double prolonged_rr=prolonged_norm*prolonged_norm;
        const bool warm_requested=agipc::guarded_fine_warm_start_enabled();
        agipc::configure_guarded_fine_warm_start(false);
        const auto reference_guard=guarded_fine_correction(guard_start.data(),prolonged_rr);
        const auto reference_metrics=true_metrics();save_solution(prefix+"guarded_prolongated_reference.f64.bin");
        agipc::configure_guarded_fine_warm_start(warm_requested);
        m_x.copy_from(saved_candidate);
        const auto guarded=guarded_fine_correction(guard_start.data(),prolonged_rr);
        const auto guarded_metrics=true_metrics();save_solution(prefix+"guarded_production_selected.f64.bin");
        passed&=guarded.value("old_direction_preserved_on_rejection",false);
        repeats.push_back({{"guarded_prolongated_reference",reference_guard},{"guarded_prolongated_reference_metrics",reference_metrics},{"guarded_production",guarded},{"guarded_production_metrics",guarded_metrics},{"repeat",repeat},{"fine_preconditioner_setup_wall_ms",setup},
            {"native_fine",native},{"strict_fine_diagnostic",strict},{"post10_cemas_diagnostic",correction}});
    }
    const Json report={{"test","native_fine_cemas16_frozen_replay_v1"},{"frame",meta.at("frame")},
        {"frame_newton_ordinal",meta.at("frame_newton_ordinal")},{"sample_kind",meta.at("sample_kind")},
        {"fine_dofs",n},{"preconditioner","original FEM CEMAS16; exact captured ABD inverse and real collision pairs"},
        {"passed",passed},{"runs",repeats},{"native_production_dispatch_changed",false},{"guarded_production_helper_tested",true},
        {"post10_candidate_applied_to_motion",false},{"performance_claim",false},
        {"timing_scope","setup separate; solve synchronized wall; input load and true metrics/serialization excluded; no fine assembly included"}};
    std::ofstream out(json_path);out<<report.dump(2)<<'\n';out.close();
    if(!out)throw std::runtime_error("fine replay report write failed");return report;
}
}
