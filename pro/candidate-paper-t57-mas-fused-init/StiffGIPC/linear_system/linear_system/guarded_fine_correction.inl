// Additional bounded CEMAS trial; m_x is untouched until every gate passes.
// Kept separate from native PCG so its original convergence remains intact.
namespace gipc {
Json GlobalLinearSystem::guarded_fine_correction(const double* prolonged,double prolonged_residual_squared)
{
    const auto begin=std::chrono::steady_clock::now();
    const int n=static_cast<int>(m_x.size());
    for(auto* v:{&m_guard_old,&m_guard_trial,&m_guard_r,&m_guard_z,&m_guard_p,&m_guard_ap,&m_guard_ax})v->resize(n);
    auto& old=m_guard_old;auto& x=m_guard_trial;auto& r=m_guard_r;
    auto& z=m_guard_z;auto& p=m_guard_p;auto& ap=m_guard_ap;auto& ax=m_guard_ax;
    const bool warm=agipc::guarded_fine_warm_start_enabled();
    int true_spmv_count=0;
    old.buffer_view().copy_from(m_x.buffer_view());
    CUDA_SAFE_CALL(cudaMemcpy(x.data(),warm?old.data():prolonged,n*sizeof(double),cudaMemcpyDeviceToDevice));
    auto metrics=[&](const cudatool::DeviceDenseVector<double>& v){
        ++true_spmv_count;
        spmv(1.0,v.cview(),0.0,ax.view());
        fine_replay_residual<<<(n+255)/256,256>>>(m_b.data(),ax.data(),r.data(),n);
        const double rr=fine_replay_dot(r.data(),r.data(),n);
        const double norm2=fine_replay_dot(v.data(),v.data(),n);
        const double pg=fine_replay_dot(v.data(),m_b.data(),n),php=fine_replay_dot(v.data(),ax.data(),n);
        return Json{{"squared_residual",rr},{"squared_direction_norm",norm2},
            {"p_dot_rhs",pg},{"p_dot_Hp",php},{"predicted_quadratic_decrease",pg-.5*php},
            {"finite_direction",std::isfinite(norm2)&&std::isfinite(rr)&&std::isfinite(pg)&&std::isfinite(php)}};
    };
    const auto old_metrics=metrics(old);
    const double rhs2=fine_replay_dot(m_b.data(),m_b.data(),n);
    // metrics(old) already left the exact old residual in r.
    const auto initial_metrics=warm?old_metrics:metrics(x);
    const double guard_reference=warm?prolonged_residual_squared:initial_metrics.at("squared_residual").get<double>();
    const double initial=initial_metrics.at("squared_residual");
    double rr=initial;int iterations=0;std::string stop="residual_tolerance";
    double setup_ms=0;
    if(!std::isfinite(rhs2)||rhs2<=0||!std::isfinite(guard_reference)||guard_reference<0||!initial_metrics.value("finite_direction",false))stop="invalid_initial_state";
    else if(rr>1e-6*rhs2){
        if(!m_fine_preconditioner_ready){
            const auto start=std::chrono::steady_clock::now();
            if(m_global_preconditioner)m_global_preconditioner->do_assemble(*gipc_global_triplet);
            for(auto& preconditioner:m_local_preconditioners)if(preconditioner->preconditioner_id!=0)preconditioner->assemble();
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            setup_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            m_fine_preconditioner_ready=true;
        }
        apply_preconditioner(z.view(),r.cview());p.buffer_view().copy_from(z.buffer_view());
        double rz=fine_replay_dot(r.data(),z.data(),n);stop="iteration_cap";
        while(iterations<10 && rr>1e-6*rhs2){
            spmv(1.0,p.cview(),0.0,ap.view());const double pap=fine_replay_dot(p.data(),ap.data(),n);
            if(!std::isfinite(rz)||rz<=0||!std::isfinite(pap)||pap<=0){stop="invalid_rz_or_curvature";break;}
            const double alpha=rz/pap;if(!std::isfinite(alpha)){stop="nonfinite_alpha";break;}
            fine_replay_update<<<(n+255)/256,256>>>(x.data(),r.data(),p.data(),ap.data(),alpha,n);++iterations;
            rr=fine_replay_dot(r.data(),r.data(),n);
            if(!std::isfinite(rr)){stop="nonfinite_residual";break;}
            if(rr<=1e-6*rhs2){stop="residual_tolerance";break;}
            apply_preconditioner(z.view(),r.cview());const double next=fine_replay_dot(r.data(),z.data(),n);
            const double beta=next/rz;
            if(!std::isfinite(next)||next<=0||!std::isfinite(beta)){stop="invalid_next_rz_or_beta";break;}
            fine_replay_direction<<<(n+255)/256,256>>>(p.data(),z.data(),beta,n);rz=next;
        }
    }
    const auto raw=warm&&iterations==0?old_metrics:metrics(x);
    const double raw_rr=raw.value("squared_residual",std::numeric_limits<double>::infinity());
    const bool residual_guard=!raw.value("finite_direction",false)||raw_rr>guard_reference*(1+1e-6)*(1+1e-6);
    if(residual_guard)CUDA_SAFE_CALL(cudaMemcpy(x.data(),warm?old.data():prolonged,n*sizeof(double),cudaMemcpyDeviceToDevice));
    const auto trial=residual_guard?(warm?old_metrics:metrics(x)):raw;
    const double trial_rr=trial.value("squared_residual",std::numeric_limits<double>::infinity());
    const double qold=old_metrics.value("predicted_quadratic_decrease",std::numeric_limits<double>::infinity());
    const double qtrial=trial.value("predicted_quadratic_decrease",-std::numeric_limits<double>::infinity());
    // Reject model ties and improvements within reduction roundoff.
    const double margin=1e-12*std::max({std::abs(qold),std::abs(qtrial),1e-300});
    const bool direction_gate=(stop=="iteration_cap"||stop=="residual_tolerance")
        && trial.value("finite_direction",false)&&trial.value("p_dot_rhs",0.0)>0
        && std::isfinite(trial_rr)&&trial_rr<=guard_reference*(1+1e-6)*(1+1e-6);
    const bool model_gate=old_metrics.value("finite_direction",false)&&std::isfinite(qold)
        &&std::isfinite(qtrial)&&qtrial>qold+margin;
    const bool selected=direction_gate&&model_gate;
    if(selected)m_x.buffer_view().copy_from(x.buffer_view());
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    return Json{{"attempted",true},{"selected",selected},{"iterations",iterations},{"max_extra_iterations",10},
        {"stop_reason",stop},{"direction_gate_passed",direction_gate},{"model_gate_passed",model_gate},
        {"selection_reason",selected?"better_quadratic_model":(!direction_gate?"direction_gate_rejected":"model_gate_rejected")},
        {"start_source",warm?"accepted_post10":"prolongated"},{"initial_true_spmv_reused",warm},
        {"true_metrics_spmv_count",true_spmv_count},{"prolongated_residual_squared_reference",guard_reference},
        {"residual_guard_restored_accepted",warm&&residual_guard},
        {"residual_guard_restored_prolongated",!warm&&residual_guard},{"old_direction_preserved_on_rejection",true},
        {"old_metrics",old_metrics},{"initial_metrics",initial_metrics},{"raw_metrics",raw},{"trial_metrics",trial},
        {"model_margin",margin},{"rhs_squared",rhs2},{"preconditioner_setup_wall_ms",setup_ms},
        {"total_wall_ms",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count()},
        {"timing_scope","all additional workspace, old/trial true SpMV, fine preconditioner setup, bounded PCG and model selection"}};
}
}
