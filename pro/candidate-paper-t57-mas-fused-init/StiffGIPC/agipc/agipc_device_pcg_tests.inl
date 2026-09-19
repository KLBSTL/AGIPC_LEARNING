// Included within namespace agipc, after the Galerkin implementation namespace.
__global__ void device_pcg_unit_spmv(const double* diagonal,const double* p,double* ap,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)ap[i]=diagonal[i]*p[i];
}
__global__ void device_pcg_unit_identity(const double* r,double* z,int count) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)z[i]=r[i];
}
gipc::Json device_pcg_self_test() {
    using namespace device_pcg;
    Workspace workspace;
    cudatool::CudaDeviceBuffer<double> diagonal,x,r,z,p,ap;
    gipc::Json cases=gipc::Json::array();
    // Deliberately reuse allocations across block-reduction tails and a shrink.
    for(int batch:{1,4,8,16}) for(int kind=0;kind<10;++kind) {
        const int count=kind==0?769:kind==1?257:kind==2?3:kind==9?769:1;
        std::vector<double> hd(count),rhs(count);
        for(int i=0;i<count;++i) {
            hd[i]=1.0+(i%29)*0.07;rhs[i]=0.2+std::sin(0.21*i);
            if(kind==9)hd[i]=std::pow(10.0,8.0*i/(count-1));
        }
        std::string name=kind==0?"spd_769":kind==1?"spd_257":kind==2?"shrink_3":kind==3?"batch_tail_convergence":
            kind==4?"zero_rhs":kind==5?"nonpositive_curvature":kind==6?"nonfinite_curvature":
            kind==7?"nonpositive_rz":kind==8?"cap_one":"cap_512";
        if(kind==4)rhs[0]=0;
        if(kind==5)hd[0]=-2;
        if(kind==6)hd[0]=std::numeric_limits<double>::quiet_NaN();
        double initial_rr=0;for(double v:rhs)initial_rr+=v*v;
        auto control=initial(initial_rr,kind==7?-initial_rr:initial_rr,kind==8?1:512,true);
        if(kind==9)control.tolerance2=0; // Unit-only forced cap, production remains 1e-6*rr.
        if(kind==8) { // Two distinct eigenvalues cannot converge in one update.
            hd={2,4,7};rhs={1,2,3};initial_rr=14;
            control=initial(initial_rr,initial_rr,1,true);
        }
        const int n=static_cast<int>(rhs.size()),grid=(n+threads-1)/threads;
        diagonal.copy_from_host(hd);r.copy_from_host(rhs);z.copy_from_host(rhs);p.copy_from_host(rhs);
        x.resize(n);x.reset_zero();ap.resize(n);workspace.resize(n);
        workspace.control.copy_from_host(std::vector<Control>{control});
        int queued=0,downloads=0;
        auto enqueue=[&]() {
            device_pcg_unit_spmv<<<grid,threads>>>(diagonal.data(),p.data(),ap.data(),n);
            reduce(workspace,p.data(),ap.data(),n,false);
            device_pcg::update_x_r<<<grid,threads>>>(x.data(),r.data(),p.data(),ap.data(),n,workspace.control.data());
            device_pcg_unit_identity<<<grid,threads>>>(r.data(),z.data(),n);
            reduce(workspace,r.data(),z.data(),n,true);
            device_pcg::update_p<<<grid,threads>>>(p.data(),z.data(),n,workspace.control.data());
        };
        while(control.status==Running) {
            const int slots=std::min(batch,control.max_iterations-control.iterations);
            if(slots<=0)throw std::runtime_error("Device PCG fixture did not stop at cap");
            for(int k=0;k<slots;++k){enqueue();++queued;}
            CUDA_SAFE_CALL(cudaMemcpy(&control,workspace.control.data(),sizeof(control),cudaMemcpyDeviceToHost));++downloads;
        }
        std::vector<double> actual,actual_r,actual_p;
        x.copy_to_host(actual);r.copy_to_host(actual_r);p.copy_to_host(actual_p);
        const auto stopped=control;
        for(int k=0;k<4;++k)enqueue();
        std::vector<double> after_x,after_r,after_p;
        x.copy_to_host(after_x);r.copy_to_host(after_r);p.copy_to_host(after_p);
        CUDA_SAFE_CALL(cudaMemcpy(&control,workspace.control.data(),sizeof(control),cudaMemcpyDeviceToHost));
        const bool frozen=actual==after_x && actual_r==after_r && actual_p==after_p
            && control.iterations==stopped.iterations && control.status==stopped.status && control.rr==stopped.rr;
        int expected=Converged;
        if(kind==5)expected=NonpositiveCurvature;
        if(kind==6)expected=NonfiniteCurvature;
        if(kind==7)expected=InvalidRz;
        if(kind==8 || kind==9)expected=IterationCap;
        double residual2=0,error2=0,reference2=0;
        if(kind<=4)for(int i=0;i<n;++i) {
            const double residual=rhs[i]-hd[i]*actual[i],reference=rhs[i]/hd[i];
            residual2+=residual*residual;error2+=(actual[i]-reference)*(actual[i]-reference);reference2+=reference*reference;
        }
        const double relative_error=std::sqrt(error2/std::max(reference2,1e-300));
        bool passed=frozen && control.status==expected && control.iterations<=control.max_iterations
            && workspace.a.size()==static_cast<std::size_t>(grid) && workspace.bytes()>0;
        if(kind<=4)passed&=residual2<=initial_rr*1e-6*(1+2e-6) && relative_error<2e-3;
        if(kind>=5 && kind<=7)passed&=control.iterations==0 && actual==std::vector<double>(n,0) && actual_r==rhs;
        if(kind==8 || kind==9)passed&=control.iterations==control.max_iterations;
        if(kind==3 && batch>1)passed&=queued==batch && control.iterations==1;
        cases.push_back({{"name",name},{"batch",batch},{"count",n},{"status",control.status},
            {"effective_iterations",control.iterations},{"queued_iterations",queued},{"downloads",downloads},
            {"true_residual2",residual2},{"cpu_reference_relative_error",relative_error},
            {"stopped_vectors_unchanged",frozen},{"workspace_bytes",workspace.bytes()},{"passed",passed}});
        if(!passed)throw std::runtime_error("Device PCG fixture failed: "+cases.back().dump());
    }
    // Exercise residual and quotient failure gates directly, then verify all
    // queued vector updates are inert. These states are hard to produce with SPD.
    for(int kind=0;kind<3;++kind) {
        auto control=initial(1,1,512,true);
        workspace.resize(1);workspace.control.copy_from_host(std::vector<Control>{control});
        workspace.a.copy_from_host(std::vector<double>{kind==0?std::numeric_limits<double>::quiet_NaN():kind==1?1e-310:1.0});
        workspace.b.copy_from_host(std::vector<double>{kind==2?-1.0:1.0});
        if(kind==1) {control.rz=1e300;workspace.control.copy_from_host(std::vector<Control>{control});}
        finish_dots<<<1,threads>>>(workspace.a.data(),workspace.b.data(),1,kind!=1,workspace.control.data());
        x.copy_from_host(std::vector<double>{3});r.copy_from_host(std::vector<double>{4});p.copy_from_host(std::vector<double>{5});
        ap.copy_from_host(std::vector<double>{6});z.copy_from_host(std::vector<double>{7});
        device_pcg::update_x_r<<<1,threads>>>(x.data(),r.data(),p.data(),ap.data(),1,workspace.control.data());
        device_pcg::update_p<<<1,threads>>>(p.data(),z.data(),1,workspace.control.data());
        std::vector<double> hx,hr,hp;x.copy_to_host(hx);r.copy_to_host(hr);p.copy_to_host(hp);
        CUDA_SAFE_CALL(cudaMemcpy(&control,workspace.control.data(),sizeof(control),cudaMemcpyDeviceToHost));
        const int expected=kind==0?NonfiniteResidual:kind==1?NonfiniteStep:InvalidRz;
        const bool passed=control.status==expected && hx[0]==3 && hr[0]==4 && hp[0]==5;
        cases.push_back({{"name",kind==0?"nonfinite_residual":kind==1?"nonfinite_alpha":"invalid_next_rz"},{"passed",passed}});
        if(!passed)throw std::runtime_error("Device PCG scalar failure gate failed");
    }
    return {{"test","device_coarse_pcg"},{"passed",true},{"cases",cases}};
}
