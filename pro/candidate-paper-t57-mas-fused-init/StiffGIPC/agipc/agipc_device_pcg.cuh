#pragma once
#include <cuda_tools/cuda_device_buffer.h>
#include <gipc/utils/json.h>
#include <cmath>
#include <string>

namespace agipc {
void configure_coarse_pcg_batch(int batch); // 0 = original host control; positive = bounded device control
gipc::Json device_pcg_self_test();
namespace device_pcg {
constexpr int threads=256;
enum Status { Running, Converged, NonfiniteCurvature, NonpositiveCurvature,
              NonfiniteResidual, InvalidRz, IterationCap, NonfiniteStep };
struct Control {
    double rr=0,rz=0,tolerance2=0,alpha=0,beta=0;
    int iterations=0,max_iterations=0,status=Running,positive_rz=0;
};
inline Control initial(double rr,double rz,int limit,bool positive) {
    Control c;c.rr=rr;c.rz=rz;c.tolerance2=1e-6*rr;c.max_iterations=limit;c.positive_rz=positive;
    if(!std::isfinite(rr) || rr<0)c.status=NonfiniteResidual;
    else if(rr<=c.tolerance2)c.status=Converged;
    return c;
}
inline std::string failure(int status) {
    switch(status) {
        case NonfiniteCurvature:return "nonfinite_curvature";
        case NonpositiveCurvature:return "nonpositive_curvature";
        case NonfiniteResidual:return "nonfinite_residual";
        case InvalidRz:return "invalid_preconditioned_residual";
        case IterationCap:return "iteration_cap";
        case NonfiniteStep:return "nonfinite_pcg_step";
        default:return {};
    }
}
struct Workspace {
    cudatool::CudaDeviceBuffer<double> a,b;
    cudatool::CudaDeviceBuffer<Control> control;
    void resize(int count) { a.resize((count+threads-1)/threads);b.resize(a.size());control.resize(1); }
    std::size_t bytes()const {return (a.capacity()+b.capacity())*sizeof(double)+control.capacity()*sizeof(Control);}
};

// Two passes and a fixed tree avoid per-iteration Thrust temporary allocations
// and scalar downloads. Logical count excludes MAS identity padding.
__global__ void dot_partials(const double* x,const double* y,double* a,double* b,
                            int count,bool pair,const Control* control) {
    if(control->status!=Running)return;
    __shared__ double sa[threads],sb[threads];
    const int t=threadIdx.x,i=blockIdx.x*threads+t;
    const double xi=i<count?x[i]:0,yi=i<count?y[i]:0;
    sa[t]=pair?xi*xi:xi*yi;sb[t]=pair?xi*yi:0;
    __syncthreads();
    for(int offset=threads/2;offset;offset/=2) {
        if(t<offset){sa[t]+=sa[t+offset];sb[t]+=sb[t+offset];}__syncthreads();
    }
    if(t==0){a[blockIdx.x]=sa[0];if(pair)b[blockIdx.x]=sb[0];}
}
__global__ void finish_dots(const double* a,const double* b,int count,bool pair,Control* control) {
    if(control->status!=Running)return;
    __shared__ double sa[threads],sb[threads];
    const int t=threadIdx.x;double va=0,vb=0;
    for(int i=t;i<count;i+=threads){va+=a[i];if(pair)vb+=b[i];}
    sa[t]=va;sb[t]=vb;__syncthreads();
    for(int offset=threads/2;offset;offset/=2) {
        if(t<offset){sa[t]+=sa[t+offset];sb[t]+=sb[t+offset];}__syncthreads();
    }
    if(t)return;
    auto& c=*control;
    if(!pair) {
        if(!isfinite(c.rz) || (c.positive_rz && c.rz<=0)){c.status=InvalidRz;return;}
        if(!isfinite(sa[0])){c.status=NonfiniteCurvature;return;}
        if(sa[0]<=0){c.status=NonpositiveCurvature;return;}
        c.alpha=c.rz/sa[0];
        if(!isfinite(c.alpha)){c.status=NonfiniteStep;return;}
        ++c.iterations;
    } else {
        c.rr=sa[0];
        if(!isfinite(c.rr) || c.rr<0){c.status=NonfiniteResidual;return;}
        if(c.rr<=c.tolerance2){c.status=Converged;return;}
        const double next_rz=sb[0];
        if(!isfinite(next_rz) || fabs(c.rz)<1e-30 || (c.positive_rz && next_rz<=0))
            {c.status=InvalidRz;return;}
        c.beta=next_rz/c.rz;
        if(!isfinite(c.beta)){c.status=NonfiniteStep;return;}
        c.rz=next_rz;
        if(c.iterations>=c.max_iterations)c.status=IterationCap;
    }
}
__global__ void update_x_r(double* x,double* r,const double* p,const double* ap,int count,const Control* c) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count && c->status==Running){x[i]+=c->alpha*p[i];r[i]-=c->alpha*ap[i];}
}
__global__ void update_p(double* p,const double* z,int count,const Control* c) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count && c->status==Running)p[i]=z[i]+c->beta*p[i];
}
inline void reduce(Workspace& w,const double* x,const double* y,int count,bool pair) {
    const int blocks=(count+threads-1)/threads;
    dot_partials<<<blocks,threads>>>(x,y,w.a.data(),w.b.data(),count,pair,w.control.data());
    finish_dots<<<1,threads>>>(w.a.data(),w.b.data(),blocks,pair,w.control.data());
}
} }
