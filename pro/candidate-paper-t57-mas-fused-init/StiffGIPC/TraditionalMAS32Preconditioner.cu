//
// TraditionalMAS32Preconditioner.cu
// Reuses the c499 MAS CUDA skeleton with the GPU_IPC 32-node bank size.
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#include "TraditionalMAS32Preconditioner.cuh"
#include "cuda_tools/cuda_tools.h"
#include "device_launch_parameters.h"
#include <math_constants.h>
#include <cuda_tools/cuda_all.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <vector>
#include <bitset>
#include <fstream>
#include <iostream>
#include <limits>
#include <cublas_v2.h>
#include <agipc/agipc_mas_check_backend.cuh>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "cooperative_groups.h"
using namespace cooperative_groups;
//#include "Eigen/Eigen"
using namespace std;
#define SYME
#define GROUP

namespace gpu_mas32
{

namespace
{
constexpr int kCheckDimension=3*GPU_MAS_BANKSIZE,kCheckThreads=128;
bool gemm_local_checks_enabled=false;
bool tiled_local_checks_enabled=false;
constexpr int kLocalCheckBatch=256;
struct LocalGpuCheck
{
    int local_finite,inverse_finite,local_spd,inverse_spd;
    double minimum_local_diagonal,minimum_inverse_diagonal,inverse_residual;
    int reference_rechecked;
};

template<typename Matrix>
__device__ double packed_scalar(const Matrix& matrix,int row,int col)
{
    int br=row/3,bc=col/3,rr=row%3,cc=col%3;
    if(br>bc) { const int tmp=br;br=bc;bc=tmp;const int tr=rr;rr=cc;cc=tr; }
    const int index=GPU_MAS_BANKSIZE*br-br*(br+1)/2+bc;
    return static_cast<double>(matrix.M[index](rr,cc));
}

__device__ void check_cholesky(double* work,int* positive)
{
    constexpr int n=kCheckDimension;
    for(int k=0;k<n;++k)
    {
        if(threadIdx.x==0)
        {
            const double pivot=work[k*n+k];
            if(!isfinite(pivot) || pivot<=0) *positive=0;
            else work[k*n+k]=sqrt(pivot);
        }
        __syncthreads();
        if(!*positive) break;
        const int row=threadIdx.x;
        if(row>k && row<n) work[row*n+k]/=work[k*n+k];
        __syncthreads();
        if(row>k && row<n)
            for(int col=k+1;col<=row;++col)
                work[row*n+col]-=work[row*n+k]*work[col*n+k];
        __syncthreads();
    }
}

struct GemmCheckWorkspace {
    cublasHandle_t handle=nullptr;
    double* storage=nullptr;
    int capacity=0;
    std::size_t allocated_bytes=0;
    ~GemmCheckWorkspace() { if(handle) cublasDestroy(handle); if(storage) cudaFree(storage); }
    bool ensure(int required) {
        if(!handle && cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS) return false;
        if(cublasSetStream(handle,cudaStreamPerThread)!=CUBLAS_STATUS_SUCCESS
           || cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH)!=CUBLAS_STATUS_SUCCESS) return false;
        if(capacity>=required) return true;
        const auto bytes=std::size_t(required)*(3*kCheckDimension*kCheckDimension+1)*sizeof(double);
        double* replacement=nullptr;
        if(cudaMalloc(reinterpret_cast<void**>(&replacement),bytes)!=cudaSuccess) { cudaGetLastError(); return false; }
        if(storage) CUDA_SAFE_CALL(cudaFree(storage));
        storage=replacement; capacity=required; allocated_bytes=bytes;
        return true;
    }
    double* a() { return storage; }
    double* b() { return storage+std::size_t(capacity)*kCheckDimension*kCheckDimension; }
    double* product() { return storage+2*std::size_t(capacity)*kCheckDimension*kCheckDimension; }
    double* bounds() { return storage+3*std::size_t(capacity)*kCheckDimension*kCheckDimension; }
};

__global__ void materialize_local_checks(const __GEIGEN__::GPUMas32MatrixSymT* local,
                                         const __GEIGEN__::GPUMas32MatrixSymf* inverse,
                                         double* a,double* b,double* bounds) {
    constexpr int n=kCheckDimension;
    __shared__ double norm_a[kCheckThreads],norm_b[kCheckThreads];
    const int id=blockIdx.x,tid=threadIdx.x;
    double aa=0,bb=0;
    for(int i=tid;i<n*n;i+=blockDim.x) {
        const int row=i%n,col=i/n;
        double av=packed_scalar(local[id],row,col);
        if(row==col && av==0) av=1;
        const double bv=packed_scalar(inverse[id],row,col);
        a[id*n*n+i]=av; b[id*n*n+i]=bv;
        aa+=av*av; bb+=bv*bv;
    }
    norm_a[tid]=aa; norm_b[tid]=bb; __syncthreads();
    if(tid==0) {
        double a2=0,b2=0;
        for(int i=0;i<kCheckThreads;++i) { a2+=norm_a[i]; b2+=norm_b[i]; }
        // Conservative FP64 dot/norm rounding band. Ambiguous or nonfinite
        // bands always use the original packed product, preserving its gate.
        bounds[id]=64.0*2.2204460492503131e-16*n*sqrt(a2)*sqrt(b2)/sqrt(double(n))+1e-11;
    }
}

// Preserve the packed reference's FP64 k=0..95 accumulation. Only the
// reads are tiled: no tensor cores, alternate precision or reduction tree.
__global__ void tiled_local_products(
    const __GEIGEN__::GPUMas32MatrixSymT* local,
    const __GEIGEN__::GPUMas32MatrixSymf* inverse,
    double* products,double* bounds)
{
    constexpr int n=kCheckDimension,tile=16;
    __shared__ double a[tile][tile],b[tile][tile];
    const int x=threadIdx.x,y=threadIdx.y;
    const int row=blockIdx.y*tile+y,col=blockIdx.x*tile+x,id=blockIdx.z;
    double product=0;
    for(int base=0;base<n;base+=tile)
    {
        double av=packed_scalar(local[id],row,base+x);
        if(row==base+x && av==0) av=1;
        a[y][x]=av;
        b[y][x]=packed_scalar(inverse[id],base+y,col);
        __syncthreads();
        #pragma unroll 1
        for(int k=0;k<tile;++k) product+=a[y][k]*b[k][x];
        __syncthreads();
    }
    products[id*n*n+col*n+row]=product;
    // Keep the original packed fallback for values close to the gate.
    if(blockIdx.x==0 && blockIdx.y==0 && x==0 && y==0) bounds[id]=1e-11;
}

struct TiledCheckWorkspace
{
    cudatool::CudaDeviceBuffer<double> products,bounds;
    void resize(int count)
    {
        products.resize(count*kCheckDimension*kCheckDimension);
        bounds.resize(count);
    }
};

__global__ void check_local_matrices(const __GEIGEN__::GPUMas32MatrixSymT* local,
                                     const __GEIGEN__::GPUMas32MatrixSymf* inverse,
                                     LocalGpuCheck* results,const double* products=nullptr,
                                     const double* bounds=nullptr)
{
    constexpr int n=kCheckDimension;
    extern __shared__ double work[];
    __shared__ int finite[2],positive[2];
    __shared__ double inverse_residual;
    __shared__ int recheck;
    __shared__ double partial[kCheckThreads],diag_local[kCheckThreads],diag_inverse[kCheckThreads];
    const int tid=threadIdx.x,id=blockIdx.x;
    if(tid<2) finite[tid]=1;
    diag_local[tid]=diag_inverse[tid]=CUDART_INF;
    __syncthreads();
    for(int i=tid;i<n*n;i+=blockDim.x)
    {
        const int row=i/n,col=i%n;
        double a=packed_scalar(local[id],row,col);
        const double b=packed_scalar(inverse[id],row,col);
        if(row==col && a==0) a=1; // Matches the complete CPU diagnostic.
        if(!isfinite(a)) atomicExch(&finite[0],0);
        if(!isfinite(b)) atomicExch(&finite[1],0);
        work[i]=a;
        if(row==col) { diag_local[tid]=fmin(diag_local[tid],a);diag_inverse[tid]=fmin(diag_inverse[tid],b); }
    }
    __syncthreads();
    if(tid<2) positive[tid]=finite[tid];
    __syncthreads();
    if(finite[0]) check_cholesky(work,&positive[0]);
    for(int i=tid;i<n*n;i+=blockDim.x)
    {
        double a=packed_scalar(local[id],i/n,i%n);
        if(i/n==i%n && a==0) a=1;
        work[i]=a;
    }
    __syncthreads();
    double sum=0;
    if(finite[0] && finite[1])
        for(int i=tid;i<n*n;i+=blockDim.x)
        {
            const int row=i/n,col=i%n;
            double product=products?products[id*n*n+col*n+row]:0;
            if(!products) for(int k=0;k<n;++k) product+=work[row*n+k]*packed_scalar(inverse[id],k,col);
            const double error=product-(row==col?1.0:0.0);
            sum+=error*error;
        }
    partial[tid]=sum;
    __syncthreads();
    if(tid==0)
    {
        double norm2=0;
        for(int i=0;i<kCheckThreads;++i) norm2+=partial[i];
        inverse_residual=sqrt(norm2/n);
        recheck=products && (!isfinite(inverse_residual) || !isfinite(bounds[id])
            || fabs(inverse_residual-1e-3)<=bounds[id]);
    }
    __syncthreads();
    if(recheck) {
        sum=0;
        if(finite[0] && finite[1]) for(int i=tid;i<n*n;i+=blockDim.x) {
            const int row=i/n,col=i%n;
            double product=0;
            for(int k=0;k<n;++k) product+=work[row*n+k]*packed_scalar(inverse[id],k,col);
            const double error=product-(row==col?1.0:0.0);
            sum+=error*error;
        }
        partial[tid]=sum; __syncthreads();
        if(tid==0) { double norm2=0; for(int i=0;i<kCheckThreads;++i) norm2+=partial[i]; inverse_residual=sqrt(norm2/n); }
        __syncthreads();
    }
    // For symmetric B, A SPD and ||I-AB||_2 < 1 imply B SPD. The accepted
    // Frobenius residual is at most 1e-3*sqrt(96), so the second Cholesky is
    // needed only for failed or borderline local checks.
    if(finite[1] && !(positive[0] && inverse_residual<=1e-3))
    {
        for(int i=tid;i<n*n;i+=blockDim.x) work[i]=packed_scalar(inverse[id],i/n,i%n);
        __syncthreads();
        check_cholesky(work,&positive[1]);
    }
    if(tid==0)
    {
        double min_a=CUDART_INF,min_b=CUDART_INF;
        for(int i=0;i<kCheckThreads;++i)
        { min_a=fmin(min_a,diag_local[i]);min_b=fmin(min_b,diag_inverse[i]); }
        results[id]={finite[0],finite[1],positive[0],positive[1],min_a,min_b,inverse_residual,recheck};
    }
}

gipc::Json gpu_local_checks(const __GEIGEN__::GPUMas32MatrixSymT* local,
                            const __GEIGEN__::GPUMas32MatrixSymf* inverse,int count,
                            bool gemm=gemm_local_checks_enabled,std::vector<LocalGpuCheck>* per_matrix=nullptr,
                            bool tiled=tiled_local_checks_enabled)
{
    if(count<1) return {{"passed",false},{"failure_reason","empty_local_matrices"}};
    constexpr int bytes=kCheckDimension*kCheckDimension*sizeof(double);
    CUDA_SAFE_CALL(cudaFuncSetAttribute(check_local_matrices,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
    cudatool::CudaDeviceBuffer<LocalGpuCheck> device_result;
    device_result.resize(count);
    int gemm_batches=0,api_fallback_batches=0;
    std::size_t workspace_bytes=0;
    int tiled_batches=0;
    if(tiled) {
        thread_local TiledCheckWorkspace workspace;
        workspace.resize(kLocalCheckBatch);
        workspace_bytes=std::size_t(kLocalCheckBatch)*(kCheckDimension*kCheckDimension+1)*sizeof(double);
        for(int offset=0;offset<count;offset+=kLocalCheckBatch) {
            const int batch=std::min(count-offset,kLocalCheckBatch);
            tiled_local_products<<<dim3(6,6,batch),dim3(16,16)>>>(
                local+offset,inverse+offset,workspace.products.data(),workspace.bounds.data());
            CUDA_SAFE_CALL(cudaGetLastError());
            check_local_matrices<<<batch,kCheckThreads,bytes>>>(local+offset,inverse+offset,
                device_result.data()+offset,workspace.products.data(),workspace.bounds.data());
            ++tiled_batches;
        }
    } else if(gemm) {
        thread_local GemmCheckWorkspace workspace;
        const bool ready=workspace.ensure(kLocalCheckBatch);
        workspace_bytes=workspace.allocated_bytes;
        const double alpha=1,beta=0;
        for(int offset=0;offset<count;offset+=kLocalCheckBatch) {
            const int batch=std::min(count-offset,kLocalCheckBatch);
            bool product_ready=false;
            if(ready) {
                materialize_local_checks<<<batch,kCheckThreads>>>(local+offset,inverse+offset,workspace.a(),workspace.b(),workspace.bounds());
                CUDA_SAFE_CALL(cudaGetLastError());
                product_ready=cublasDgemmStridedBatched(workspace.handle,CUBLAS_OP_N,CUBLAS_OP_N,
                    kCheckDimension,kCheckDimension,kCheckDimension,&alpha,
                    workspace.a(),kCheckDimension,kCheckDimension*kCheckDimension,
                    workspace.b(),kCheckDimension,kCheckDimension*kCheckDimension,&beta,
                    workspace.product(),kCheckDimension,kCheckDimension*kCheckDimension,batch)==CUBLAS_STATUS_SUCCESS;
            }
            if(product_ready) ++gemm_batches; else ++api_fallback_batches;
            check_local_matrices<<<batch,kCheckThreads,bytes>>>(local+offset,inverse+offset,device_result.data()+offset,
                product_ready?workspace.product():nullptr,product_ready?workspace.bounds():nullptr);
        }
    } else check_local_matrices<<<count,kCheckThreads,bytes>>>(local,inverse,device_result.data());
    CUDA_SAFE_CALL(cudaGetLastError());
    std::vector<LocalGpuCheck> results;
    device_result.copy_to_host(results);
    if(per_matrix) *per_matrix=results;
    int local_spd=0,inverse_spd=0,nonfinite_a=0,nonfinite_b=0,first_a=-1,first_b=-1;
    double min_a=std::numeric_limits<double>::infinity(),min_b=min_a,max_residual=0;
    int reference_rechecks=0;
    for(int i=0;i<count;++i)
    {
        const auto& r=results[i]; local_spd+=r.local_spd;inverse_spd+=r.inverse_spd;
        reference_rechecks+=r.reference_rechecked;
        nonfinite_a+=!r.local_finite;nonfinite_b+=!r.inverse_finite;
        if(!r.local_spd && first_a<0) first_a=i;
        if(!r.inverse_spd && first_b<0) first_b=i;
        min_a=std::min(min_a,r.minimum_local_diagonal);min_b=std::min(min_b,r.minimum_inverse_diagonal);
        if(!std::isfinite(r.inverse_residual)) max_residual=std::numeric_limits<double>::infinity();
        else max_residual=std::max(max_residual,r.inverse_residual);
    }
    return {{"implementation","gpu_fp64_local_checks"},{"matrix_count",count},
        {"product_backend_requested",tiled?"tiled":(gemm?"gemm":"packed")},{"gemm_batches",gemm_batches},
        {"tiled_batches",tiled_batches},{"product_accumulation",tiled?"fp64_sequential_k":"legacy"},
        {"batch_limit",kLocalCheckBatch},{"workspace_bytes",workspace_bytes},
        {"workspace_scope","explicit A/B/product/bounds; excludes cuBLAS handle and internal allocations"},
        {"api_fallback_batches",api_fallback_batches},{"packed_reference_rechecks",reference_rechecks},
        {"local_spd_count",local_spd},{"inverse_spd_count",inverse_spd},
        {"first_local_spd_failure",first_a},{"first_inverse_spd_failure",first_b},
        {"nonfinite_local_count",nonfinite_a},{"nonfinite_inverse_count",nonfinite_b},
        {"minimum_local_diagonal",min_a},{"minimum_inverse_diagonal",min_b},
        {"maximum_inverse_residual",max_residual},
        {"passed",local_spd==count && inverse_spd==count && !nonfinite_a && !nonfinite_b && max_residual<=1e-3}};
}
}

void configure_local_check_gemm(bool enabled) { gemm_local_checks_enabled=enabled; }
void configure_local_check_tiled(bool enabled) { tiled_local_checks_enabled=enabled; }

gipc::Json run_local_diagnostics_gpu_self_test()
{
    std::vector<__GEIGEN__::GPUMas32MatrixSymT> local(6);
    std::vector<__GEIGEN__::GPUMas32MatrixSymf> inverse(6);
    for(int id=0;id<6;++id)
        for(int row=0;row<GPU_MAS_BANKSIZE;++row)
            for(int col=row;col<GPU_MAS_BANKSIZE;++col)
            {
                const int index=GPU_MAS_BANKSIZE*row-row*(row+1)/2+col;
                local[id].M[index].setZero();inverse[id].M[index].setZero();
                if(row==col) {local[id].M[index].setIdentity();inverse[id].M[index].setIdentity();}
            }
    local[1].M[0](0,0)=-1;inverse[2].M[0](0,0)=-1;
    inverse[3].M[0](0,0)=std::numeric_limits<float>::quiet_NaN();
    for(int row=0;row<GPU_MAS_BANKSIZE;++row)
        inverse[4].M[GPU_MAS_BANKSIZE*row-row*(row+1)/2+row]*=2.0f;
    local[5].M[0](0,0)=std::numeric_limits<double>::quiet_NaN();
    cudatool::CudaDeviceBuffer<__GEIGEN__::GPUMas32MatrixSymT> a;
    cudatool::CudaDeviceBuffer<__GEIGEN__::GPUMas32MatrixSymf> b;
    a.copy_from_host(local);b.copy_from_host(inverse);
    gipc::Json cases=gipc::Json::array();bool passed=true;
    for(int id=0;id<6;++id)
    {
        const auto result=gpu_local_checks(a.data()+id,b.data()+id,1,false,nullptr,false);
        const bool ok=result.value("passed",false)==(id==0)
            && (id!=1 || result.value("local_spd_count",1)==0)
            && (id!=2 || result.value("inverse_spd_count",1)==0)
            && (id!=3 || result.value("nonfinite_inverse_count",0)==1)
            && (id!=4 || result.value("maximum_inverse_residual",0.0)>1e-3)
            && (id!=5 || result.value("nonfinite_local_count",0)==1);
        cases.push_back({{"case",id},{"expected_pass",id==0},{"check_passed",ok},{"diagnostic",result}});passed&=ok;
    }
    constexpr int count=258,n=kCheckDimension;
    local.resize(count); inverse.resize(count);
    for(int id=0;id<count;++id) {
        const int type=id%10;
        double denom=1;
        for(int i=0;i<n;++i) { const double u=.1*std::sin(double(i)); const double d=(type==1?1e-4:.05)*(1+i%7*.03); denom+=u*u/d; }
        for(int br=0;br<GPU_MAS_BANKSIZE;++br) for(int bc=br;bc<GPU_MAS_BANKSIZE;++bc) {
            const int index=GPU_MAS_BANKSIZE*br-br*(br+1)/2+bc;
            for(int r=0;r<3;++r) for(int c=0;c<3;++c) {
                const int row=3*br+r,col=3*bc+c;
                double av=row==col?1:0,bv=av;
                if(type<=1) {
                    const double ur=.1*std::sin(double(row)),uc=.1*std::sin(double(col));
                    const double dr=(type==1?1e-4:.05)*(1+row%7*.03),dc=(type==1?1e-4:.05)*(1+col%7*.03);
                    av=ur*uc+(row==col?dr:0);
                    bv=(row==col?1/dr:0)-ur*uc/(dr*dc*denom);
                }
                if(type==6 && row==col) av=0; // Original zero-diagonal policy.
                if((type==7 || type==8) && row==col) {
                    const float bdiag=1.001f;
                    bv=bdiag; av=(1.001+(type==7?-1e-13:1e-13))/double(bdiag);
                }
                if(type==9 && row==col) bv=2;
                local[id].M[index](r,c)=av; inverse[id].M[index](r,c)=static_cast<float>(bv);
            }
        }
        if(type==2) local[id].M[0](0,0)=-1;
        if(type==3) inverse[id].M[0](0,0)=-1;
        if(type==4) inverse[id].M[0](0,0)=std::numeric_limits<float>::quiet_NaN();
        if(type==5) local[id].M[0](0,0)=std::numeric_limits<double>::quiet_NaN();
    }
    a.copy_from_host(local); b.copy_from_host(inverse);
    std::vector<LocalGpuCheck> reference,candidate;
    const auto packed=gpu_local_checks(a.data(),b.data(),count,false,&reference,false);
    const auto gemm=gpu_local_checks(a.data(),b.data(),count,true,&candidate,false);
    int mismatches=0,near_threshold_checks=0;
    double max_norm_difference=0;
    for(int id=0;id<count;++id) {
        const auto& r=reference[id]; const auto& s=candidate[id];
        const bool same=r.local_finite==s.local_finite && r.inverse_finite==s.inverse_finite
            && r.local_spd==s.local_spd && r.inverse_spd==s.inverse_spd
            && (r.inverse_residual<=1e-3)==(s.inverse_residual<=1e-3);
        const double difference=std::abs(r.inverse_residual-s.inverse_residual);
        const bool norms_same=(!std::isfinite(r.inverse_residual) && !std::isfinite(s.inverse_residual))
            || difference<=1e-9*(1+std::abs(r.inverse_residual));
        const bool boundary_ok=(id%10!=7 && id%10!=8) || s.reference_rechecked;
        mismatches+=!(same && norms_same && boundary_ok);
        if(id%10==7 || id%10==8) near_threshold_checks+=s.reference_rechecked;
        if(std::isfinite(difference)) max_norm_difference=std::max(max_norm_difference,difference);
    }
    passed=passed && mismatches==0 && gemm.value("gemm_batches",0)==2 && gemm.value("api_fallback_batches",1)==0;
    std::vector<LocalGpuCheck> tiled_results;
    const auto tiled_report=gpu_local_checks(a.data(),b.data(),count,false,&tiled_results,true);
    int tiled_mismatches=0,tiled_finite_norm_bit_mismatches=0;
    for(int id=0;id<count;++id) {
        const auto& old=reference[id]; const auto& trial=tiled_results[id];
        const bool same=old.local_finite==trial.local_finite && old.inverse_finite==trial.inverse_finite
            && old.local_spd==trial.local_spd && old.inverse_spd==trial.inverse_spd
            && (old.inverse_residual<=1e-3)==(trial.inverse_residual<=1e-3);
        tiled_mismatches+=!same;
        if(std::isfinite(old.inverse_residual) && old.inverse_residual!=trial.inverse_residual)
            ++tiled_finite_norm_bit_mismatches;
    }
    passed=passed && tiled_mismatches==0 && tiled_finite_norm_bit_mismatches==0
        && tiled_report.value("tiled_batches",0)==2;
    return {{"passed",passed},{"cases",cases},{"tiled_reference",{
        {"matrix_count",count},{"decision_mismatches",tiled_mismatches},
        {"finite_norm_bit_mismatches",tiled_finite_norm_bit_mismatches},{"diagnostic",tiled_report}}},
        {"batched_reference",{
        {"matrix_count",count},{"decision_mismatches",mismatches},{"near_threshold_reference_checks",near_threshold_checks},
        {"maximum_norm_difference",max_norm_difference},{"packed",packed},{"gemm",gemm}}}};
}

gipc::Json TraditionalMAS32Preconditioner::local_diagnostics_gpu() const
{ return gpu_local_checks(d_inverseMatMas,d_precondMatMas,totalNumberClusters/GPU_MAS_BANKSIZE); }

bool local_diagnostics_agree(const gipc::Json& gpu,const gipc::Json& cpu)
{
    for(const char* key : {"passed","matrix_count","local_spd_count","inverse_spd_count",
        "first_local_spd_failure","first_inverse_spd_failure","nonfinite_local_count","nonfinite_inverse_count"})
        if(gpu.at(key)!=cpu.at(key)) return false;
    const auto& a=gpu.at("maximum_inverse_residual");const auto& b=cpu.at("maximum_inverse_residual");
    return a.is_number() && b.is_number() && std::abs(a.get<double>()-b.get<double>())<=1e-9;
}

void TraditionalMAS32Preconditioner::refresh_fixed_graph_bcoo(Eigen::Matrix3d* values,int* rows,int* cols,
                                                            uint32_t* indices,int offset,int count)
{
    if(totalNodes<1) return;
    CUDA_SAFE_CALL(cudaMemset(d_inverseMatMas,0,totalNumberClusters/GPU_MAS_BANKSIZE*sizeof(__GEIGEN__::GPUMas32MatrixSymT)));
    PrepareHessian_bcoo(values,rows,cols,indices,offset,count);
}

__global__ void _buildCML0_new(const unsigned int* _neighborStart,
                               unsigned int*       _neighborNum,
                               unsigned int*       _neighborList,
                               unsigned int*       _fineConnectedMsk,
                               int*                _partId_map_real,
                               int*                _real_map_partId,
                               int                 number);
__global__ void _preparePrefixSumL0_new(int*          _prefixOriginal,
                                        unsigned int* _fineConnectedMsk,
                                        int*          _partId_map_real,
                                        int           vertNum);
__global__ void _buildLevel1_new(int2*               _levelSize,
                                 int*                _coarseSpaceTable,
                                 int*                _goingNext,
                                 const unsigned int* _fineConnectedMsk,
                                 const int*          _prefixSumOriginal,
                                 const int*          _prefixOriginal,
                                 int*                _partId_map_real,
                                 int                 number);

HierarchySelfTestResult run_hierarchy_self_test()
{
    constexpr int valid_nodes  = GPU_MAS_BANKSIZE + 3;
    constexpr int padded_nodes = GPU_MAS_BANKSIZE * 2;
    constexpr int warp_count   = padded_nodes / GPU_MAS_BANKSIZE;

    std::vector<unsigned int> neighbor_start(valid_nodes);
    std::vector<unsigned int> neighbor_count(valid_nodes);
    std::vector<unsigned int> neighbor_list;
    for(int vertex = 0; vertex < valid_nodes; ++vertex)
    {
        neighbor_start[vertex] = static_cast<unsigned int>(neighbor_list.size());
        const int group_begin = vertex < GPU_MAS_BANKSIZE ? 0 : GPU_MAS_BANKSIZE;
        const int group_end = vertex < GPU_MAS_BANKSIZE ? GPU_MAS_BANKSIZE : valid_nodes;
        if(vertex > group_begin)
            neighbor_list.push_back(static_cast<unsigned int>(vertex - 1));
        if(vertex + 1 < group_end)
            neighbor_list.push_back(static_cast<unsigned int>(vertex + 1));
        neighbor_count[vertex] = static_cast<unsigned int>(neighbor_list.size())
                                 - neighbor_start[vertex];
    }

    std::vector<int> part_to_real(padded_nodes, -1);
    std::vector<int> real_to_part(valid_nodes);
    for(int vertex = 0; vertex < valid_nodes; ++vertex)
    {
        part_to_real[vertex] = vertex;
        real_to_part[vertex] = vertex;
    }

    unsigned int* d_neighbor_start = nullptr;
    unsigned int* d_neighbor_count = nullptr;
    unsigned int* d_neighbor_list  = nullptr;
    unsigned int* d_fine_mask      = nullptr;
    int*          d_part_to_real   = nullptr;
    int*          d_real_to_part   = nullptr;
    int*          d_prefix         = nullptr;
    int*          d_prefix_sum     = nullptr;
    int2*         d_level_size     = nullptr;
    int*          d_coarse         = nullptr;
    int*          d_going_next     = nullptr;

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighbor_start,
                              valid_nodes * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighbor_count,
                              valid_nodes * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighbor_list,
                              neighbor_list.size() * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_fine_mask,
                              valid_nodes * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_part_to_real, padded_nodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_real_to_part, valid_nodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_prefix, warp_count * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_prefix_sum, warp_count * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_level_size, 2 * sizeof(int2)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_coarse, valid_nodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_going_next, valid_nodes * sizeof(int)));

    CUDA_SAFE_CALL(cudaMemcpy(d_neighbor_start,
                              neighbor_start.data(),
                              valid_nodes * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_neighbor_count,
                              neighbor_count.data(),
                              valid_nodes * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_neighbor_list,
                              neighbor_list.data(),
                              neighbor_list.size() * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_part_to_real,
                              part_to_real.data(),
                              padded_nodes * sizeof(int),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_real_to_part,
                              real_to_part.data(),
                              valid_nodes * sizeof(int),
                              cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemset(d_fine_mask, 0, valid_nodes * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMemset(d_prefix, 0, warp_count * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_prefix_sum, 0, warp_count * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_level_size, 0, 2 * sizeof(int2)));
    CUDA_SAFE_CALL(cudaMemset(d_coarse, 0xff, valid_nodes * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemset(d_going_next, 0xff, valid_nodes * sizeof(int)));

    constexpr int block_size = DEFAULT_BLOCKSIZE;
    _buildCML0_new<<<1, block_size>>>(d_neighbor_start,
                                      d_neighbor_count,
                                      d_neighbor_list,
                                      d_fine_mask,
                                      d_part_to_real,
                                      d_real_to_part,
                                      padded_nodes);
    _preparePrefixSumL0_new<<<1, block_size>>>(
        d_prefix, d_fine_mask, d_part_to_real, padded_nodes);
    thrust::exclusive_scan(thrust::device_ptr<int>(d_prefix),
                           thrust::device_ptr<int>(d_prefix) + warp_count,
                           thrust::device_ptr<int>(d_prefix_sum));
    _buildLevel1_new<<<1, GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE>>>(d_level_size,
                                                                 d_coarse,
                                                                 d_going_next,
                                                                 d_fine_mask,
                                                                 d_prefix_sum,
                                                                 d_prefix,
                                                                 d_part_to_real,
                                                                 padded_nodes);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    std::vector<unsigned int> fine_mask(valid_nodes);
    std::vector<int>          coarse(valid_nodes);
    std::vector<int>          going_next(valid_nodes);
    std::vector<int>          prefix(warp_count);
    int2                      level_size{};
    CUDA_SAFE_CALL(cudaMemcpy(fine_mask.data(),
                              d_fine_mask,
                              valid_nodes * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(coarse.data(),
                              d_coarse,
                              valid_nodes * sizeof(int),
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(going_next.data(),
                              d_going_next,
                              valid_nodes * sizeof(int),
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(prefix.data(),
                              d_prefix,
                              warp_count * sizeof(int),
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(&level_size,
                              d_level_size + 1,
                              sizeof(int2),
                              cudaMemcpyDeviceToHost));

    HierarchySelfTestResult result;
    result.valid_nodes         = valid_nodes;
    result.padded_nodes        = padded_nodes;
    result.expected_components = 2;
    result.actual_components   = level_size.x;
    for(int vertex = 0; vertex < valid_nodes; ++vertex)
    {
        const bool first_group = vertex < GPU_MAS_BANKSIZE;
        const unsigned int expected_mask = first_group ? 0xffffffffU : 0x7U;
        const int expected_coarse = first_group ? 0 : 1;
        if(fine_mask[vertex] != expected_mask)
            ++result.fine_mask_mismatches;
        if(coarse[vertex] != expected_coarse)
            ++result.coarse_mapping_mismatches;
        if(going_next[vertex] != padded_nodes + expected_coarse)
            ++result.going_next_mismatches;
    }
    result.passed = prefix[0] == 1 && prefix[1] == 1
                    && level_size.x == result.expected_components
                    && level_size.y == padded_nodes
                    && result.fine_mask_mismatches == 0
                    && result.coarse_mapping_mismatches == 0
                    && result.going_next_mismatches == 0;

    CUDA_SAFE_CALL(cudaFree(d_neighbor_start));
    CUDA_SAFE_CALL(cudaFree(d_neighbor_count));
    CUDA_SAFE_CALL(cudaFree(d_neighbor_list));
    CUDA_SAFE_CALL(cudaFree(d_fine_mask));
    CUDA_SAFE_CALL(cudaFree(d_part_to_real));
    CUDA_SAFE_CALL(cudaFree(d_real_to_part));
    CUDA_SAFE_CALL(cudaFree(d_prefix));
    CUDA_SAFE_CALL(cudaFree(d_prefix_sum));
    CUDA_SAFE_CALL(cudaFree(d_level_size));
    CUDA_SAFE_CALL(cudaFree(d_coarse));
    CUDA_SAFE_CALL(cudaFree(d_going_next));
    return result;
}

__global__ void _buildCML0(const unsigned int* _neighborStart,
                           unsigned int*       _neighborNum,
                           unsigned int*       _neighborList,
                           unsigned int*       _fineConnectedMsk,
                           int                 vertNum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= vertNum)
        return;
    int          warpId      = idx / GPU_MAS_BANKSIZE;
    int          laneId      = idx % GPU_MAS_BANKSIZE;
    int          numNeighbor = _neighborNum[idx];
    unsigned int connectMsk  = (1U << laneId);
    int          nk          = 0;
    int          startId     = _neighborStart[idx];
    for(int i = 0; i < numNeighbor; i++)
    {
        int vIdConnected     = _neighborList[startId + i];
        int warpIdxConnected = vIdConnected / GPU_MAS_BANKSIZE;
        if(warpId == warpIdxConnected)
        {
            unsigned int laneIdxConnected = vIdConnected % GPU_MAS_BANKSIZE;
            connectMsk |= (1U << laneIdxConnected);
        }
        else
        {
            _neighborList[startId + nk] = vIdConnected;
            nk++;
        }
    }
    _neighborNum[idx]      = nk;
    _fineConnectedMsk[idx] = connectMsk;
}

__global__ void _buildCML0_new(const unsigned int* _neighborStart,
                               unsigned int*       _neighborNum,
                               unsigned int*       _neighborList,
                               unsigned int*       _fineConnectedMsk,
                               int*                _partId_map_real,
                               int*                _real_map_partId,
                               int                 number)
{
    int tdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tdx >= number)
        return;
    int warpId = tdx / GPU_MAS_BANKSIZE;
    int laneId = tdx % GPU_MAS_BANKSIZE;
    int idx    = _partId_map_real[tdx];
    if(idx >= 0)
    {

        int          numNeighbor = _neighborNum[idx];
        unsigned int connectMsk  = (1U << laneId);
        int          nk          = 0;
        int          startId     = _neighborStart[idx];
        for(int i = 0; i < numNeighbor; i++)
        {
            int vIdConnected = _neighborList[startId + i];
            //vIdConnected         = _real_map_partId[vIdConnected];
            int warpIdxConnected = _real_map_partId[vIdConnected] / GPU_MAS_BANKSIZE;
            if(warpId == warpIdxConnected)
            {
                unsigned int laneIdxConnected = _real_map_partId[vIdConnected] % GPU_MAS_BANKSIZE;
                connectMsk |= (1U << laneIdxConnected);
            }
            else
            {
                _neighborList[startId + nk] = vIdConnected;
                nk++;
            }
        }
        _neighborNum[idx]      = nk;
        _fineConnectedMsk[idx] = connectMsk;
    }
}


__device__ unsigned int _LanemaskLt(int laneIdx)
{
    return (1U << laneIdx) - 1;
}

__global__ void _preparePrefixSumL0(int* _prefixOriginal, unsigned int* _fineConnectedMsk, int vertNum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= vertNum)
        return;
    int          warpId      = idx / GPU_MAS_BANKSIZE;
    int          localWarpId = threadIdx.x / GPU_MAS_BANKSIZE;
    int          laneId      = idx % GPU_MAS_BANKSIZE;
    const unsigned int active_mask = __activemask();
    unsigned int connectMsk  = _fineConnectedMsk[idx];
    //unsigned int connectMsk = cacheMask1;
    __shared__ int unsigned cacheMask[DEFAULT_BLOCKSIZE];
    __shared__ int          prefixSum[DEFAULT_WARPNUM];
    if(laneId == 0)
    {
        prefixSum[localWarpId] = 0;
    }
    cacheMask[threadIdx.x] = connectMsk;
    __syncwarp(active_mask);
    unsigned int visited   = (1U << laneId);
    while(connectMsk != 0xffffffffU)
    {
        unsigned int todo = visited ^ connectMsk;

        if(!todo)
            break;

        unsigned int nextVist = __ffs(todo) - 1;
        visited |= (1U << nextVist);
        connectMsk |= cacheMask[nextVist + localWarpId * GPU_MAS_BANKSIZE];  //__shfl_sync(0xffffffff, cacheMask, nextVist);//?????!!!!!
    }

    _fineConnectedMsk[idx] = connectMsk;

    unsigned int electedPrefix = __popc(connectMsk & _LanemaskLt(laneId));

    if(electedPrefix == 0)
    {
        //prefixSum[warpId]++;
        atomicAdd(prefixSum + localWarpId, 1);
    }

    __syncwarp(active_mask);
    if(laneId == 0)
    {
        _prefixOriginal[warpId] = prefixSum[localWarpId];
    }
}

__global__ void _preparePrefixSumL0_new(int*          _prefixOriginal,
                                        unsigned int* _fineConnectedMsk,
                                        int*          _partId_map_real,
                                        //int*          _real_map_partId,
                                        int vertNum)
{
    int tdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tdx >= vertNum)
        return;
    int warpId      = tdx / GPU_MAS_BANKSIZE;
    int localWarpId = threadIdx.x / GPU_MAS_BANKSIZE;
    int laneId      = tdx % GPU_MAS_BANKSIZE;

    int idx = _partId_map_real[tdx];
    const unsigned int active_mask = __ballot_sync(__activemask(), idx >= 0);


    //unsigned int connectMsk = cacheMask1;
    __shared__ int unsigned cacheMask[DEFAULT_BLOCKSIZE];
    __shared__ int          prefixSum[DEFAULT_WARPNUM];

    if(idx >= 0)
    {

        unsigned int connectMsk = _fineConnectedMsk[idx];
        if(laneId == 0)
        {
            prefixSum[localWarpId] = 0;
        }
        cacheMask[threadIdx.x] = connectMsk;
        __syncwarp(active_mask);
        unsigned int visited   = (1U << laneId);
        while(connectMsk != 0xffffffffU)
        {
            unsigned int todo = visited ^ connectMsk;

            if(!todo)
                break;

            unsigned int nextVist = __ffs(todo) - 1;
            visited |= (1U << nextVist);
            connectMsk |= cacheMask[nextVist + localWarpId * GPU_MAS_BANKSIZE];  //__shfl_sync(0xffffffff, cacheMask, nextVist);//?????!!!!!
        }

        _fineConnectedMsk[idx] = connectMsk;

        unsigned int electedPrefix = __popc(connectMsk & _LanemaskLt(laneId));

        if(electedPrefix == 0)
        {
            //prefixSum[warpId]++;
            atomicAdd(prefixSum + localWarpId, 1);
        }

        __syncwarp(active_mask);
        if(laneId == 0)
        {
            _prefixOriginal[warpId] = prefixSum[localWarpId];
        }
    }
}


__global__ void _buildLevel1(int2*               _levelSize,
                             int*                _coarseSpaceTable,
                             int*                _goingNext,
                             const unsigned int* _fineConnectedMsk,
                             const int*          _prefixSumOriginal,
                             const int*          _prefixOriginal,
                             int                 vertNum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= vertNum)
        return;
    int warpId      = idx / GPU_MAS_BANKSIZE;
    int localWarpId = threadIdx.x / GPU_MAS_BANKSIZE;
    int laneId      = idx % GPU_MAS_BANKSIZE;
    const unsigned int active_mask = __activemask();

    __shared__ unsigned int electedMask[GPU_MAS_BANKSIZE];
    __shared__ unsigned int lanePrefix[GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE];
    if(laneId == 0)
    {
        electedMask[localWarpId] = 0;
    }
    __syncwarp(active_mask);
    if(idx == vertNum - 1)
    {
        _levelSize[1].x = _prefixSumOriginal[warpId] + _prefixOriginal[warpId];
        _levelSize[1].y = (vertNum + GPU_MAS_BANKSIZE - 1) / GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE;
    }

    unsigned int connMsk = _fineConnectedMsk[idx];

    unsigned int electedPrefix = __popc(connMsk & _LanemaskLt(laneId));

    if(electedPrefix == 0)
    {
        atomicOr(electedMask + localWarpId, (1U << laneId));
    }
    __syncwarp(active_mask);

    //unsigned int lanePrefix2 = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
    //lanePrefix2 += _prefixSumOriginal[warpId];

    //unsigned int elected_lane = __ffs(connMsk) - 1;
    //unsigned int theLanePrefix = __shfl_sync(0xffffffff, lanePrefix2, elected_lane);

    lanePrefix[threadIdx.x] = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
    lanePrefix[threadIdx.x] += _prefixSumOriginal[warpId];
    __syncwarp(active_mask);

    unsigned int elected_lane = __ffs(connMsk) - 1;
    unsigned int theLanePrefix = lanePrefix[elected_lane + GPU_MAS_BANKSIZE * localWarpId];  //__shfl_sync(0xffffffff, lanePrefix, elected_lane);


    _coarseSpaceTable[idx + 0 * vertNum] = theLanePrefix;
    _goingNext[idx] = theLanePrefix + (vertNum + GPU_MAS_BANKSIZE - 1) / GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE;
}


__global__ void _buildLevel1_new(int2*               _levelSize,
                                 int*                _coarseSpaceTable,
                                 int*                _goingNext,
                                 const unsigned int* _fineConnectedMsk,
                                 const int*          _prefixSumOriginal,
                                 const int*          _prefixOriginal,
                                 int*                _partId_map_real,
                                 int                 number)
{
    int tdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tdx >= number)
        return;
    int warpId      = tdx / GPU_MAS_BANKSIZE;
    int localWarpId = threadIdx.x / GPU_MAS_BANKSIZE;
    int laneId      = tdx % GPU_MAS_BANKSIZE;
    const unsigned int warp_mask = __activemask();

    __shared__ unsigned int electedMask[GPU_MAS_BANKSIZE];
    __shared__ unsigned int lanePrefix[GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE];
    if(laneId == 0)
    {
        electedMask[localWarpId] = 0;
    }
    __syncwarp(warp_mask);
    if(tdx == number - 1)
    {
        _levelSize[1].x = _prefixSumOriginal[warpId] + _prefixOriginal[warpId];
        _levelSize[1].y = (number + GPU_MAS_BANKSIZE - 1) / GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE;
    }
    int idx = _partId_map_real[tdx];
    const unsigned int active_mask = __ballot_sync(warp_mask, idx >= 0);
    if(idx >= 0)
    {

        unsigned int connMsk = _fineConnectedMsk[idx];

        unsigned int electedPrefix = __popc(connMsk & _LanemaskLt(laneId));

        if(electedPrefix == 0)
        {
            atomicOr(electedMask + localWarpId, (1U << laneId));
        }
        __syncwarp(active_mask);

        //unsigned int lanePrefix2 = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
        //lanePrefix2 += _prefixSumOriginal[warpId];

        //unsigned int elected_lane = __ffs(connMsk) - 1;
        //unsigned int theLanePrefix = __shfl_sync(0xffffffff, lanePrefix2, elected_lane);

        lanePrefix[threadIdx.x] = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
        lanePrefix[threadIdx.x] += _prefixSumOriginal[warpId];
        __syncwarp(active_mask);

        unsigned int elected_lane = __ffs(connMsk) - 1;
        unsigned int theLanePrefix =
            lanePrefix[elected_lane + GPU_MAS_BANKSIZE * localWarpId];  //__shfl_sync(0xffffffff, lanePrefix, elected_lane);


        _coarseSpaceTable[idx] = theLanePrefix;
        _goingNext[idx] = theLanePrefix + (number + GPU_MAS_BANKSIZE - 1) / GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE;
    }
}


__global__ void _buildConnectMaskLx(const unsigned int* _neighborStart,
                                    unsigned int*       _neighborNum,
                                    unsigned int*       _neighborList,
                                    int*                _coarseSpaceTable,
                                    unsigned int*       _nextConnectedMsk,
                                    const unsigned int* _fineConnectedMsk,
                                    int                 level,
                                    int                 vertNum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= vertNum)
        return;
    int warpId      = idx / GPU_MAS_BANKSIZE;
    int localWarpId = threadIdx.x / GPU_MAS_BANKSIZE;
    int laneId      = idx % GPU_MAS_BANKSIZE;
    const unsigned int active_mask = __activemask();

    unsigned int prefixMsk = _fineConnectedMsk[idx];
    unsigned int connMsk   = 0;
    unsigned int coarseIdx = _coarseSpaceTable[(level - 1) * vertNum + idx];
    int          kn        = _neighborNum[idx];
    int          nk        = 0;
    int          startId   = _neighborStart[idx];
    for(int i = 0; i < kn; i++)
    {
        unsigned int connect = _neighborList[startId + i];
        unsigned int coarseConnect = _coarseSpaceTable[(level - 1) * vertNum + connect];

        if(coarseIdx / GPU_MAS_BANKSIZE == coarseConnect / GPU_MAS_BANKSIZE)
        {
            unsigned int off = coarseConnect % GPU_MAS_BANKSIZE;
            connMsk |= (1U << off);
        }
        else
        {
            _neighborList[startId + nk] = connect;
            nk++;
        }
    }

    _neighborNum[idx] = nk;

    __shared__ int cacheMsk[DEFAULT_BLOCKSIZE];
    cacheMsk[threadIdx.x] = 0;
    __syncwarp(active_mask);

    unsigned int cacheIndex = 0;
    if(__popc(prefixMsk) == GPU_MAS_BANKSIZE)
    {
        atomicOr(cacheMsk + localWarpId * GPU_MAS_BANKSIZE, connMsk);
        cacheIndex = localWarpId * GPU_MAS_BANKSIZE;
        //if (laneId == 0) {
        //	cacheMsk[localWarpId] = 0;
        //}
    }
    else
    {
        unsigned int electedLane = __ffs(prefixMsk) - 1;
        if(connMsk)
        {
            atomicOr(cacheMsk + localWarpId * GPU_MAS_BANKSIZE + electedLane, connMsk);
        }
        cacheIndex = localWarpId * GPU_MAS_BANKSIZE + electedLane;
    }
    __syncwarp(active_mask);
    connMsk = cacheMsk[cacheIndex];

    unsigned int electedPrefix = __popc(prefixMsk & _LanemaskLt(laneId));

    if(connMsk && electedPrefix == 0)
    {
        atomicOr(_nextConnectedMsk + coarseIdx, connMsk);
    }
}

__global__ void _buildConnectMaskLx_new(const unsigned int* _neighborStart,
                                        unsigned int*       _neighborNum,
                                        unsigned int*       _neighborList,
                                        int*                _coarseSpaceTable,
                                        unsigned int*       _nextConnectedMsk,
                                        const unsigned int* _fineConnectedMsk,
                                        int                 level,
                                        int*                _partId_map_real,
                                        //int*                _real_map_partId,
                                        int vertNum,
                                        int number)
{
    int tdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tdx >= number)
        return;
    int            warpId      = tdx / GPU_MAS_BANKSIZE;
    int            localWarpId = threadIdx.x / GPU_MAS_BANKSIZE;
    int            laneId      = tdx % GPU_MAS_BANKSIZE;
    __shared__ int cacheMsk[DEFAULT_BLOCKSIZE];
    int            idx = _partId_map_real[tdx];
    const unsigned int active_mask = __ballot_sync(__activemask(), idx >= 0);
    if(idx >= 0)
    {

        unsigned int prefixMsk = _fineConnectedMsk[idx];
        unsigned int connMsk   = 0;
        unsigned int coarseIdx = _coarseSpaceTable[(level - 1) * vertNum + idx];
        int          kn        = _neighborNum[idx];
        int          nk        = 0;
        int          startId   = _neighborStart[idx];
        for(int i = 0; i < kn; i++)
        {
            unsigned int connect = _neighborList[startId + i];
            unsigned int coarseConnect = _coarseSpaceTable[(level - 1) * vertNum + connect];

            if(coarseIdx / GPU_MAS_BANKSIZE == coarseConnect / GPU_MAS_BANKSIZE)
            {
                unsigned int off = coarseConnect % GPU_MAS_BANKSIZE;
                connMsk |= (1U << off);
            }
            else
            {
                _neighborList[startId + nk] = connect;
                nk++;
            }
        }

        _neighborNum[idx] = nk;


        cacheMsk[threadIdx.x] = 0;
        __syncwarp(active_mask);

        unsigned int cacheIndex = 0;
        if(__popc(prefixMsk) == GPU_MAS_BANKSIZE)
        {
            atomicOr(cacheMsk + localWarpId * GPU_MAS_BANKSIZE, connMsk);
            cacheIndex = localWarpId * GPU_MAS_BANKSIZE;
            //if (laneId == 0) {
            //	cacheMsk[localWarpId] = 0;
            //}
        }
        else
        {
            unsigned int electedLane = __ffs(prefixMsk) - 1;
            if(connMsk)
            {
                atomicOr(cacheMsk + localWarpId * GPU_MAS_BANKSIZE + electedLane, connMsk);
            }
            cacheIndex = localWarpId * GPU_MAS_BANKSIZE + electedLane;
        }
        __syncwarp(active_mask);
        connMsk = cacheMsk[cacheIndex];

        unsigned int electedPrefix = __popc(prefixMsk & _LanemaskLt(laneId));

        if(connMsk && electedPrefix == 0)
        {
            atomicOr(_nextConnectedMsk + coarseIdx, connMsk);
        }
    }
}


__global__ void _nextLevelCluster(unsigned int* _nextConnectedMsk, unsigned int* _nextPrefix, int number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int            warpId      = idx / GPU_MAS_BANKSIZE;
    int            localWarpId = threadIdx.x / GPU_MAS_BANKSIZE;
    int            laneId      = idx % GPU_MAS_BANKSIZE;
    const unsigned int active_mask = __activemask();
    __shared__ int prefixSum[DEFAULT_WARPNUM];
    if(laneId == 0)
    {
        prefixSum[localWarpId] = 0;
    }
    unsigned int connMsk = (1U << laneId);

    connMsk |= _nextConnectedMsk[idx];

    //unsigned int cachedMsk = connMsk;

    __shared__ unsigned int cachedMsk[DEFAULT_BLOCKSIZE];
    cachedMsk[threadIdx.x] = connMsk;
    __syncwarp(active_mask);
    unsigned int visited   = (1U << laneId);

    while(true)
    {
        unsigned int todo = visited ^ connMsk;

        if(!todo)
            break;

        unsigned int nextVisit = __ffs(todo) - 1;

        visited |= (1U << nextVisit);

        connMsk |= cachedMsk[nextVisit + localWarpId * GPU_MAS_BANKSIZE];  //__shfl_sync(0xffffffff, cachedMsk, nextVisit);
    }

    _nextConnectedMsk[idx] = connMsk;

    unsigned int electedPrefix = __popc(connMsk & _LanemaskLt(laneId));

    if(electedPrefix == 0)
    {
        atomicAdd(prefixSum + localWarpId, 1);
    }

    __syncwarp(active_mask);
    if(laneId == 0)
        _nextPrefix[warpId] = prefixSum[localWarpId];
}

__global__ void _prefixSumLx(int2*         _levelSize,
                             unsigned int* _nextPrefix,
                             unsigned int* _nextPrefixSum,
                             unsigned int* _nextConnectMsk,
                             int*          _goingNext,
                             int           level,
                             int           levelBegin,
                             int           number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int warpId      = idx / GPU_MAS_BANKSIZE;
    int localWarpId = threadIdx.x / GPU_MAS_BANKSIZE;
    int laneId      = idx % GPU_MAS_BANKSIZE;
    const unsigned int active_mask = __activemask();

    __shared__ unsigned int electedMask[GPU_MAS_BANKSIZE];
    __shared__ unsigned int lanePrefix[GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE];
    if(laneId == 0)
    {
        electedMask[localWarpId] = 0;
    }
    __syncwarp(active_mask);

    if(idx == number - 1)
    {
        _levelSize[level + 1].x = _nextPrefixSum[warpId] + _nextPrefix[warpId];
        _levelSize[level + 1].y = levelBegin + (number + GPU_MAS_BANKSIZE - 1) / GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE;
    }

    unsigned int connMsk = _nextConnectMsk[idx];

    unsigned int electedPrefix = __popc(connMsk & _LanemaskLt(laneId));

    if(electedPrefix == 0)
    {
        atomicOr(electedMask + localWarpId, (1U << laneId));
    }
    __syncwarp(active_mask);

    lanePrefix[threadIdx.x] = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
    lanePrefix[threadIdx.x] += _nextPrefixSum[warpId];
    __syncwarp(active_mask);

    unsigned int elected_lane = __ffs(connMsk) - 1;
    unsigned int theLanePrefix = lanePrefix[elected_lane + GPU_MAS_BANKSIZE * localWarpId];  //__shfl_sync(0xffffffff, lanePrefix, elected_lane);

    _nextConnectMsk[idx] = theLanePrefix;
    _goingNext[idx + levelBegin] =
        theLanePrefix + levelBegin + (number + GPU_MAS_BANKSIZE - 1) / GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE;
}

__global__ void _computeNextLevel(int*          _coarseSpaceTable,
                                  unsigned int* _nextConnectMsk,
                                  int           level,
                                  int           number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int next = _coarseSpaceTable[(level - 1) * number + idx];
    _coarseSpaceTable[(level)*number + idx] = _nextConnectMsk[next];
}

__global__ void _aggregationKernel(int*                _denseLevel,
                                   __GEIGEN__::itable* _coarseTable,
                                   int*                _goingNext,
                                   int                 levelNum,
                                   int                 number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int currentId = idx;
    //int aggLevel  = levelNum - 1;
    //__shared__ int4 ctable[DEFAULT_BLOCKSIZE];
    __GEIGEN__::itable ctable;
    for(int l = 0; l < levelNum - 1; l++)
    {
        int next = _goingNext[currentId];

        //int next0 = __shfl_sync(0xffffffff, next, 0);
        //printf("%d   %d   %d    %d\n", next, next0, l,  idx);
        //if (next == next0) {
        //	aggLevel = std::min(l, aggLevel);
        //}

        currentId           = next;
        *(ctable.index + l) = next;
    }

    //_denseLevel[idx] = aggLevel;

    //printf("%d   %d\n", aggLevel, idx);

    _coarseTable[idx] = ctable;
}




__global__ void expand_symmetric_mas32(
    __GEIGEN__::GPUMas32MatrixT* full_matrix,
    const __GEIGEN__::GPUMas32MatrixSymT* symmetric_matrix,
    int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    const int dimension = GPU_MAS_BANKSIZE * 3;
    const int mat_id    = idx / dimension;
    const int row       = idx % dimension;
    for(int col = 0; col < dimension; ++col)
    {
        const int block_row = row / 3;
        const int block_col = col / 3;
        if(block_col >= block_row)
        {
            const int block_index = GPU_MAS_BANKSIZE * block_row
                                    - block_row * (block_row + 1) / 2 + block_col;
            full_matrix[mat_id].m[row][col] =
                symmetric_matrix[mat_id].M[block_index](row % 3, col % 3);
        }
        else
        {
            const int block_index = GPU_MAS_BANKSIZE * block_col
                                    - block_col * (block_col + 1) / 2 + block_row;
            full_matrix[mat_id].m[row][col] =
                symmetric_matrix[mat_id].M[block_index](col % 3, row % 3);
        }
    }
    if(full_matrix[mat_id].m[row][row] == 0.0)
        full_matrix[mat_id].m[row][row] = 1.0;
}

// GPU_IPC's MAS32 inversion stores one column in shared memory instead of an
// entire 96x96 matrix, keeping the kernel below the RTX 3070 shared-memory cap.
__global__ void __inverse6_P96x96(__GEIGEN__::GPUMas32MatrixSymf* preconditioner,
                                  __GEIGEN__::GPUMas32MatrixT* full_matrix,
                                  int numbers)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= numbers)
        return;

    const int dimension = GPU_MAS_BANKSIZE * 3;
    const int mat_id    = idx / dimension;
    const int i         = idx % dimension;
    const int local_mat = threadIdx.x / dimension;
    __shared__ Precision_T column[32 / GPU_MAS_BANKSIZE][GPU_MAS_BANKSIZE * 3];

    for(int j = 0; j < dimension; ++j)
    {
        __syncthreads();
        const Precision_T pivot = full_matrix[mat_id].m[j][j];
        column[local_mat][i] = full_matrix[mat_id].m[i][j];
        __syncthreads();
        full_matrix[mat_id].m[i][j] = i == j ? 1.0 : 0.0;
        __syncthreads();
        full_matrix[mat_id].m[j][i] /= pivot;
        __syncthreads();
        for(int k = 0; k < dimension; ++k)
        {
            if(k != j)
            {
                const Precision_T rate = -column[local_mat][k];
                full_matrix[mat_id].m[k][i] += rate * full_matrix[mat_id].m[j][i];
            }
        }
    }
    __syncthreads();

    for(int row = 0; row < dimension; ++row)
    {
        const int block_row = row / 3;
        const int block_col = i / 3;
        if(block_col >= block_row)
        {
            const int block_index = GPU_MAS_BANKSIZE * block_row
                                    - block_row * (block_row + 1) / 2 + block_col;
            preconditioner[mat_id].M[block_index](row % 3, i % 3) =
                static_cast<float>(full_matrix[mat_id].m[row][i]);
        }
    }
}

// Stable alternative to the legacy Gauss-Jordan explicit inverse.  The MAS
// hierarchy and 32-node local domains are unchanged; only each local solve is
// represented by an FP64 Cholesky factor.  Near-null affine modes use the
// same tiny relative pivot floor as a conventional local SPD factorization.
__global__ void factor_mas32_cholesky(__GEIGEN__::GPUMas32MatrixT* matrices,
                                      int matrix_count,
                                      int* status)
{
    constexpr int dimension = GPU_MAS_BANKSIZE * 3;
    const int matrix_id = blockIdx.x;
    if(matrix_id >= matrix_count)
        return;
    auto& matrix = matrices[matrix_id];
    __shared__ double pivot;
    __shared__ double floor_value;
    if(threadIdx.x == 0)
    {
        double scale = 0.0;
        for(int i = 0; i < dimension; ++i)
            scale = fmax(scale, fabs(matrix.m[i][i]));
        floor_value = fmax(1.0e-12, scale * 1.0e-12);
    }
    __syncthreads();

    for(int k = 0; k < dimension; ++k)
    {
        if(threadIdx.x == 0)
        {
            double value = matrix.m[k][k];
            if(!isfinite(value))
            {
                atomicAdd(status, 1);
                value = floor_value;
            }
            if(value < floor_value)
            {
                atomicAdd(status + 1, 1);
                value = floor_value;
            }
            pivot = sqrt(value);
            matrix.m[k][k] = pivot;
        }
        __syncthreads();
        for(int i = k + 1 + threadIdx.x; i < dimension; i += blockDim.x)
        {
            double value = matrix.m[i][k] / pivot;
            if(!isfinite(value))
            {
                atomicAdd(status, 1);
                value = 0.0;
            }
            matrix.m[i][k] = value;
        }
        __syncthreads();
        for(int index = threadIdx.x; index < dimension * dimension; index += blockDim.x)
        {
            const int i = index / dimension;
            const int j = index - i * dimension;
            if(i >= j && j > k)
                matrix.m[i][j] -= matrix.m[i][k] * matrix.m[j][k];
        }
        __syncthreads();
    }
}

__global__ void apply_mas32_cholesky(
    const __GEIGEN__::GPUMas32MatrixT* factors,
    const Eigen::Vector3f* rhs,
    Precision_T3* output,
    int matrix_count)
{
    constexpr int dimension = GPU_MAS_BANKSIZE * 3;
    const int matrix_id = blockIdx.x;
    if(matrix_id >= matrix_count)
        return;
    const auto& factor = factors[matrix_id];
    const int lane = threadIdx.x;
    __shared__ double values[dimension];
    const int first_node = matrix_id * GPU_MAS_BANKSIZE;
    for(int i = 0; i < dimension; ++i)
    {
        double sum = 0.0;
        for(int j = lane; j < i; j += 32)
            sum += factor.m[i][j] * values[j];
        for(int offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if(lane == 0)
            values[i] = (static_cast<double>(rhs[first_node + i / 3][i % 3]) - sum)
                        / factor.m[i][i];
        __syncwarp();
    }
    for(int i = dimension - 1; i >= 0; --i)
    {
        double sum = 0.0;
        for(int j = i + 1 + lane; j < dimension; j += 32)
            sum += factor.m[j][i] * values[j];
        for(int offset = 16; offset; offset >>= 1)
            sum += __shfl_down_sync(0xffffffffu, sum, offset);
        if(lane == 0)
        {
            values[i] = (values[i] - sum) / factor.m[i][i];
            const int node = first_node + i / 3;
            const float value = static_cast<float>(values[i]);
            if(i % 3 == 0) output[node].x = value;
            else if(i % 3 == 1) output[node].y = value;
            else output[node].z = value;
        }
        __syncwarp();
    }
}

// Build the symmetric FP64 inverse from the already validated Cholesky
// factor once per Newton matrix.  This retains the stable factorization while
// replacing serial triangular solves inside every PCG iteration by a parallel
// dense matrix-vector product.
__global__ void invert_mas32_cholesky(
    const __GEIGEN__::GPUMas32MatrixT* factors,
    __GEIGEN__::GPUMas32MatrixSymT* inverse,
    int matrix_count,
    int* status)
{
    constexpr int dimension = GPU_MAS_BANKSIZE * 3;
    const int matrix_id = blockIdx.x;
    const int column = threadIdx.x;
    if(matrix_id >= matrix_count || column >= dimension)
        return;
    const auto& factor = factors[matrix_id];
    double solution[dimension];
    for(int i = 0; i < dimension; ++i)
    {
        double sum = i == column ? 1.0 : 0.0;
        for(int j = 0; j < i; ++j)
            sum -= factor.m[i][j] * solution[j];
        solution[i] = sum / factor.m[i][i];
    }
    for(int i = dimension - 1; i >= 0; --i)
    {
        double sum = solution[i];
        for(int j = i + 1; j < dimension; ++j)
            sum -= factor.m[j][i] * solution[j];
        solution[i] = sum / factor.m[i][i];
        if(!isfinite(solution[i])) atomicAdd(status, 1);
    }
    for(int row = 0; row < dimension; ++row)
    {
        const int block_row = row / 3;
        const int block_col = column / 3;
        if(block_col >= block_row)
        {
            const int index = GPU_MAS_BANKSIZE * block_row
                              - block_row * (block_row + 1) / 2 + block_col;
            inverse[matrix_id].M[index](row % 3, column % 3) = solution[row];
        }
    }
}

__global__ void apply_mas32_inverse_fp64(
    const __GEIGEN__::GPUMas32MatrixSymT* inverse,
    const Eigen::Vector3f* rhs,
    Precision_T3* output,
    int matrix_count)
{
    constexpr int dimension = GPU_MAS_BANKSIZE * 3;
    const int matrix_id = blockIdx.x;
    const int row = threadIdx.x;
    if(matrix_id >= matrix_count || row >= dimension)
        return;
    const int first_node = matrix_id * GPU_MAS_BANKSIZE;
    double value = 0.0;
    for(int column = 0; column < dimension; ++column)
        value += packed_scalar(inverse[matrix_id],row,column)
                 * static_cast<double>(rhs[first_node + column / 3][column % 3]);
    const int node = first_node + row / 3;
    const float converted = static_cast<float>(value);
    if(row % 3 == 0) output[node].x = converted;
    else if(row % 3 == 1) output[node].y = converted;
    else output[node].z = converted;
}


__global__ void __buildMultiLevelR_optimized_new(const double3* _R,
                                                 Eigen::Vector3f*  _multiLR,
                                                 int*           _goingNext,
                                                 int*           _prefixOrigin,
                                                 unsigned int*  _fineConnectMsk,
                                                 int* _partId_map_real,
                                                 int  levelNum,
                                                 int  numbers)
{
    int pdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pdx >= numbers)
        return;

    Eigen::Vector3f r;
    int             idx = _partId_map_real[pdx];
    if(idx >= 0)
    {

        r[0] = _R[idx].x;
        r[1] = _R[idx].y;
        r[2] = _R[idx].z;
    }
    else
    {
        r[0] = 0;
        r[1] = 0;
        r[2] = 0;
    }

    int laneId      = threadIdx.x % GPU_MAS_BANKSIZE;
    int localWarpId = threadIdx.x / GPU_MAS_BANKSIZE;
    int gwarpId     = pdx / GPU_MAS_BANKSIZE;
    int level       = 0;
    //int rdx         = _real_map_partId[idx];
    _multiLR[pdx] = r;

    __shared__ FloatP c_sumResidual[DEFAULT_BLOCKSIZE * 3];

    __shared__ int prefixSum[DEFAULT_WARPNUM];

    if(laneId == 0)
    {
        prefixSum[localWarpId] = _prefixOrigin[gwarpId];
    }
    const unsigned int warp_mask = __activemask();
    __syncwarp(warp_mask);

    const unsigned int active_mask = __ballot_sync(warp_mask, idx >= 0);

    if(idx >= 0)
    {

        unsigned int connectMsk = _fineConnectMsk[idx];

        if(prefixSum[localWarpId] == 1)
        {
            for(int iter = 1; iter < GPU_MAS_BANKSIZE; iter <<= 1)
            {
                const float tmpx = __shfl_down_sync(active_mask, r[0], iter);
                const float tmpy = __shfl_down_sync(active_mask, r[1], iter);
                const float tmpz = __shfl_down_sync(active_mask, r[2], iter);
                const int source_lane = laneId + iter;
                if(source_lane < GPU_MAS_BANKSIZE
                   && (active_mask & (1U << source_lane)))
                {
                    r[0] += tmpx;
                    r[1] += tmpy;
                    r[2] += tmpz;
                }
            }
            //int level = 0;

            if(laneId == __ffs(active_mask) - 1)
            {
                while(level < levelNum - 1)
                {
                    level++;
                    idx = _goingNext[idx];
                    atomicAdd(&(_multiLR[idx][0]), r[0]);
                    atomicAdd(&(_multiLR[idx][1]), r[1]);
                    atomicAdd(&(_multiLR[idx][2]), r[2]);
                }
            }
            return;
        }
        else
        {
            int elected_lane = __ffs(connectMsk) - 1;

            c_sumResidual[threadIdx.x]                         = 0;
            c_sumResidual[threadIdx.x + DEFAULT_BLOCKSIZE]     = 0;
            c_sumResidual[threadIdx.x + 2 * DEFAULT_BLOCKSIZE] = 0;
            __syncwarp(active_mask);
            atomicAdd(c_sumResidual + localWarpId * GPU_MAS_BANKSIZE + elected_lane, r[0]);
            atomicAdd(c_sumResidual + localWarpId * GPU_MAS_BANKSIZE + elected_lane + DEFAULT_BLOCKSIZE,
                      r[1]);
            atomicAdd(c_sumResidual + localWarpId * GPU_MAS_BANKSIZE + elected_lane + 2 * DEFAULT_BLOCKSIZE,
                      r[2]);
            __syncwarp(active_mask);

            unsigned int electedPrefix = __popc(connectMsk & _LanemaskLt(laneId));
            if(electedPrefix == 0)
            {
                while(level < levelNum - 1)
                {
                    level++;
                    idx = _goingNext[idx];
                    atomicAdd(&(_multiLR[idx][0]), c_sumResidual[threadIdx.x]);
                    atomicAdd(&(_multiLR[idx][1]),
                              c_sumResidual[threadIdx.x + DEFAULT_BLOCKSIZE]);
                    atomicAdd(&(_multiLR[idx][2]),
                              c_sumResidual[threadIdx.x + DEFAULT_BLOCKSIZE * 2]);
                }
            }
        }
    }
}


__global__ void __collectFinalZ_new(double3*                  _Z,
                                    const Precision_T3*       d_multiLevelZ,
                                    const __GEIGEN__::itable* _coarseTable,
                                    int*                      _real_map_partId,
                                    int                       levelnum,
                                    int                       number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    Precision_T3 cz;  // = d_multiLevelZ[idx];
    int          rdx            = _real_map_partId[idx];
    cz.x                        = d_multiLevelZ[rdx].x;
    cz.y                        = d_multiLevelZ[rdx].y;
    cz.z                        = d_multiLevelZ[rdx].z;
    __GEIGEN__::itable table    = _coarseTable[idx];
    int*               tablePtr = table.index;
    for(int i = 1; i < levelnum; i++)
    {
        int now = *(tablePtr + i - 1);
        cz.x += d_multiLevelZ[now].x;
        cz.y += d_multiLevelZ[now].y;
        cz.z += d_multiLevelZ[now].z;
    }

    _Z[idx].x = cz.x;
    _Z[idx].y = cz.y;
    _Z[idx].z = cz.z;
}



__global__ void _schwarzLocalXSym3(const __GEIGEN__::GPUMas32MatrixSymf* Pred,
                                   const Eigen::Vector3f*              mR,
                                   Precision_T3*                    mZ,
                                   int                              number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int hessianSize = (GPU_MAS_BANKSIZE * 3) * (GPU_MAS_BANKSIZE);

    int Hid  = idx / hessianSize;
    int MRid = (idx % hessianSize) / (GPU_MAS_BANKSIZE);
    int MCid = (idx % hessianSize) % (GPU_MAS_BANKSIZE);

    int vrid = Hid * GPU_MAS_BANKSIZE + MRid / 3;
    int vcid = Hid * GPU_MAS_BANKSIZE + MCid;

    int r3id = MRid % 3;

    int    lvrid = vrid % GPU_MAS_BANKSIZE;
    int    lvcid = vcid % GPU_MAS_BANKSIZE;
    FloatP rdata = 0;

    __shared__ Eigen::Vector3f smR[GPU_MAS_BANKSIZE];

    if(threadIdx.x < GPU_MAS_BANKSIZE)
    {
        smR[threadIdx.x] = mR[vcid];
    }
    __syncthreads();

    if(lvcid >= lvrid)
    {
        int index = GPU_MAS_BANKSIZE * lvrid - lvrid * (lvrid + 1) / 2 + lvcid;
        rdata     = Pred[Hid].M[index](r3id, 0) * smR[lvcid][0]
                + Pred[Hid].M[index](r3id, 1) * smR[lvcid][1]
                + Pred[Hid].M[index](r3id, 2) * smR[lvcid][2];
    }
    else
    {
        int index = GPU_MAS_BANKSIZE * lvcid - lvcid * (lvcid + 1) / 2 + lvrid;
        rdata     = Pred[Hid].M[index](0, r3id) * smR[lvcid][0]
                + Pred[Hid].M[index](1, r3id) * smR[lvcid][1]
                + Pred[Hid].M[index](2, r3id) * smR[lvcid][2];
    }
    //__syncthreads();
    int  warpId    = threadIdx.x & 0x1f;
    int  landidx   = threadIdx.x % GPU_MAS_BANKSIZE;
    bool bBoundary = (landidx == 0) || (warpId == 0);

    unsigned int mark     = __ballot_sync(0xffffffff, bBoundary);  // a bit-mask
    mark                  = __brev(mark);
    int          clzlen   = __clz(mark << (warpId + 1));
    unsigned int interval = std::min(clzlen, 31 - warpId);

    int maxSize = std::min(32, GPU_MAS_BANKSIZE);
    for(int iter = 1; iter < maxSize; iter <<= 1)
    {
        FloatP tmpx = __shfl_down_sync(0xffffffff, rdata, iter);
        if(interval >= iter)
        {

            rdata += tmpx;
        }
    }

    if(bBoundary)
    {
        atomicAdd((&(mZ[vrid].x) + MRid % 3), rdata);
    }
}


__global__ void _schwarzLocalXSym6(const __GEIGEN__::GPUMas32MatrixSymf* Pred,
                                   const Eigen::Vector3f*           mR,
                                   Precision_T3*                    mZ,
                                   int                              number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int hessianSize = (GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE);

    int Hid   = idx / hessianSize;
    int lvrid = (idx % hessianSize) / (GPU_MAS_BANKSIZE);
    int lvcid = (idx % hessianSize) % (GPU_MAS_BANKSIZE);

    int vrid = Hid * GPU_MAS_BANKSIZE + lvrid;
    int vcid = Hid * GPU_MAS_BANKSIZE + lvcid;

    Eigen::Vector3f rdata;
    //rdata.setZero();

    __shared__ Eigen::Vector3f smR[GPU_MAS_BANKSIZE];

    if(threadIdx.x < GPU_MAS_BANKSIZE)
    {
        smR[threadIdx.x] = mR[vcid];
    }
    __syncthreads();

    if(vcid >= vrid)
    {
        int index = GPU_MAS_BANKSIZE * lvrid - lvrid * (lvrid + 1) / 2 + lvcid;
        rdata     = Pred[Hid].M[index] * smR[lvcid];
    }
    else
    {
        int index = GPU_MAS_BANKSIZE * lvcid - lvcid * (lvcid + 1) / 2 + lvrid;
        rdata     = Pred[Hid].M[index].transpose() * smR[lvcid];
    }
    //__syncthreads();
    int  warpId    = threadIdx.x & 0x1f;
    int  landidx   = threadIdx.x % GPU_MAS_BANKSIZE;
    bool bBoundary = (landidx == 0) || (warpId == 0);

    unsigned int mark     = __ballot_sync(0xffffffff, bBoundary);  // a bit-mask
    mark                  = __brev(mark);
    int          clzlen   = __clz(mark << (warpId + 1));
    unsigned int interval = std::min(clzlen, 31 - warpId);

    int maxSize = std::min(32, GPU_MAS_BANKSIZE);
    for(int iter = 1; iter < maxSize; iter <<= 1)
    {
        FloatP tmpx = __shfl_down_sync(0xffffffff, rdata[0], iter);
        FloatP tmpy = __shfl_down_sync(0xffffffff, rdata[1], iter);
        FloatP tmpz = __shfl_down_sync(0xffffffff, rdata[2], iter);
        if(interval >= iter)
        {

            rdata[0] += tmpx;
            rdata[1] += tmpy;
            rdata[2] += tmpz;
        }
    }

    if(bBoundary)
    {
        atomicAdd((&(mZ[vrid].x)), rdata[0]);
        atomicAdd((&(mZ[vrid].y)), rdata[1]);
        atomicAdd((&(mZ[vrid].z)), rdata[2]);
    }
}


__device__ void get_index(int& row, int& col, const int& hash, const int& size)
{
    //row = 0;
    for(row = 0; row < size; row++)
    {
        col = hash - size * row + row * (row + 1) / 2;
        if(col >= 0 && col < size)
        {
            if(size * row - row * (row + 1) / 2 + col == hash)
                return;
        }
    }
}


__global__ void _schwarzLocalXSym9(const __GEIGEN__::GPUMas32MatrixSymf* Pred,
                                   const Eigen::Vector3f*           mR,
                                   Precision_T3*                    mZ,
                                   int                              number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;

    int hessianSize = (GPU_MAS_BANKSIZE * (1 + GPU_MAS_BANKSIZE)) / 2;

    int Hid   = idx / hessianSize;
    int index = (idx % hessianSize);
    int lvrid, lvcid;
    get_index(lvrid, lvcid, index, GPU_MAS_BANKSIZE);

    int vrid = Hid * GPU_MAS_BANKSIZE + lvrid;
    int vcid = Hid * GPU_MAS_BANKSIZE + lvcid;

    __shared__ int row_ids[GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE];
    row_ids[threadIdx.x] = vrid;

    __syncthreads();
    int prev_i = -1;
    if(threadIdx.x > 0)
    {
        prev_i = row_ids[threadIdx.x - 1];
    }

    auto block_value = Pred[Hid].M[index];
    Eigen::Vector3f rdata = block_value * mR[vcid];

    if(vrid != vcid)  // process lower triangle
    {
        Eigen::Vector3f vec_ =
            block_value.transpose() * mR[vrid];

        atomicAdd((&(mZ[vcid].x)), vec_[0]);
        atomicAdd((&(mZ[vcid].y)), vec_[1]);
        atomicAdd((&(mZ[vcid].z)), vec_[2]);
    }


    int warpId = threadIdx.x & 0x1f;
    //int lane_id = threadIdx.x % GPU_MAS_BANKSIZE;

    bool bBoundary = (warpId == 0) || (prev_i != vrid);
    auto mask_val  = __activemask();

    unsigned int mark     = __ballot_sync(mask_val, bBoundary);  // a bit-mask
    mark                  = __brev(mark);
    int          clzlen   = __clz(mark << (warpId + 1));
    unsigned int interval = std::min(clzlen, 31 - warpId);

    mark = interval;
    for(int iter = 1; iter & 0x1f; iter <<= 1)
    {
        int tmp = __shfl_down_sync(__activemask(), mark, iter);
        if(tmp > mark)
            mark = tmp;
    }
    int maxSize = __shfl_sync(mask_val, mark, 0);
    //__syncthreads();

    for(int iter = 1; iter < maxSize; iter <<= 1)
    {
        float tmpx = __shfl_down_sync(mask_val, rdata[0], iter);
        float tmpy = __shfl_down_sync(mask_val, rdata[1], iter);
        float tmpz = __shfl_down_sync(mask_val, rdata[2], iter);
        if(interval >= iter)
        {

            rdata[0] += tmpx;
            rdata[1] += tmpy;
            rdata[2] += tmpz;
        }
    }

    if(bBoundary)
    {
        atomicAdd((&(mZ[vrid].x)), rdata[0]);
        atomicAdd((&(mZ[vrid].y)), rdata[1]);
        atomicAdd((&(mZ[vrid].z)), rdata[2]);
    }
}


__global__ void _buildCollisionConnection_new(unsigned int* _pConnect,
                                              const int*    _pCoarseSpaceTable,
                                              const const int4* _collisionPair,
                                              const int* _real_map_partId,
                                              int        level,
                                              int        node_offset,
                                              int        vertNum,
                                              int        number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= number)
        return;
    int4 MMCVIDI              = _collisionPair[idx];
    int* collitionPairStartId = &(MMCVIDI.x);
    if(MMCVIDI.x >= 0)
    {
        if(MMCVIDI.w < 0)
        {
            MMCVIDI.w = -MMCVIDI.w - 1;
        }

        for(int i = 0; i < 4; i++)
            collitionPairStartId[i] -= node_offset;

        int cpVertNum = 4;
        int cpVid[4];
        if(_pCoarseSpaceTable)
        {
            for(int i = 0; i < 4; i++)
                if(collitionPairStartId[i] >= 0)
                    cpVid[i] =
                        _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                else
                    cpVid[i] = -1;
        }
        else
        {
            for(int i = 0; i < 4; i++)
                if(collitionPairStartId[i] >= 0)
                    cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                else
                    cpVid[i] = -1;
        }

        unsigned int connMsk[4] = {0};

        for(int i = 0; i < 4; i++)
        {
            for(int j = i + 1; j < 4; j++)
            {
                unsigned int myId = cpVid[i];
                unsigned int otId = cpVid[j];

                if(myId == otId || myId < 0 || otId < 0)
                {
                    continue;
                }
                if(myId / GPU_MAS_BANKSIZE == otId / GPU_MAS_BANKSIZE)
                {
                    connMsk[i] |= (1U << (otId % GPU_MAS_BANKSIZE));
                    connMsk[j] |= (1U << (myId % GPU_MAS_BANKSIZE));
                }
            }
        }
        if(_pCoarseSpaceTable)
        {
            for(int i = 0; i < 4; i++)
                if(cpVid[i] >= 0)
                {
                    atomicOr(_pConnect + cpVid[i], connMsk[i]);
                }
        }
        else
        {
            for(int i = 0; i < 4; i++)
                if(collitionPairStartId[i] >= 0)
                {
                    atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                }
        }
    }
    else
    {
        int v0I   = -MMCVIDI.x - 1;
        MMCVIDI.x = v0I;
        if(MMCVIDI.z < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.z = -MMCVIDI.z - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;

                for(int i = 0; i < 4; i++)
                    collitionPairStartId[i] -= node_offset;

                int cpVertNum = 4;
                int cpVid[4];
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] =
                                _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                        else
                            cpVid[i] = -1;
                }
                else
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                        else
                            cpVid[i] = -1;
                }

                unsigned int connMsk[4] = {0};

                for(int i = 0; i < 4; i++)
                {
                    for(int j = i + 1; j < 4; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId || myId < 0 || otId < 0)
                        {
                            continue;
                        }
                        if(myId / GPU_MAS_BANKSIZE == otId / GPU_MAS_BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % GPU_MAS_BANKSIZE));
                            connMsk[j] |= (1U << (myId % GPU_MAS_BANKSIZE));
                        }
                    }
                }

                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 4; i++)
                        if(cpVid[i] >= 0)
                        {
                            atomicOr(_pConnect + cpVid[i], connMsk[i]);
                        }
                }
                else
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                        {
                            atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                        }
                }
            }
            else
            {
                int cpVertNum = 2;
                int cpVid[2];

                for(int i = 0; i < 2; i++)
                    collitionPairStartId[i] -= node_offset;
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 2; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] =
                                _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                        else
                            cpVid[i] = -1;
                }
                else
                {
                    for(int i = 0; i < 2; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                        else
                            cpVid[i] = -1;
                }

                unsigned int connMsk[2] = {0};

                for(int i = 0; i < 2; i++)
                {
                    for(int j = i + 1; j < 2; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId || myId < 0 || otId < 0)
                        {
                            continue;
                        }
                        if(myId / GPU_MAS_BANKSIZE == otId / GPU_MAS_BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % GPU_MAS_BANKSIZE));
                            connMsk[j] |= (1U << (myId % GPU_MAS_BANKSIZE));
                        }
                    }
                }

                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 2; i++)
                        if(cpVid[i] >= 0)
                        {
                            atomicOr(_pConnect + cpVid[i], connMsk[i]);
                        }
                }
                else
                {
                    for(int i = 0; i < 2; i++)
                        if(collitionPairStartId[i] >= 0)
                        {
                            atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                        }
                }
            }
        }
        else if(MMCVIDI.w < 0)
        {
            if(MMCVIDI.y < 0)
            {
                MMCVIDI.y = -MMCVIDI.y - 1;
                MMCVIDI.w = -MMCVIDI.w - 1;
                for(int i = 0; i < 4; i++)
                    collitionPairStartId[i] -= node_offset;
                int cpVertNum = 4;
                int cpVid[4];
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] =
                                _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                        else
                            cpVid[i] = -1;
                }
                else
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                        else
                            cpVid[i] = -1;
                }

                unsigned int connMsk[4] = {0};

                for(int i = 0; i < 4; i++)
                {
                    for(int j = i + 1; j < 4; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId || myId < 0 || otId < 0)
                        {
                            continue;
                        }
                        if(myId / GPU_MAS_BANKSIZE == otId / GPU_MAS_BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % GPU_MAS_BANKSIZE));
                            connMsk[j] |= (1U << (myId % GPU_MAS_BANKSIZE));
                        }
                    }
                }

                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 4; i++)
                        if(cpVid[i] >= 0)
                        {
                            atomicOr(_pConnect + cpVid[i], connMsk[i]);
                        }
                }
                else
                {
                    for(int i = 0; i < 4; i++)
                        if(collitionPairStartId[i] >= 0)
                        {
                            atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                        }
                }
            }
            else
            {
                int cpVertNum = 3;
                int cpVid[3];
                for(int i = 0; i < 3; i++)
                    collitionPairStartId[i] -= node_offset;
                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 3; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] =
                                _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                        else
                            cpVid[i] = -1;
                }
                else
                {
                    for(int i = 0; i < 3; i++)
                        if(collitionPairStartId[i] >= 0)
                            cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                        else
                            cpVid[i] = -1;
                }

                unsigned int connMsk[3] = {0};

                for(int i = 0; i < 3; i++)
                {
                    for(int j = i + 1; j < 3; j++)
                    {
                        unsigned int myId = cpVid[i];
                        unsigned int otId = cpVid[j];

                        if(myId == otId || myId < 0 || otId < 0)
                        {
                            continue;
                        }
                        if(myId / GPU_MAS_BANKSIZE == otId / GPU_MAS_BANKSIZE)
                        {
                            connMsk[i] |= (1U << (otId % GPU_MAS_BANKSIZE));
                            connMsk[j] |= (1U << (myId % GPU_MAS_BANKSIZE));
                        }
                    }
                }

                if(_pCoarseSpaceTable)
                {
                    for(int i = 0; i < 3; i++)
                        if(cpVid[i] >= 0)
                        {
                            atomicOr(_pConnect + cpVid[i], connMsk[i]);
                        }
                }
                else
                {
                    for(int i = 0; i < 3; i++)
                        if(collitionPairStartId[i] >= 0)
                        {
                            atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                        }
                }
            }
        }
        else
        {
            int cpVertNum = 4;
            int cpVid[4];
            for(int i = 0; i < 4; i++)
                collitionPairStartId[i] -= node_offset;
            if(_pCoarseSpaceTable)
            {
                for(int i = 0; i < 4; i++)
                    if(collitionPairStartId[i] >= 0)
                        cpVid[i] =
                            _pCoarseSpaceTable[collitionPairStartId[i] + (level - 1) * vertNum];
                    else
                        cpVid[i] = -1;
            }
            else
            {
                for(int i = 0; i < 4; i++)
                    if(collitionPairStartId[i] >= 0)
                        cpVid[i] = _real_map_partId[collitionPairStartId[i]];
                    else
                        cpVid[i] = -1;
            }

            unsigned int connMsk[4] = {0};

            for(int i = 0; i < 4; i++)
            {
                for(int j = i + 1; j < 4; j++)
                {
                    unsigned int myId = cpVid[i];
                    unsigned int otId = cpVid[j];

                    if(myId == otId || myId < 0 || otId < 0)
                    {
                        continue;
                    }
                    if(myId / GPU_MAS_BANKSIZE == otId / GPU_MAS_BANKSIZE)
                    {
                        connMsk[i] |= (1U << (otId % GPU_MAS_BANKSIZE));
                        connMsk[j] |= (1U << (myId % GPU_MAS_BANKSIZE));
                    }
                }
            }

            if(_pCoarseSpaceTable)
            {
                for(int i = 0; i < 4; i++)
                    if(cpVid[i] >= 0)
                    {
                        atomicOr(_pConnect + cpVid[i], connMsk[i]);
                    }
            }
            else
            {
                for(int i = 0; i < 4; i++)
                    if(collitionPairStartId[i] >= 0)
                    {
                        atomicOr(_pConnect + collitionPairStartId[i], connMsk[i]);
                    }
            }
        }
    }
}


void TraditionalMAS32Preconditioner::BuildConnectMaskL0()
{

    //int number = totalNodes;
#ifdef GROUP
    int number    = totalMapNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    _buildCML0_new<<<numBlocks, blockSize>>>(d_neighborStart,
                                             d_neighborNum,
                                             d_neighborList,
                                             d_fineConnectMask,
                                             d_partId_map_real,
                                             d_real_map_partId,
                                             number);
#else
    int number    = totalNodes;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    _buildCML0<<<numBlocks, blockSize>>>(
        d_neighborStart, d_neighborNum, d_neighborList, d_fineConnectMask, number);
#endif
}

void TraditionalMAS32Preconditioner::PreparePrefixSumL0()
{
    //int number = totalNodes;
#ifdef GROUP
    int number    = totalMapNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    _preparePrefixSumL0_new<<<numBlocks, blockSize>>>(
        d_prefixOriginal, d_fineConnectMask, d_partId_map_real, number);
#else
    int number    = totalNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    _preparePrefixSumL0<<<numBlocks, blockSize>>>(d_prefixOriginal, d_fineConnectMask, number);
#endif
}

void TraditionalMAS32Preconditioner::BuildLevel1()
{
    //int number = totalNodes;
#ifdef GROUP
    int number    = totalMapNodes;
    if(number < 1)
        return;
    int blockSize = GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    //exclusive(d_prefixOriginal, d_prefixSumOriginal); wait to do;
    int warpNum = (number + GPU_MAS_BANKSIZE - 1) / GPU_MAS_BANKSIZE;
    thrust::exclusive_scan(thrust::device_ptr<int>(d_prefixOriginal),
                           thrust::device_ptr<int>(d_prefixOriginal) + warpNum,
                           thrust::device_ptr<int>(d_prefixSumOriginal));
    _buildLevel1_new<<<numBlocks, blockSize>>>(d_levelSize,
                                               d_coarseSpaceTables,
                                               d_goingNext,
                                               d_fineConnectMask,
                                               d_prefixSumOriginal,
                                               d_prefixOriginal,
                                               d_partId_map_real,
                                               number);
#else
    int number    = totalNodes;
    int blockSize = GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    //exclusive(d_prefixOriginal, d_prefixSumOriginal); wait to do;
    int warpNum = (number + GPU_MAS_BANKSIZE - 1) / GPU_MAS_BANKSIZE;
    thrust::exclusive_scan(thrust::device_ptr<int>(d_prefixOriginal),
                           thrust::device_ptr<int>(d_prefixOriginal) + warpNum,
                           thrust::device_ptr<int>(d_prefixSumOriginal));
    _buildLevel1<<<numBlocks, blockSize>>>(d_levelSize,
                                           d_coarseSpaceTables,
                                           d_goingNext,
                                           d_fineConnectMask,
                                           d_prefixSumOriginal,
                                           d_prefixOriginal,
                                           number);
#endif
}

void TraditionalMAS32Preconditioner::BuildConnectMaskLx(int level)
{
    //int number = totalNodes;
#ifdef GROUP
    int number    = totalMapNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    _buildConnectMaskLx_new<<<numBlocks, blockSize>>>(d_neighborStart,
                                                      d_neighborNum,
                                                      d_neighborList,
                                                      d_coarseSpaceTables,
                                                      d_nextConnectMask,
                                                      d_fineConnectMask,
                                                      level,
                                                      d_partId_map_real,
                                                      totalNodes,
                                                      number);
#else
    int number    = totalNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    _buildConnectMaskLx<<<numBlocks, blockSize>>>(d_neighborStart,
                                                  d_neighborNum,
                                                  d_neighborList,
                                                  d_coarseSpaceTables,
                                                  d_nextConnectMask,
                                                  d_fineConnectMask,
                                                  level,
                                                  number);
#endif
}

void TraditionalMAS32Preconditioner::NextLevelCluster(int level)
{
    int number    = h_clevelSize.x;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    _nextLevelCluster<<<numBlocks, blockSize>>>(d_nextConnectMask, d_nextPrefix, number);
}

void TraditionalMAS32Preconditioner::ComputeNextLevel(int level)
{
    int number    = totalNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    _computeNextLevel<<<numBlocks, blockSize>>>(
        d_coarseSpaceTables, d_nextConnectMask, level, number);
}

void TraditionalMAS32Preconditioner::PrefixSumLx(int level)
{
    int number     = h_clevelSize.x;
    if(number < 1)
        return;
    int levelBegin = h_clevelSize.y;
    int blockSize  = GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE;
    int numBlocks  = (number + blockSize - 1) / blockSize;

    int warpNum = (number + GPU_MAS_BANKSIZE - 1) / GPU_MAS_BANKSIZE;
    thrust::exclusive_scan(thrust::device_ptr<unsigned int>(d_nextPrefix),
                           thrust::device_ptr<unsigned int>(d_nextPrefix) + warpNum,
                           thrust::device_ptr<unsigned int>(d_nextPrefixSum));

    _prefixSumLx<<<numBlocks, blockSize>>>(
        d_levelSize, d_nextPrefix, d_nextPrefixSum, d_nextConnectMask, d_goingNext, level, levelBegin, number);
}

void TraditionalMAS32Preconditioner::AggregationKernel()
{
    int number    = totalNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    _aggregationKernel<<<numBlocks, blockSize>>>(
        d_denseLevel, d_coarseTable, d_goingNext, levelnum, number);
}


void TraditionalMAS32Preconditioner::computeNumLevels(int vertNum)
{
    int totalSz = 0;
    int nLevel  = 1;
    int levelSz = (vertNum + GPU_MAS_BANKSIZE - 1) / GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE;
    totalSz += levelSz;

    while(levelSz > GPU_MAS_BANKSIZE)
    {
        levelSz /= GPU_MAS_BANKSIZE;

        nLevel++;
        levelSz = (levelSz + GPU_MAS_BANKSIZE - 1) / GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE;
        totalSz += levelSz;
    }
    nLevel   = nLevel + 1;
    levelnum = nLevel > 6 ? 6 : nLevel;
    printf("level num:  %d\n", levelnum);
    //totalSize = totalSz * SizeRatio;
}

void TraditionalMAS32Preconditioner::BuildCollisionConnection(unsigned int* connectionMsk,
                                                 int*          coarseTableSpace,
                                                 int           level,
                                                 int           cpNum)
{
    int number    = cpNum;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
#ifdef GROUP
    _buildCollisionConnection_new<<<numBlocks, blockSize>>>(connectionMsk,
                                                            coarseTableSpace,
                                                            _collisonPairs,
                                                            d_real_map_partId,
                                                            level,
                                                            collision_node_Offset,
                                                            totalNodes,
                                                            number);

#else
    _buildCollisionConnection<<<numBlocks, blockSize>>>(
        connectionMsk, coarseTableSpace, _collisonPairs, level, collision_node_Offset, totalNodes, number);

#endif
}
int TraditionalMAS32Preconditioner::ReorderRealtime(int cpNum)
{
    CUDA_SAFE_CALL(cudaMemset(d_levelSize, 0, levelnum * sizeof(int2)));


    BuildConnectMaskL0();
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    if(cpNum)
        BuildCollisionConnection(d_fineConnectMask, nullptr, -1, cpNum);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    PreparePrefixSumL0();

    BuildLevel1();
    for(int level = 1; level < levelnum; level++)
    {
        CUDA_SAFE_CALL(cudaMemset(d_nextConnectMask, 0, totalNodes * sizeof(int)));

        BuildConnectMaskLx(level);
        //CUDA_SAFE_CALL(cudaDeviceSynchronize());
        if(cpNum)
            BuildCollisionConnection(d_nextConnectMask, d_coarseSpaceTables, level, cpNum);

        CUDA_SAFE_CALL(cudaMemcpy(&h_clevelSize, d_levelSize + level, sizeof(int2), cudaMemcpyDeviceToHost));

        NextLevelCluster(level);



        PrefixSumLx(level);

        ComputeNextLevel(level);

    }

    CUDA_SAFE_CALL(cudaMemcpy(&h_clevelSize, d_levelSize + levelnum, sizeof(int2), cudaMemcpyDeviceToHost));

    totalNumberClusters = h_clevelSize.y;

    AggregationKernel();

    return totalNumberClusters;
}

namespace
{
__global__ void prepare_hessian_bcoo_kernel(int                   tripletNum,
                                            int                   offset,
                                            int                   levelNum,
                                            int*                  _goingNext,
                                            __GEIGEN__::GPUMas32MatrixSymT* _invMatrix,
                                            int*                  _real_map_partId,
                                            uint32_t*             indices,
                                            Eigen::Matrix3d*      triplet_values,
                                            int*                  row_ids,
                                            int*                  col_ids)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if(I >= tripletNum)
        return;
    int index                              = indices[I];
    auto vertRid_real                      = row_ids[index];
    auto vertCid_real                       = col_ids[index];
    auto H = triplet_values[index];
    vertRid_real -= offset;
    vertCid_real -= offset;
    int vertCid = _real_map_partId[vertCid_real];
    int vertRid = _real_map_partId[vertRid_real];
    int cPid    = vertCid / GPU_MAS_BANKSIZE;

    if(vertCid / GPU_MAS_BANKSIZE == vertRid / GPU_MAS_BANKSIZE)
    {
        if(vertCid >= vertRid)
        {
            int bvRid = vertRid % GPU_MAS_BANKSIZE;
            int bvCid = vertCid % GPU_MAS_BANKSIZE;
            int index = GPU_MAS_BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;

            _invMatrix[cPid].M[index] = H;
        }
        else
        {
            // The global BCOO is upper triangular in the original vertex
            // numbering. Morton reordering can reverse that order inside a
            // MAS32 bank, so store the transposed block in the compressed
            // upper triangle instead of silently dropping it.
            int bvRid = vertRid % GPU_MAS_BANKSIZE;
            int bvCid = vertCid % GPU_MAS_BANKSIZE;
            int index = GPU_MAS_BANKSIZE * bvCid
                        - bvCid * (bvCid + 1) / 2 + bvRid;
            _invMatrix[cPid].M[index] = H.transpose();
        }
    }
    else
    {
        int level = 0;
        while(level < levelNum - 1)
        {
            level++;
            if(level == 1)
            {
                vertCid = _goingNext[vertCid_real];
                vertRid = _goingNext[vertRid_real];
            }
            else
            {
                vertCid = _goingNext[vertCid];
                vertRid = _goingNext[vertRid];
            }
            cPid = vertCid / GPU_MAS_BANKSIZE;
            if(vertCid / GPU_MAS_BANKSIZE == vertRid / GPU_MAS_BANKSIZE)
            {
                if(vertCid >= vertRid)
                {
                    int bvRid = vertRid % GPU_MAS_BANKSIZE;
                    int bvCid = vertCid % GPU_MAS_BANKSIZE;
                    int index = GPU_MAS_BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;
                    for(int i = 0; i < 3; i++)
                    {
                        for(int j = 0; j < 3; j++)
                        {
                            atomicAdd(
                                &(_invMatrix[cPid].M[index](i, j)),
                                H(i, j));
                            if(vertCid == vertRid)
                            {
                                atomicAdd(
                                    &(_invMatrix[cPid].M[index](i, j)),
                                    H(j, i));
                            }
                        }
                    }
                }
                else
                {
                    int bvRid = vertRid % GPU_MAS_BANKSIZE;
                    int bvCid = vertCid % GPU_MAS_BANKSIZE;
                    int index = GPU_MAS_BANKSIZE * bvCid - bvCid * (bvCid + 1) / 2 + bvRid;
                    for(int i = 0; i < 3; i++)
                    {
                        for(int j = 0; j < 3; j++)
                        {
                            atomicAdd(&(_invMatrix[cPid].M[index](i, j)),
                                      H(j, i));
                        }
                    }
                }
            }
        }
    }
}

__global__ void prepare_hessian_bcoo_sum_kernel(int                   tripletNum,
                                                int                   levelNum,
                                                int*                  _goingNext,
                                                __GEIGEN__::GPUMas32MatrixSymT* _invMatrix,
                                                int*                  _partId_map_real,
                                                unsigned int*         _fineConnectMsk,
                                                int*                  _prefix0)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= tripletNum)
        return;
    int HSIZE = (GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE);
    int Hid   = idx / HSIZE;
    int LMRid = (idx % HSIZE) / GPU_MAS_BANKSIZE;
    int LMCid = (idx % HSIZE) % GPU_MAS_BANKSIZE;

    int MRid = Hid * GPU_MAS_BANKSIZE + LMRid;
    int MCid = Hid * GPU_MAS_BANKSIZE + LMCid;

    int            rdx = _partId_map_real[MRid];
    int            cdx = _partId_map_real[MCid];
    __shared__ int prefix;

    if(threadIdx.x == 0)
    {
        prefix = _prefix0[Hid];
    }
    __syncthreads();
    Eigen::Matrix3d mat3;
    if(LMCid >= LMRid)
    {
        int index = GPU_MAS_BANKSIZE * LMRid - LMRid * (LMRid + 1) / 2 + LMCid;
        mat3 = _invMatrix[Hid].M[index];
    }
    else
    {
        int index = GPU_MAS_BANKSIZE * LMCid - LMCid * (LMCid + 1) / 2 + LMRid;
        mat3 = _invMatrix[Hid].M[index].transpose();
    }

    const bool         mapped      = (rdx >= 0) && (cdx >= 0);
    const unsigned int active_mask = __ballot_sync(__activemask(), mapped);

    if(mapped)
    {
        if(prefix == 1)
        {
            int warpId = threadIdx.x & 0x1f;
            bool bBoundary = (warpId == 0);
            unsigned int mark = __ballot_sync(active_mask, bBoundary);
            mark = __brev(mark);
            int clzlen = __clz(mark << (warpId + 1));
            unsigned int interval = std::min(clzlen, 31 - warpId);
            for(int iter = 1; iter < 32; iter <<= 1)
            {
                Eigen::Matrix3d matTemp;
                for(int i = 0; i < 3; i++)
                {
                    for(int j = 0; j < 3; j++)
                    {
                        matTemp(i, j) =
                            __shfl_down_sync(active_mask, mat3(i, j), iter);
                    }
                }
                if(interval >= iter)
                {
                    mat3 = mat3 + matTemp;
                }
            }
            __syncwarp(active_mask);
            int level = 0;
            if(bBoundary)
            {
                int nextId = _goingNext[rdx];
                while(level < levelNum - 1)
                {
                    level++;
                    int cPid  = nextId / GPU_MAS_BANKSIZE;
                    int bvRid = nextId % GPU_MAS_BANKSIZE;
                    int bvCid = nextId % GPU_MAS_BANKSIZE;
                    int index = GPU_MAS_BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;
                    for(int i = 0; i < 3; i++)
                    {
                        for(int j = 0; j < 3; j++)
                        {
                            atomicAdd(
                                &(_invMatrix[cPid].M[index](i, j)),
                                mat3(i, j));
                        }
                    }
                    nextId = _goingNext[nextId];
                }
            }
        }
        else
        {
            int level = 0;
            while(level < levelNum - 1)
            {
                level++;
                rdx      = _goingNext[rdx];
                cdx      = _goingNext[cdx];
                int cPid = cdx / GPU_MAS_BANKSIZE;
                if(rdx / GPU_MAS_BANKSIZE == cdx / GPU_MAS_BANKSIZE)
                {
                    if(cdx >= rdx)
                    {
                        int bvRid = rdx % GPU_MAS_BANKSIZE;
                        int bvCid = cdx % GPU_MAS_BANKSIZE;
                        int index = GPU_MAS_BANKSIZE * bvRid - bvRid * (bvRid + 1) / 2 + bvCid;

                        for(int i = 0; i < 3; i++)
                        {
                            for(int j = 0; j < 3; j++)
                            {
                                atomicAdd(&(_invMatrix[cPid].M[index](i, j)),
                                          mat3(i, j));
                            }
                        }
                    }
                }
            }
        }
    }
}
}  // namespace

void TraditionalMAS32Preconditioner::PrepareHessian_bcoo(Eigen::Matrix3d* triplet_values,
                                            int*             row_ids,
                                            int*             col_ids,
                                            uint32_t*        indices,
                                            int              offset,
                                            int              triplet_number)
{
    //cudaEvent_t start, end0, end1, end2;
    //cudaEventCreate(&start);
    //cudaEventCreate(&end0);
    //cudaEventCreate(&end1);

    //cudaEventRecord(start);



    using namespace cudatool;
    int tripletNum = triplet_number;
    if(true)
    {
        LaunchCudaKernal_default(
            tripletNum,
            256,
            0,
            prepare_hessian_bcoo_kernel,
            tripletNum,
            offset,
            levelnum,
            d_goingNext,
            d_inverseMatMas,
            d_real_map_partId,
            indices,
            triplet_values,
            row_ids,
            col_ids);

        tripletNum    = totalMapNodes * GPU_MAS_BANKSIZE;
        // The 32x32 logical matrix is reduced independently by row-sized
        // warps.  Launching all 1024 logical entries in one block requires
        // more registers than sm_86 provides (82 * 1024 > 65536).  Splitting
        // it into four 256-thread blocks preserves the warp-local algorithm
        // while remaining launchable on RTX 30-series GPUs.
        int threadNum = 256;
        int blockNum  = (tripletNum + threadNum - 1) / threadNum;

        LaunchCudaKernalNamed(
            "TraditionalMAS32::prepare_hessian_bcoo_sum_kernel",
            blockNum,
            threadNum,
            0,
            prepare_hessian_bcoo_sum_kernel,
            tripletNum,
            levelnum,
            d_goingNext,
            d_inverseMatMas,
            d_partId_map_real,
            d_fineConnectMask,
            d_prefixOriginal);
    }


    //cudaEventRecord(end0);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    int blockSize2 = 32 * 3;
    //int number2    = totalNumberClusters / GPU_MAS_BANKSIZE;
    int number2    = totalNumberClusters * 3;
    if(number2 < 1)
        return;
    int numBlocks2 = (number2 + blockSize2 - 1) / blockSize2;

    expand_symmetric_mas32<<<numBlocks2, blockSize2>>>(
        d_MatMas, d_inverseMatMas, number2);
    if(factorized_local_solve)
    {
        CUDA_SAFE_CALL(cudaMemset(d_factor_status, 0, 2 * sizeof(int)));
        factor_mas32_cholesky<<<numBlocks2, 128>>>(d_MatMas, numBlocks2,
                                                   d_factor_status);
        invert_mas32_cholesky<<<numBlocks2, 96>>>(d_MatMas,d_inverseMatMas,
                                                  numBlocks2,d_factor_status);
    }
    else
        __inverse6_P96x96<<<numBlocks2, blockSize2>>>(d_precondMatMas, d_MatMas, number2);

    //cudaEventRecord(end1);

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    //float time0, time1, time2, time3, time4;
    //cudaEventElapsedTime(&time0, start, end0);
    //cudaEventElapsedTime(&time1, end0, end1);
    ////cudaEventElapsedTime(&time2, end1, end2);

    //printf("\n\ntime0 = %f,  time1 = %f\n\n", time0, time1);

    //(cudaEventDestroy(start));
    //(cudaEventDestroy(end0));
    //(cudaEventDestroy(end1));
    //(cudaEventDestroy(end2));
}


void TraditionalMAS32Preconditioner::BuildMultiLevelR(const double3* R)
{


#ifdef GROUP
    int number = totalMapNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    __buildMultiLevelR_optimized_new<<<numBlocks, blockSize>>>(
        R, d_multiLevelR, d_goingNext, d_prefixOriginal, d_fineConnectMask, d_partId_map_real, levelnum, number);

#else
    int number = totalNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
    __buildMultiLevelR_optimized<<<numBlocks, blockSize>>>(
        R, d_multiLevelR, d_goingNext, d_fineConnectMask, levelnum, number);
#endif
}

void TraditionalMAS32Preconditioner::SchwarzLocalXSym()
{
    //int matNum    = totalNumberClusters / GPU_MAS_BANKSIZE;
    int number    = totalNumberClusters * GPU_MAS_BANKSIZE * 3;
    if(number < 1)
        return;
    int blockSize = GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    //_schwarzLocalXSym1<<<numBlocks, blockSize>>>(d_MatMas, d_multiLevelR, d_multiLevelZ, number);
    _schwarzLocalXSym3<<<numBlocks, blockSize>>>(
        d_precondMatMas, d_multiLevelR, d_multiLevelZ, number);
}

void TraditionalMAS32Preconditioner::SchwarzLocalXSym_block3()
{
    //int matNum    = totalNumberClusters / GPU_MAS_BANKSIZE;
    int number = totalNumberClusters * GPU_MAS_BANKSIZE;
    if(number < 1)
        return;
    int blockSize = GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    //_schwarzLocalXSym1<<<numBlocks, blockSize>>>(d_MatMas, d_multiLevelR, d_multiLevelZ, number);
    if(factorized_local_solve)
        apply_mas32_inverse_fp64<<<totalNumberClusters / GPU_MAS_BANKSIZE, 96>>>(
            d_inverseMatMas, d_multiLevelR, d_multiLevelZ,
            totalNumberClusters / GPU_MAS_BANKSIZE);
    else
        _schwarzLocalXSym6<<<numBlocks, blockSize>>>(
            d_precondMatMas, d_multiLevelR, d_multiLevelZ, number);
}

void TraditionalMAS32Preconditioner::SchwarzLocalXSym_sym()
{
    int matNum    = totalNumberClusters / GPU_MAS_BANKSIZE;
    int number = matNum * (1 + GPU_MAS_BANKSIZE) * GPU_MAS_BANKSIZE / 2;
    if(number < 1)
        return;
    int blockSize = GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;

    //_schwarzLocalXSym1<<<numBlocks, blockSize>>>(d_MatMas, d_multiLevelR, d_multiLevelZ, number);
    _schwarzLocalXSym9<<<numBlocks, blockSize>>>(
        d_precondMatMas, d_multiLevelR, d_multiLevelZ, number);
}

void TraditionalMAS32Preconditioner::CollectFinalZ(double3* Z)
{
    int number = totalNodes;
    if(number < 1)
        return;
    int blockSize = DEFAULT_BLOCKSIZE;
    int numBlocks = (number + blockSize - 1) / blockSize;
#ifdef GROUP
    __collectFinalZ_new<<<numBlocks, blockSize>>>(
        Z, d_multiLevelZ, d_coarseTable, d_real_map_partId, levelnum, number);
#else
    __collectFinalZ<<<numBlocks, blockSize>>>(Z, d_multiLevelZ, d_coarseTable, levelnum, number);
#endif

}



void TraditionalMAS32Preconditioner::setPreconditioner_bcoo(Eigen::Matrix3d* triplet_values,
                                               int*             row_ids,
                                               int*             col_ids,
                                               uint32_t*        indices,
                                               int              offset,
                                               int              triplet_num,
                                               int              cpNum)
{
    if(totalNodes < 1)
        return;
    CUDA_SAFE_CALL(cudaMemcpy(d_neighborList,
                              d_neighborListInit,
                              neighborListSize * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));
    //CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborStart, tetMesh.neighborStart.data(), ipc.vertexNum * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_neighborNum,
                              d_neighborNumInit,
                              totalNodes * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));


    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    ReorderRealtime(cpNum);

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

#ifdef SYME

    CUDA_SAFE_CALL(cudaMemset(
        d_inverseMatMas, 0, totalNumberClusters / GPU_MAS_BANKSIZE * sizeof(__GEIGEN__::GPUMas32MatrixSymT)));
#else
    CUDA_SAFE_CALL(cudaMemset(
        d_MatMas, 0, totalNumberClusters / GPU_MAS_BANKSIZE * sizeof(__GEIGEN__::GPUMas32MatrixT)));
#endif
    PrepareHessian_bcoo(triplet_values, row_ids, col_ids, indices, offset, triplet_num);

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
}


void TraditionalMAS32Preconditioner::preconditioning(const double3* R, double3* Z)
{
    if(totalNodes < 1)
        return;
    CUDA_SAFE_CALL(cudaMemset(d_multiLevelR + totalMapNodes,
                              0,
                              (totalNumberClusters - totalMapNodes) * sizeof(Eigen::Vector3f)));

    // The FP64 inverse-apply kernel owns one thread per scalar row and
    // overwrites every component of every padded hierarchy node.  Clearing
    // the output first is therefore redundant on the factorized path and
    // adds one launch plus a full-buffer write to every PCG iteration.  The
    // legacy atomic accumulation kernel still requires a zeroed destination.
    if(!factorized_local_solve)
        CUDA_SAFE_CALL(cudaMemset(d_multiLevelZ, 0,
                                  totalNumberClusters * sizeof(Precision_T3)));

    //cudaEvent_t start, end0, end1, end2;
    //cudaEventCreate(&start);
    //cudaEventCreate(&end0);
    //cudaEventCreate(&end1);
    //cudaEventCreate(&end2);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //cudaEventRecord(start);
    BuildMultiLevelR(R);
    //cudaEventRecord(end0);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    SchwarzLocalXSym_block3();
    //cudaEventRecord(end1);
    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    CollectFinalZ(Z);
    //cudaEventRecord(end2);

    //CUDA_SAFE_CALL(cudaDeviceSynchronize());

    //float time0, time1, time2, time3, time4;
    //cudaEventElapsedTime(&time0, start, end0);
    //cudaEventElapsedTime(&time1, end0, end1);
    //cudaEventElapsedTime(&time2, end1, end2);

    //printf("\n\npreconditioning  time0 = %f,  time1 = %f,  time1 = %f\n\n", time0, time1, time2);

    //(cudaEventDestroy(start));
    //(cudaEventDestroy(end0));
    //(cudaEventDestroy(end1));
    //(cudaEventDestroy(end2));
}

gipc::Json TraditionalMAS32Preconditioner::factor_diagnostics() const
{
    if(!factorized_local_solve)
        return {{"passed",false},{"failure_reason","factorized_local_solve_disabled"}};
    int status[2] = {};
    CUDA_SAFE_CALL(cudaMemcpy(status, d_factor_status, sizeof(status),
                              cudaMemcpyDeviceToHost));
    return {{"implementation","fp64_cholesky_inverse_apply"},
            {"matrix_count",totalNumberClusters / GPU_MAS_BANKSIZE},
            {"relative_pivot_floor",1.0e-12},
            {"nonfinite_factor_entries",status[0]},
            {"floored_pivots",status[1]},
            {"passed",status[0] == 0}};
}

gipc::Json TraditionalMAS32Preconditioner::numerical_diagnostics(const double3* R) const
{
    constexpr int bank_size = GPU_MAS_BANKSIZE;
    constexpr int dimension = GPU_MAS_BANKSIZE * 3;
    const int matrix_count = totalNumberClusters / GPU_MAS_BANKSIZE;

    std::vector<__GEIGEN__::GPUMas32MatrixSymT> local_matrices(matrix_count);
    std::vector<__GEIGEN__::GPUMas32MatrixSymf> inverse_matrices(matrix_count);
    std::vector<Precision_T3> multilevel_z(totalNumberClusters);
    std::vector<__GEIGEN__::itable> coarse_table(totalNodes);
    std::vector<int> real_to_part(totalNodes);
    std::vector<double3> rhs(totalNodes);
    CUDA_SAFE_CALL(cudaMemcpy(local_matrices.data(),
                              d_inverseMatMas,
                              local_matrices.size()
                                  * sizeof(__GEIGEN__::GPUMas32MatrixSymT),
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(inverse_matrices.data(),
                              d_precondMatMas,
                              inverse_matrices.size()
                                  * sizeof(__GEIGEN__::GPUMas32MatrixSymf),
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(multilevel_z.data(),
                              d_multiLevelZ,
                              multilevel_z.size() * sizeof(Precision_T3),
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(coarse_table.data(),
                              d_coarseTable,
                              coarse_table.size() * sizeof(__GEIGEN__::itable),
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(real_to_part.data(),
                              d_real_map_partId,
                              real_to_part.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(rhs.data(),
                              R,
                              rhs.size() * sizeof(double3),
                              cudaMemcpyDeviceToHost));

    int local_spd_count = 0;
    int inverse_spd_count = 0;
    int nonfinite_local_count = 0;
    int nonfinite_inverse_count = 0;
    int first_local_spd_failure = -1;
    int first_inverse_spd_failure = -1;
    double maximum_inverse_residual = 0.0;
    double minimum_local_diagonal = std::numeric_limits<double>::infinity();
    double minimum_inverse_diagonal = std::numeric_limits<double>::infinity();

    for(int matrix_id = 0; matrix_id < matrix_count; ++matrix_id)
    {
        Eigen::Matrix<double, dimension, dimension> local =
            Eigen::Matrix<double, dimension, dimension>::Zero();
        Eigen::Matrix<double, dimension, dimension> inverse =
            Eigen::Matrix<double, dimension, dimension>::Zero();
        for(int block_row = 0; block_row < bank_size; ++block_row)
        {
            for(int block_col = block_row; block_col < bank_size; ++block_col)
            {
                const int index = bank_size * block_row
                                  - block_row * (block_row + 1) / 2 + block_col;
                const Eigen::Matrix3d local_block = local_matrices[matrix_id].M[index];
                const Eigen::Matrix3d inverse_block =
                    inverse_matrices[matrix_id].M[index].cast<double>();
                local.template block<3, 3>(block_row * 3, block_col * 3) =
                    local_block;
                inverse.template block<3, 3>(block_row * 3, block_col * 3) =
                    inverse_block;
                if(block_row != block_col)
                {
                    local.template block<3, 3>(block_col * 3, block_row * 3) =
                        local_block.transpose();
                    inverse.template block<3, 3>(block_col * 3, block_row * 3) =
                        inverse_block.transpose();
                }
            }
        }
        for(int diagonal = 0; diagonal < dimension; ++diagonal)
            if(local(diagonal, diagonal) == 0.0)
                local(diagonal, diagonal) = 1.0;

        const bool local_finite = local.allFinite();
        const bool inverse_finite = inverse.allFinite();
        if(!local_finite)
            ++nonfinite_local_count;
        if(!inverse_finite)
            ++nonfinite_inverse_count;
        minimum_local_diagonal =
            std::min(minimum_local_diagonal, local.diagonal().minCoeff());
        minimum_inverse_diagonal =
            std::min(minimum_inverse_diagonal, inverse.diagonal().minCoeff());

        const bool local_spd = local_finite
                               && Eigen::LLT<decltype(local)>(local).info()
                                      == Eigen::Success;
        const bool inverse_spd = inverse_finite
                                 && Eigen::LLT<decltype(inverse)>(inverse).info()
                                        == Eigen::Success;
        if(local_spd)
            ++local_spd_count;
        else if(first_local_spd_failure < 0)
            first_local_spd_failure = matrix_id;
        if(inverse_spd)
            ++inverse_spd_count;
        else if(first_inverse_spd_failure < 0)
            first_inverse_spd_failure = matrix_id;

        if(local_finite && inverse_finite)
        {
            const auto residual = local * inverse
                                  - Eigen::Matrix<double, dimension, dimension>::Identity();
            maximum_inverse_residual =
                std::max(maximum_inverse_residual,
                         residual.norm() / std::sqrt(static_cast<double>(dimension)));
        }
    }

    std::vector<long double> level_rtz(levelnum, 0.0L);
    for(int vertex = 0; vertex < totalNodes; ++vertex)
    {
        int cluster = real_to_part[vertex];
        const auto accumulate = [&](int level, int cluster_id) {
            const auto& z = multilevel_z[cluster_id];
            level_rtz[level] += static_cast<long double>(rhs[vertex].x) * z.x
                                + static_cast<long double>(rhs[vertex].y) * z.y
                                + static_cast<long double>(rhs[vertex].z) * z.z;
        };
        accumulate(0, cluster);
        for(int level = 1; level < levelnum; ++level)
        {
            cluster = coarse_table[vertex].index[level - 1];
            accumulate(level, cluster);
        }
    }

    gipc::Json result;
    result["matrix_count"] = matrix_count;
    result["local_spd_count"] = local_spd_count;
    result["inverse_spd_count"] = inverse_spd_count;
    result["first_local_spd_failure"] = first_local_spd_failure;
    result["first_inverse_spd_failure"] = first_inverse_spd_failure;
    result["nonfinite_local_count"] = nonfinite_local_count;
    result["nonfinite_inverse_count"] = nonfinite_inverse_count;
    result["minimum_local_diagonal"] = minimum_local_diagonal;
    result["minimum_inverse_diagonal"] = minimum_inverse_diagonal;
    result["maximum_inverse_residual"] = maximum_inverse_residual;
    result["level_rtz"] = gipc::Json::array();
    long double level_sum = 0.0L;
    for(const auto value : level_rtz)
    {
        result["level_rtz"].push_back(static_cast<double>(value));
        level_sum += value;
    }
    result["level_rtz_sum"] = static_cast<double>(level_sum);
    result["passed"] = local_spd_count == matrix_count
                       && inverse_spd_count == matrix_count
                       && nonfinite_local_count == 0
                       && nonfinite_inverse_count == 0
                       && maximum_inverse_residual <= 1e-3;
    return result;
}

void TraditionalMAS32Preconditioner::initPreconditioner_Neighbor(int vertNum,
                                                    int mCollision_node_offset,
                                                    int totalNeighborNum,
                                                    int4* m_collisonPairs,
                                                    int   partMapSize)
{
    //bankSize = 32;
    if(vertNum < 1)
        return;
    int maxNodes = partMapSize > vertNum ? partMapSize : vertNum;
    computeNumLevels(maxNodes);
    totalMapNodes         = partMapSize;
    collision_node_Offset = mCollision_node_offset;
    _collisonPairs        = m_collisonPairs;
    totalNodes            = vertNum;
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_denseLevel, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_real_map_partId, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_coarseTable, vertNum * sizeof(__GEIGEN__::itable)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_coarseSpaceTables,
                              vertNum * levelnum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_levelSize, (levelnum + 1) * sizeof(int2)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_goingNext,
                              vertNum * levelnum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_prefixOriginal, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_nextPrefix, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_nextPrefixSum, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_prefixSumOriginal, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_fineConnectMask, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_nextConnectMask, vertNum * sizeof(unsigned int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborList, totalNeighborNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborStart, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborStartTemp, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborNum, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborListInit, totalNeighborNum * sizeof(int)));
    //CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborStart, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_neighborNumInit, vertNum * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_partId_map_real, partMapSize * sizeof(int)));
}

void TraditionalMAS32Preconditioner::initPreconditioner_Matrix()
{
    if(totalNodes < 1)
        return;
    CUDA_SAFE_CALL(cudaMemcpy(d_neighborList,
                              d_neighborListInit,
                              neighborListSize * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));
    //CUDA_SAFE_CALL(cudaMemcpy(ipc.pcg_data.MP.d_neighborStart, tetMesh.neighborStart.data(), ipc.vertexNum * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_neighborNum,
                              d_neighborNumInit,
                              totalNodes * sizeof(unsigned int),
                              cudaMemcpyDeviceToDevice));

    const int hierarchy_cluster_count = ReorderRealtime(0);
    int totalCluster = static_cast<int>(hierarchy_cluster_count * 1.05) + GPU_MAS_BANKSIZE;
    const size_t matrix_count =
        (static_cast<size_t>(totalCluster) + GPU_MAS_BANKSIZE - 1) / GPU_MAS_BANKSIZE;
    const size_t matrix_bytes = matrix_count
                                * (sizeof(__GEIGEN__::GPUMas32MatrixSymT)
                                   + sizeof(__GEIGEN__::GPUMas32MatrixT)
                                   + sizeof(__GEIGEN__::GPUMas32MatrixSymf));
    std::cout << "traditional GPU MAS32 hierarchy clusters: "
              << hierarchy_cluster_count << ", matrix workspace MiB: "
              << (matrix_bytes / (1024.0 * 1024.0)) << std::endl;
    size_t free_bytes_before = 0;
    size_t total_bytes       = 0;
    CUDA_SAFE_CALL(cudaMemGetInfo(&free_bytes_before, &total_bytes));
    std::cout << "traditional GPU MAS32 CUDA memory before matrices MiB: free="
              << (free_bytes_before / (1024.0 * 1024.0)) << ", total="
              << (total_bytes / (1024.0 * 1024.0)) << std::endl;
#ifdef SYME
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_inverseMatMas,
                              matrix_count * sizeof(__GEIGEN__::GPUMas32MatrixSymT)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_MatMas,
                              matrix_count * sizeof(__GEIGEN__::GPUMas32MatrixT)));
#else
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_MatMas,
                              totalCluster / GPU_MAS_BANKSIZE * sizeof(__GEIGEN__::GPUMas32MatrixT)));
#endif

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_precondMatMas,
                              matrix_count * sizeof(__GEIGEN__::GPUMas32MatrixSymf)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_factor_status, 2 * sizeof(int)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_multiLevelR, totalCluster * sizeof(Eigen::Vector3f)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_multiLevelZ, totalCluster * sizeof(Precision_T3)));
    size_t free_bytes_after = 0;
    CUDA_SAFE_CALL(cudaMemGetInfo(&free_bytes_after, &total_bytes));
    std::cout << "traditional GPU MAS32 CUDA memory after matrices MiB: free="
              << (free_bytes_after / (1024.0 * 1024.0)) << std::endl;
}

void TraditionalMAS32Preconditioner::FreeMAS()
{
    CUDA_SAFE_CALL(cudaFree(d_denseLevel));
    CUDA_SAFE_CALL(cudaFree(d_coarseSpaceTables));
    CUDA_SAFE_CALL(cudaFree(d_coarseTable));
    CUDA_SAFE_CALL(cudaFree(d_levelSize));
    CUDA_SAFE_CALL(cudaFree(d_goingNext));
    CUDA_SAFE_CALL(cudaFree(d_prefixOriginal));
    CUDA_SAFE_CALL(cudaFree(d_nextPrefix));
    CUDA_SAFE_CALL(cudaFree(d_nextPrefixSum));
    CUDA_SAFE_CALL(cudaFree(d_prefixSumOriginal));
    CUDA_SAFE_CALL(cudaFree(d_fineConnectMask));
    CUDA_SAFE_CALL(cudaFree(d_nextConnectMask));
    CUDA_SAFE_CALL(cudaFree(d_neighborList));
    CUDA_SAFE_CALL(cudaFree(d_neighborListInit));
    CUDA_SAFE_CALL(cudaFree(d_neighborStart));
    CUDA_SAFE_CALL(cudaFree(d_neighborStartTemp));
    CUDA_SAFE_CALL(cudaFree(d_neighborNum));
    CUDA_SAFE_CALL(cudaFree(d_neighborNumInit));
    CUDA_SAFE_CALL(cudaFree(d_partId_map_real));
    CUDA_SAFE_CALL(cudaFree(d_real_map_partId));
#ifdef SYME
    CUDA_SAFE_CALL(cudaFree(d_inverseMatMas));
    CUDA_SAFE_CALL(cudaFree(d_MatMas));
#else
    CUDA_SAFE_CALL(cudaFree(d_MatMas));
#endif

    CUDA_SAFE_CALL(cudaFree(d_precondMatMas));
    CUDA_SAFE_CALL(cudaFree(d_factor_status));
    CUDA_SAFE_CALL(cudaFree(d_multiLevelR));
    CUDA_SAFE_CALL(cudaFree(d_multiLevelZ));
}
}  // namespace gpu_mas32
