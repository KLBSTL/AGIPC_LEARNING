#include <agipc/agipc_criterion.cuh>
#include <TraditionalMAS32Preconditioner.cuh>
#include <cuda_tools/cuda_all.h>
#include <linear_system/utils/spmv.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace agipc
{
namespace
{
constexpr int kReplayThreads=256;

// CUDA event elapsed time includes host launch gaps and external GPU contention.
// These single-run diagnostics are not benchmark timings.
struct ReplayTimer
{
    cudaEvent_t begin=nullptr,end=nullptr;
    std::chrono::steady_clock::time_point wall_begin;
    ReplayTimer()
    {
        CUDA_SAFE_CALL(cudaEventCreate(&begin));
        CUDA_SAFE_CALL(cudaEventCreate(&end));
        CUDA_SAFE_CALL(cudaEventRecord(begin));
        wall_begin=std::chrono::steady_clock::now();
    }
    gipc::Json finish()
    {
        CUDA_SAFE_CALL(cudaEventRecord(end));
        CUDA_SAFE_CALL(cudaEventSynchronize(end));
        float elapsed=0;
        CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed,begin,end));
        const double wall=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-wall_begin).count();
        return {{"cuda_event_elapsed_ms",elapsed},{"wall_ms",wall}};
    }
    ~ReplayTimer() { if(begin) cudaEventDestroy(begin); if(end) cudaEventDestroy(end); }
};

template <typename T>
std::vector<T> read_binary(const std::filesystem::path& path,std::size_t count)
{
    if(std::filesystem::file_size(path)!=count*sizeof(T))
        throw std::runtime_error("Coarse snapshot binary size mismatch: "+path.string());
    std::vector<T> data(count);
    std::ifstream input;
    input.exceptions(std::ios::badbit|std::ios::failbit);
    input.open(path,std::ios::binary);
    if(count) input.read(reinterpret_cast<char*>(data.data()),count*sizeof(T));
    return data;
}

double replay_dot(const double* left,const double* right,int count)
{
    return thrust::inner_product(thrust::device_ptr<const double>(left),
        thrust::device_ptr<const double>(left)+count,
        thrust::device_ptr<const double>(right),0.0);
}

__global__ void replay_apply_diagonal(const Eigen::Matrix3d* inverse,
                                      const double* residual,double* z,int blocks)
{
    const int block=blockIdx.x*blockDim.x+threadIdx.x;
    if(block>=blocks) return;
    for(int row=0;row<3;++row)
    {
        double value=0;
        for(int col=0;col<3;++col) value+=inverse[block](row,col)*residual[3*block+col];
        z[3*block+row]=value;
    }
}

__global__ void replay_update(double* x,double* r,const double* p,
                              const double* ap,double alpha,int count)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) { x[i]+=alpha*p[i]; r[i]-=alpha*ap[i]; }
}

__global__ void replay_direction(double* p,const double* z,double beta,int count)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) p[i]=z[i]+beta*p[i];
}

__global__ void replay_residual(const double* b,const double* ax,double* r,int count)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) r[i]=b[i]-ax[i];
}

void replay_spmv(GIPCTripletMatrix& matrix,const double* x,double* y)
{
    const int dofs=3*matrix.block_rows();
    gipc::Spmv spmv;
    spmv.warp_reduce_sym_spmv(1.0,matrix.block_values(),matrix.block_row_indices(),
        matrix.block_col_indices(),matrix.h_unique_key_number,
        cudatool::CDenseVectorView<double>(x,dofs),0.0,
        cudatool::DenseVectorView<double>(y,dofs));
}

struct ReplayBuffers
{
    cudatool::CudaDeviceBuffer<double> x,r,z,p,ap;
    explicit ReplayBuffers(int dofs)
    { x.resize(dofs); r.resize(dofs); z.resize(dofs); p.resize(dofs); ap.resize(dofs); }
};

template <typename Apply>
gipc::Json run_replay_pcg(GIPCTripletMatrix& matrix,const double* rhs,
                          const std::vector<double>& reference,int real_blocks,int limit,Apply apply)
{
    const int dofs=3*matrix.block_rows();
    ReplayBuffers work(dofs);
    work.x.reset_zero();
    CUDA_SAFE_CALL(cudaMemcpy(work.r.data(),rhs,dofs*sizeof(double),cudaMemcpyDeviceToDevice));
    ReplayTimer timer;
    apply(work.r.data(),work.z.data());
    CUDA_SAFE_CALL(cudaMemcpy(work.p.data(),work.z.data(),dofs*sizeof(double),cudaMemcpyDeviceToDevice));
    const double initial2=replay_dot(rhs,rhs,dofs);
    double residual2=initial2,rz=replay_dot(work.r.data(),work.z.data(),dofs);
    int iterations=0;
    std::string failure;
    while(iterations<limit && residual2>1e-6*initial2)
    {
        if(!std::isfinite(rz) || rz<=0)
        { failure="nonpositive_preconditioned_residual"; break; }
        replay_spmv(matrix,work.p.data(),work.ap.data());
        const double curvature=replay_dot(work.p.data(),work.ap.data(),dofs);
        if(!std::isfinite(curvature) || curvature<=0)
        { failure="nonpositive_or_nonfinite_curvature"; break; }
        replay_update<<<(dofs+kReplayThreads-1)/kReplayThreads,kReplayThreads>>>(
            work.x.data(),work.r.data(),work.p.data(),work.ap.data(),rz/curvature,dofs);
        ++iterations;
        residual2=replay_dot(work.r.data(),work.r.data(),dofs);
        if(!std::isfinite(residual2)) { failure="nonfinite_residual"; break; }
        if(residual2<=1e-6*initial2) break;
        apply(work.r.data(),work.z.data());
        const double next_rz=replay_dot(work.r.data(),work.z.data(),dofs);
        if(!std::isfinite(next_rz) || next_rz<=0 || std::abs(rz)<1e-30)
        { failure="invalid_preconditioned_residual"; break; }
        replay_direction<<<(dofs+kReplayThreads-1)/kReplayThreads,kReplayThreads>>>(
            work.p.data(),work.z.data(),next_rz/rz,dofs);
        rz=next_rz;
    }
    const bool recursive_converged=residual2<=1e-6*initial2;
    if(!recursive_converged && failure.empty()) failure="iteration_cap";
    const auto timing=timer.finish();
    replay_spmv(matrix,work.x.data(),work.ap.data());
    replay_residual<<<(dofs+kReplayThreads-1)/kReplayThreads,kReplayThreads>>>(
        rhs,work.ap.data(),work.r.data(),dofs);
    const double true2=replay_dot(work.r.data(),work.r.data(),dofs);
    const double rhs_dot=replay_dot(rhs,work.x.data(),dofs);
    const double quadratic=replay_dot(work.x.data(),work.ap.data(),dofs);
    std::vector<double> solution;
    work.x.copy_to_host(solution);
    double difference2=0,reference2=0;
    for(int i=0;i<dofs;++i)
    { difference2+=(solution[i]-reference[i])*(solution[i]-reference[i]); reference2+=reference[i]*reference[i]; }
    double padding2=0;
    for(int i=3*real_blocks;i<dofs;++i) padding2+=solution[i]*solution[i];
    solution.resize(3*real_blocks);
    const double relative=initial2>0?std::sqrt(true2/initial2):std::sqrt(true2);
    return {{"iterations",iterations},{"max_iterations",limit},{"relative_tolerance",1e-3},
        {"recursive_converged",recursive_converged},{"failure_reason",failure},
        {"recursive_relative_residual",initial2>0?std::sqrt(residual2/initial2):0.0},
        {"true_relative_residual",relative},{"rhs_dot_direction",rhs_dot},
        {"predicted_quadratic_decrease",rhs_dot-0.5*quadratic},
        {"pcg_timing",timing},{"solution_f64",solution},
        {"padding_solution_norm",std::sqrt(padding2)},
        {"reference_relative_direction_difference",std::sqrt(difference2/std::max(reference2,1e-300))},
        {"passed",failure.empty() && std::isfinite(relative)
            && relative<=1e-3*(1.0+1e-6) && std::isfinite(quadratic)
            && (rhs_dot>0 || initial2==0)}};
}

struct ReplayMas
{
    gpu_mas32::TraditionalMAS32Preconditioner value;
    bool allocated=false;
    ~ReplayMas() { if(allocated) value.FreeMAS(); }
};
}

gipc::Json replay_coarse_snapshot(const std::string& sample_directory)
{
    const std::filesystem::path root(sample_directory);
    std::ifstream input(root/"metadata.json");
    if(!input) throw std::runtime_error("Cannot read coarse snapshot metadata");
    gipc::Json metadata;
    input>>metadata;
    const int blocks=metadata.at("coarse_block_nodes").get<int>();
    const int unique=metadata.at("coarse_unique_blocks").get<int>();
    if(blocks<1 || blocks>(std::numeric_limits<int>::max()/3-32) || unique<1)
        throw std::runtime_error("Invalid coarse snapshot dimensions");
    if(!metadata.value("coarse_matrix_half_storage",true))
        throw std::runtime_error("Coarse replay requires symmetric half storage");
    auto values=read_binary<Eigen::Matrix3d>(root/"coarse_A_values.f64x9.bin",unique);
    auto rows=read_binary<int>(root/"coarse_A_rows.i32.bin",unique);
    auto cols=read_binary<int>(root/"coarse_A_cols.i32.bin",unique);
    auto rhs=read_binary<double>(root/"coarse_rhs.f64.bin",3*blocks);
    auto reference=read_binary<double>(root/"coarse_solution.f64.bin",3*blocks);
    const int padded=(blocks+31)/32*32;
    const int iteration_cap=metadata.value("coarse_solve",gipc::Json::object())
        .value("max_iterations",std::min(512,3*blocks));
    if(iteration_cap<1 || iteration_cap>512)
        throw std::runtime_error("Invalid coarse replay iteration cap");
    std::vector<Eigen::Matrix3d> diagonal(padded,Eigen::Matrix3d::Zero());
    std::vector<std::vector<unsigned int>> graph(padded);
    for(int i=0;i<unique;++i)
    {
        const int row=rows[i],col=cols[i];
        if(row<0 || col<0 || row>=blocks || col>=blocks || !values[i].allFinite())
            throw std::runtime_error("Invalid coarse matrix entry");
        if(row==col) diagonal[row]+=values[i];
        else { graph[row].push_back(col); graph[col].push_back(row); }
    }
    for(int i=blocks;i<padded;++i)
    { rows.push_back(i); cols.push_back(i); values.push_back(Eigen::Matrix3d::Identity()); diagonal[i].setIdentity(); }
    rhs.resize(3*padded,0.0); reference.resize(3*padded,0.0);
    for(int i=0;i<3*blocks;++i)
        if(!std::isfinite(rhs[i]) || !std::isfinite(reference[i]))
            throw std::runtime_error("Nonfinite coarse RHS or reference solution");
    ReplayTimer jacobi_setup_timer;
    std::vector<Eigen::Matrix3d> inverse(padded);
    for(int i=0;i<padded;++i)
    {
        const double scale=std::max(diagonal[i].norm(),1e-300);
        Eigen::LLT<Eigen::Matrix3d> llt(diagonal[i]);
        if((diagonal[i]-diagonal[i].transpose()).norm()>1e-12*scale
           || llt.info()!=Eigen::Success)
            throw std::runtime_error("Coarse block-Jacobi diagonal is not SPD");
        inverse[i]=llt.solve(Eigen::Matrix3d::Identity());
    }
    cudatool::CudaDeviceBuffer<Eigen::Matrix3d> device_inverse;
    device_inverse.copy_from_host(inverse);
    const auto jacobi_setup_timing=jacobi_setup_timer.finish();
    GIPCTripletMatrix matrix;
    matrix.init_var();
    matrix.reshape(padded,padded);
    matrix.m_block_values.copy_from_host(values);
    matrix.m_block_row_indices.copy_from_host(rows);
    matrix.m_block_col_indices.copy_from_host(cols);
    matrix.h_unique_key_number=static_cast<int>(values.size());
    cudatool::CudaDeviceBuffer<double> device_rhs,device_reference,device_ax;
    device_rhs.copy_from_host(rhs); device_reference.copy_from_host(reference);
    device_ax.resize(rhs.size());
    replay_spmv(matrix,device_reference.data(),device_ax.data());
    std::vector<double> gpu_ax,cpu_ax(rhs.size(),0.0);
    device_ax.copy_to_host(gpu_ax);
    for(std::size_t i=0;i<values.size();++i)
    {
        const Eigen::Map<const Eigen::Vector3d> xcol(reference.data()+3*cols[i]);
        Eigen::Map<Eigen::Vector3d> yrow(cpu_ax.data()+3*rows[i]);
        yrow+=values[i]*xcol;
        if(rows[i]!=cols[i])
        {
            const Eigen::Map<const Eigen::Vector3d> xrow(reference.data()+3*rows[i]);
            Eigen::Map<Eigen::Vector3d> ycol(cpu_ax.data()+3*cols[i]);
            ycol+=values[i].transpose()*xrow;
        }
    }
    double error2=0,norm2=0;
    for(std::size_t i=0;i<rhs.size();++i)
    { error2+=(gpu_ax[i]-cpu_ax[i])*(gpu_ax[i]-cpu_ax[i]); norm2+=cpu_ax[i]*cpu_ax[i]; }
    const double spmv_error=std::sqrt(error2/std::max(norm2,1e-300));
    const auto jacobi=run_replay_pcg(matrix,device_rhs.data(),reference,blocks,iteration_cap,
        [&](const double* r,double* z) {
            replay_apply_diagonal<<<(padded+kReplayThreads-1)/kReplayThreads,kReplayThreads>>>(
                device_inverse.data(),r,z,padded);
        });
    ReplayTimer mas_setup_timer;
    std::vector<unsigned int> neighbors,starts(padded),counts(padded);
    std::vector<int> identity(padded);
    std::iota(identity.begin(),identity.end(),0);
    for(int i=0;i<padded;++i)
    {
        auto& adjacent=graph[i];
        std::sort(adjacent.begin(),adjacent.end());
        adjacent.erase(std::unique(adjacent.begin(),adjacent.end()),adjacent.end());
        starts[i]=static_cast<unsigned int>(neighbors.size()); counts[i]=adjacent.size();
        neighbors.insert(neighbors.end(),adjacent.begin(),adjacent.end());
    }
    ReplayMas mas;
    mas.allocated=true;
    mas.value.initPreconditioner_Neighbor(padded,0,std::max<std::size_t>(1,neighbors.size()),nullptr,padded);
    mas.value.neighborListSize=static_cast<int>(neighbors.size());
    if(!neighbors.empty()) CUDA_SAFE_CALL(cudaMemcpy(mas.value.d_neighborListInit,neighbors.data(),neighbors.size()*sizeof(unsigned int),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(mas.value.d_neighborStart,starts.data(),padded*sizeof(unsigned int),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(mas.value.d_neighborNumInit,counts.data(),padded*sizeof(unsigned int),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(mas.value.d_partId_map_real,identity.data(),padded*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(mas.value.d_real_map_partId,identity.data(),padded*sizeof(int),cudaMemcpyHostToDevice));
    mas.value.initPreconditioner_Matrix();
    std::vector<uint32_t> indices(values.size());
    std::iota(indices.begin(),indices.end(),uint32_t{0});
    cudatool::CudaDeviceBuffer<uint32_t> device_indices;
    device_indices.copy_from_host(indices);
    mas.value.setPreconditioner_bcoo(matrix.block_values(),matrix.block_row_indices(),
        matrix.block_col_indices(),device_indices.data(),0,values.size(),0);
    const auto mas_setup_timing=mas_setup_timer.finish();
    cudatool::CudaDeviceBuffer<double> initial_z;
    initial_z.resize(rhs.size());
    mas.value.preconditioning(reinterpret_cast<const double3*>(device_rhs.data()),
                             reinterpret_cast<double3*>(initial_z.data()));
    const auto diagnostics=mas.value.numerical_diagnostics(
        reinterpret_cast<const double3*>(device_rhs.data()));
    const auto gpu_diagnostics=mas.value.local_diagnostics_gpu();
    const bool diagnostics_agree=gpu_mas32::local_diagnostics_agree(gpu_diagnostics,diagnostics);
    gipc::Json mas_solve={{"passed",false},{"failure_reason","invalid_local_mas_blocks"}};
    if(diagnostics.value("passed",false))
        mas_solve=run_replay_pcg(matrix,device_rhs.data(),reference,blocks,iteration_cap,
            [&](const double* r,double* z) {
                mas.value.preconditioning(reinterpret_cast<const double3*>(r),reinterpret_cast<double3*>(z));
            });
    return {{"test","agipc_frozen_coarse_gpu_replay"},{"sample_directory",sample_directory},
        {"coarse_block_nodes",blocks},{"padded_block_nodes",padded},
        {"coarse_unique_blocks",unique},{"adjacency_edges",neighbors.size()/2},
        {"ordering","snapshot block order; identity part maps"},
        {"mas_scope","Traditional MAS32; matrix-pattern graph; diagnostic intermediate"},
        {"precision","FP64 matrix/PCG; existing MAS FP32 local inverse"},
        {"relative_spmv_error",spmv_error},{"block_jacobi",jacobi},
        {"setup_timing",{{"block_jacobi",jacobi_setup_timing},{"mas32",mas_setup_timing}}},
        {"timing_scope","diagnostic single run; setup then PCG; validation and vector serialization excluded"},
        {"solution_scope","original coarse blocks only; FP64 xyz block order"},
        {"mas32_local_diagnostics",diagnostics},{"mas32",mas_solve},
        {"mas32_gpu_local_diagnostics",gpu_diagnostics},{"cpu_gpu_local_diagnostics_agree",diagnostics_agree},
        {"passed",std::isfinite(spmv_error) && spmv_error<=1e-12
            && jacobi.value("passed",false) && mas_solve.value("passed",false) && diagnostics_agree},
        {"performance_claim",false},{"simulation_dispatch_changed",false}};
}
}
