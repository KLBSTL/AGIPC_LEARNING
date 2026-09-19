#include "agipc_criterion.cuh"
#include "agipc_tiny_dense.cuh"
#include "agipc_step_statistics.cuh"
#include "agipc_mas_apply.cuh"
#include "agipc_device_pcg.cuh"

#include <cuda_tools/cuda_device_buffer.h>
#include <linear_system/linear_system/global_matrix.h>
#include <linear_system/utils/converter.h>
#include <linear_system/utils/spmv.h>
#include <TraditionalMAS32Preconditioner.cuh>

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

namespace agipc
{
namespace
{
constexpr int kThreads=256;
constexpr double kResidualGrowthTolerance=1e-6;
// Traditional MAS stores each explicit 96x96 inverse in float. A 1e-12
// pivot floor suits an FP64 factor/solve path but is below this backend's
// resolvable inverse range. Keep this shift confined to the preconditioner.
constexpr double kPaperAffineMasRelativeShift=1.0e-4;
bool collect_timing = false;
bool collect_full_diagnostics = false;
StepStatisticsMode step_statistics_mode=StepStatisticsMode::Automatic;
bool direct_mas_apply=false;
int coarse_pcg_batch=0;
bool guarded_fine_correction=false;
bool guarded_fine_warm_start=false;

bool step_history_enabled()
{
    return step_statistics_mode==StepStatisticsMode::History
        || (step_statistics_mode==StepStatisticsMode::Automatic && collect_full_diagnostics);
}

struct RuntimeCoarseMas;
struct GalerkinState
{
    std::unique_ptr<GIPCTripletMatrix> coarse_matrix;
    cudatool::CudaDeviceBuffer<double> coarse_rhs;
    cudatool::CudaDeviceBuffer<int> invalid_entries;
    cudatool::CudaDeviceBuffer<int> expansion_counts;
    cudatool::CudaDeviceBuffer<int> expansion_offsets;
    cudatool::CudaDeviceBuffer<double> coarse_solution,coarse_r,coarse_z,coarse_p,coarse_ap;
    device_pcg::Workspace device_coarse_pcg;
    cudatool::CudaDeviceBuffer<double> prolonged,fine_solution,fine_ax,fine_residual;
    cudatool::CudaDeviceBuffer<double> fine_z,fine_p,fine_ap;
    cudatool::CudaDeviceBuffer<Eigen::Matrix3d> inverse_diagonal;
    cudatool::CudaDeviceBuffer<int> diagonal_found;
    cudatool::CudaDeviceBuffer<Eigen::Matrix3d> fine_inverse_diagonal;
    cudatool::CudaDeviceBuffer<int> fine_diagonal_found,fine_invalid_entries;
    int post_max_iterations=10;
    bool use_coarse_mas32=false;
    bool use_factorized_mas32=false;
    std::size_t mas_attempts=0,mas_local_failures=0;
    double total_mas_setup_ms=0,total_mas_validation_ms=0;
    std::string mas_validation="off";
    bool mas_reuse_enabled=false;
    std::size_t mas_graph_reuses=0;
    gipc::Json mas_failure_samples=gipc::Json::array();
    std::shared_ptr<RuntimeCoarseMas> runtime_mas;
    std::shared_ptr<TinyDenseSolver> tiny_dense;
    int tiny_dense_max_dofs=0;
    std::size_t tiny_dense_attempts=0,tiny_dense_successes=0,tiny_dense_rejections=0;
    gipc::Json coarse_solve_distribution=gipc::Json::array();
    bool adoption_enabled=false;
    bool candidate_ready=false;
    std::string candidate_failure_reason="not_assembled";
    std::size_t candidate_dofs=0;
    int candidate_iterations=0;
    std::size_t adoption_attempts=0,adoptions=0,fallbacks=0;
    gipc::Json fallback_reason_counts=gipc::Json::object();
    gipc::Json guarded_totals=gipc::Json::object();
    gipc::Json last_adoption=nullptr;
    gipc::Json last=nullptr;
    std::size_t updates=0;
    double total_ms=0;
    double total_assembly_ms=0;
    double total_coarse_solve_ms=0;
    double total_post_correction_ms=0;
    gipc::Json total_linear_timing=gipc::Json::object();
    std::string fallback_diagnostics_directory;
    std::array<bool,3> frozen_quality_categories={false,false,false};
    std::size_t direction_quality_evaluations=0;
    std::size_t residual_guard_restores=0;
    gipc::Json frozen_fallback_samples=gipc::Json::array();
    std::string coarse_diagnostics_directory;
    std::array<bool,7> frozen_coarse_categories={false,false,false,false,false,false,false};
    gipc::Json frozen_coarse_samples=gipc::Json::array();
    std::string direction_freeze_directory;
    std::size_t direction_freeze_after_update=0;
    bool direction_frozen=false;
    bool direction_continue=false;
    std::vector<int> direction_capture_frames;
    std::vector<int> direction_captured_frames;
    int direction_frame=0,direction_pairs=0,direction_ground_pairs=0;
    int direction_newton_ordinal=0,direction_late_after_newton=0;
    const int4* direction_collision_pairs=nullptr;
    std::vector<int> direction_captured_late_frames;
    std::size_t tracked_coarse_peak_bytes=0;
};

GalerkinState state;

__global__ void append_mas_padding(Eigen::Matrix3d* values,int* rows,int* cols,
                                   int unique,int blocks,int padding)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<padding)
    { values[unique+i].setIdentity(); rows[unique+i]=blocks+i; cols[unique+i]=blocks+i; }
}

// The paper's homogeneous 12-DoF basis intentionally keeps dependent columns
// for planar and linear aggregates.  Regularize only the MAS preconditioner
// copy; the Galerkin operator and right-hand side remain unchanged.  This is
// the same relative pivot floor used by the earlier AGIPC coarse MAS path.
__global__ void regularize_mas_diagonal(Eigen::Matrix3d* values,
                                        const int* rows,const int* cols,int count)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count || rows[i]!=cols[i]) return;
    auto value=values[i];
    double scale=0.0;
    for(int j=0;j<3;++j) scale=fmax(scale,fabs(value(j,j)));
    const double shift=fmax(1.0e-12,scale*kPaperAffineMasRelativeShift);
    for(int j=0;j<3;++j) value(j,j)+=shift;
    values[i]=value;
}

// Explicit intermediate runtime adapter. Original Galerkin dimensions and PCG
// logical dimensions stay unchanged; direct application also pads PCG r/z storage.
struct RuntimeCoarseMas
{
    gpu_mas32::TraditionalMAS32Preconditioner value;
    bool allocated=false;
    int blocks=0,padded=0;
    bool last_reused=false;
    bool nullspace_regularization=false;
    bool factorized=false;
    const double* diagnostic_rhs=nullptr;
    std::vector<int> cached_rows,cached_cols;
    cudatool::CudaDeviceBuffer<Eigen::Matrix3d> values;
    cudatool::CudaDeviceBuffer<int> rows,cols;
    cudatool::CudaDeviceBuffer<uint32_t> indices;
    cudatool::CudaDeviceBuffer<double> residual,z;
    ~RuntimeCoarseMas() { if(allocated) value.FreeMAS(); }
    void setup(const GIPCTripletMatrix& matrix,bool allow_reuse,
               bool regularize_nullspace,bool use_factorized=false)
    {
        const int old_blocks=blocks;
        blocks=matrix.block_rows(); padded=(blocks+31)/32*32;
        const int unique=matrix.h_unique_key_number,padding=padded-blocks;
        std::vector<int> host_rows(unique),host_cols(unique);
        CUDA_SAFE_CALL(cudaMemcpy(host_rows.data(),matrix.block_row_indices(),unique*sizeof(int),cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(host_cols.data(),matrix.block_col_indices(),unique*sizeof(int),cudaMemcpyDeviceToHost));
        last_reused=allocated && allow_reuse && old_blocks==blocks
            && nullspace_regularization==regularize_nullspace
            && factorized==use_factorized
            && host_rows==cached_rows && host_cols==cached_cols;
        value.set_factorized_local_solve(use_factorized);
        if(last_reused)
        {
            // The graph, padding and index buffers are unchanged; refresh only
            // the numerical blocks and the zero-padded residual.
            values.resize(unique+padding);
            CUDA_SAFE_CALL(cudaMemcpy(values.data(),matrix.block_values(),
                unique*sizeof(Eigen::Matrix3d),cudaMemcpyDeviceToDevice));
            if(regularize_nullspace)
                regularize_mas_diagonal<<<(unique+kThreads-1)/kThreads,kThreads>>>(
                    values.data(),rows.data(),cols.data(),unique);
            residual.resize(3*padded); residual.reset_zero(); z.resize(3*padded);
            value.refresh_fixed_graph_bcoo(values.data(),rows.data(),cols.data(),
                                            indices.data(),0,unique+padding);
            return;
        }
        if(allocated) { value.FreeMAS();allocated=false; }
        std::vector<int> identity(padded);
        std::vector<std::vector<unsigned int>> graph(padded);
        for(int i=0;i<unique;++i)
        {
            const int row=host_rows[i],col=host_cols[i];
            if(row<0 || col<0 || row>=blocks || col>=blocks)
                throw std::runtime_error("Invalid runtime coarse MAS graph index");
            if(row!=col) { graph[row].push_back(col); graph[col].push_back(row); }
        }
        std::vector<unsigned int> neighbors,starts(padded),counts(padded);
        for(int i=0;i<padded;++i)
        {
            auto& adjacent=graph[i];
            std::sort(adjacent.begin(),adjacent.end());
            adjacent.erase(std::unique(adjacent.begin(),adjacent.end()),adjacent.end());
            starts[i]=static_cast<unsigned int>(neighbors.size()); counts[i]=adjacent.size();
            neighbors.insert(neighbors.end(),adjacent.begin(),adjacent.end());
        }
        std::iota(identity.begin(),identity.end(),0);
        values.resize(unique+padding); rows.resize(unique+padding); cols.resize(unique+padding);
        CUDA_SAFE_CALL(cudaMemcpy(values.data(),matrix.block_values(),unique*sizeof(Eigen::Matrix3d),cudaMemcpyDeviceToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(rows.data(),matrix.block_row_indices(),unique*sizeof(int),cudaMemcpyDeviceToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(cols.data(),matrix.block_col_indices(),unique*sizeof(int),cudaMemcpyDeviceToDevice));
        if(padding) append_mas_padding<<<1,32>>>(values.data(),rows.data(),cols.data(),unique,blocks,padding);
        if(regularize_nullspace)
            regularize_mas_diagonal<<<(unique+padding+kThreads-1)/kThreads,kThreads>>>(
                values.data(),rows.data(),cols.data(),unique+padding);
        std::vector<uint32_t> host_indices(unique+padding);
        std::iota(host_indices.begin(),host_indices.end(),uint32_t{0}); indices.copy_from_host(host_indices);
        residual.resize(3*padded); residual.reset_zero(); z.resize(3*padded);
        allocated=true;
        value.initPreconditioner_Neighbor(padded,0,std::max<std::size_t>(1,neighbors.size()),nullptr,padded);
        value.neighborListSize=static_cast<int>(neighbors.size());
        if(!neighbors.empty()) CUDA_SAFE_CALL(cudaMemcpy(value.d_neighborListInit,neighbors.data(),neighbors.size()*sizeof(unsigned int),cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(value.d_neighborStart,starts.data(),padded*sizeof(unsigned int),cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(value.d_neighborNumInit,counts.data(),padded*sizeof(unsigned int),cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(value.d_partId_map_real,identity.data(),padded*sizeof(int),cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(value.d_real_map_partId,identity.data(),padded*sizeof(int),cudaMemcpyHostToDevice));
        value.initPreconditioner_Matrix();
        value.setPreconditioner_bcoo(values.data(),rows.data(),cols.data(),indices.data(),0,unique+padding,0);
        nullspace_regularization=regularize_nullspace;
        factorized=use_factorized;
        cached_rows=std::move(host_rows);cached_cols=std::move(host_cols);
    }
    void apply(const double* input,double* output,bool direct)
    {
        // Direct callers own 3*padded doubles and zero the isolated RHS tail
        // once per coarse system. All PCG updates/dots retain 3*blocks entries.
        if(direct)
        {
            diagnostic_rhs=input;
            value.preconditioning(reinterpret_cast<const double3*>(input),reinterpret_cast<double3*>(output));
        }
        else
        {
            CUDA_SAFE_CALL(cudaMemcpy(residual.data(),input,3*blocks*sizeof(double),cudaMemcpyDeviceToDevice));
            diagnostic_rhs=residual.data();
            value.preconditioning(reinterpret_cast<const double3*>(residual.data()),reinterpret_cast<double3*>(z.data()));
            CUDA_SAFE_CALL(cudaMemcpy(output,z.data(),3*blocks*sizeof(double),cudaMemcpyDeviceToDevice));
        }
    }
    gipc::Json diagnostics(const std::string& mode) const
    {
        if(factorized)
        {
            auto result=value.factor_diagnostics();
            result["validation_mode_requested"]=mode;
            result["paper_affine_nullspace_regularization"]=false;
            return result;
        }
        if(mode=="cpu")
        {
            auto result=value.numerical_diagnostics(
                reinterpret_cast<const double3*>(diagnostic_rhs));
            result["paper_affine_nullspace_regularization"]=nullspace_regularization;
            result["relative_diagonal_shift"]=nullspace_regularization
                ?kPaperAffineMasRelativeShift:0.0;
            return result;
        }
        auto result=value.local_diagnostics_gpu();
        if(mode=="crosscheck")
        {
            const auto cpu=value.numerical_diagnostics(reinterpret_cast<const double3*>(diagnostic_rhs));
            const bool agreement=gpu_mas32::local_diagnostics_agree(result,cpu);
            result["gpu_passed"]=result.value("passed",false);
            result["cpu_gpu_agreement"]=agreement;result["cpu_reference"]=cpu;
            result["passed"]=result.value("passed",false) && agreement;
        }
        result["paper_affine_nullspace_regularization"]=nullspace_regularization;
        result["relative_diagonal_shift"]=nullspace_regularization
            ?kPaperAffineMasRelativeShift:0.0;
        return result;
    }
};

__device__ void transform_for_block(int fine_block,int prefix_blocks,
                                    const int* fine_to_coarse,
                                    const int* coarse_block_bases,
                                    const int* basis_masks,
                                    const double3* rest_positions,
                                    int& base,int& width,double phi[4])
{
    phi[0]=1.0; phi[1]=phi[2]=phi[3]=0.0;
    if(fine_block<prefix_blocks)
    {
        base=fine_block; width=1; return;
    }
    const int local=fine_block-prefix_blocks;
    const int coarse=fine_to_coarse[local];
    base=prefix_blocks+coarse_block_bases[coarse];
    const int mask=basis_masks[coarse];
    width=__popc(static_cast<unsigned>(mask));
    const double3 x=rest_positions[local];
    const double raw[4]={1.0,x.x,x.y,x.z};
    int output=0;
    for(int column=0;column<4;++column)
        if(mask&(1<<column)) phi[output++]=raw[column];
}

__global__ void count_expanded_blocks(const int* fine_rows,const int* fine_cols,
                                      int* counts,const int* fine_to_coarse,
                                      const int* coarse_block_bases,
                                      const int* basis_masks,const double3* rest_positions,
                                      int prefix_blocks,int count)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    int row_base,row_width,col_base,col_width; double row_phi[4],col_phi[4];
    transform_for_block(fine_rows[i],prefix_blocks,fine_to_coarse,coarse_block_bases,
                        basis_masks,rest_positions,row_base,row_width,row_phi);
    transform_for_block(fine_cols[i],prefix_blocks,fine_to_coarse,coarse_block_bases,
                        basis_masks,rest_positions,col_base,col_width,col_phi);
    counts[i]=fine_rows[i]==fine_cols[i]
        ? row_width*(row_width+1)/2 : row_width*col_width;
}

__global__ void emit_expanded_blocks(const Eigen::Matrix3d* fine_values,
                           const int* fine_rows,const int* fine_cols,
                           Eigen::Matrix3d* coarse_values,int* coarse_rows,int* coarse_cols,
                           const int* offsets,const int* fine_to_coarse,
                           const int* coarse_block_bases,const int* basis_masks,
                           const double3* rest_positions,int prefix_blocks,
                           int fine_block_count,int* invalid_entries,int count)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    const int fine_blocks=fine_block_count;
    const int row=fine_rows[i], col=fine_cols[i];
    if(row<0 || row>=fine_blocks || col<0 || col>=fine_blocks)
    {
        atomicAdd(invalid_entries,1);
        coarse_rows[i]=coarse_cols[i]=0;
        coarse_values[i].setZero();
        return;
    }
    int row_base,row_width,col_base,col_width; double row_phi[4],col_phi[4];
    transform_for_block(row,prefix_blocks,fine_to_coarse,coarse_block_bases,
                        basis_masks,rest_positions,row_base,row_width,row_phi);
    transform_for_block(col,prefix_blocks,fine_to_coarse,coarse_block_bases,
                        basis_masks,rest_positions,col_base,col_width,col_phi);
    const Eigen::Matrix3d input=fine_values[i];
    int output=offsets[i];
    for(int a=0;a<row_width;++a) for(int b=0;b<col_width;++b)
    {
        if(row==col && a>b) continue;
        int mapped_row=row_base+a,mapped_col=col_base+b;
        Eigen::Matrix3d value=(row_phi[a]*col_phi[b])*input;
        if(row!=col && mapped_row==mapped_col)
            value+=value.transpose().eval();
        else if(mapped_row>mapped_col)
        {
            const int tmp=mapped_row; mapped_row=mapped_col; mapped_col=tmp;
            value=value.transpose().eval();
        }
        coarse_rows[output]=mapped_row;
        coarse_cols[output]=mapped_col;
        coarse_values[output]=value;
        for(int e=0;e<9;++e)
            if(!isfinite(value.data()[e])) atomicAdd(invalid_entries,1);
        ++output;
    }
}

__global__ void reduce_mixed_rhs(const double* fine_rhs,
                           double* coarse_rhs,
                           const int* fine_to_coarse,
                           const int* coarse_block_bases,
                           const int* basis_masks,
                           const double3* rest_positions,int prefix_blocks,
                           int* invalid_entries,
                           int fine_blocks)
{
    const int block=blockIdx.x*blockDim.x+threadIdx.x;
    if(block>=fine_blocks) return;
    int base,width; double phi[4];
    transform_for_block(block,prefix_blocks,fine_to_coarse,coarse_block_bases,
                        basis_masks,rest_positions,base,width,phi);
    for(int component=0;component<3;++component)
    {
        const double value=fine_rhs[3*block+component];
        if(!isfinite(value)) { atomicAdd(invalid_entries,1); continue; }
        for(int a=0;a<width;++a)
            atomicAdd(coarse_rhs+3*(base+a)+component,phi[a]*value);
    }
}

__global__ void initialize_inverse_diagonal(Eigen::Matrix3d* inverse,int* found,int count)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) { inverse[i].setIdentity(); found[i]=0; }
}

__global__ void extract_inverse_diagonal(const Eigen::Matrix3d* values,const int* rows,
                                         const int* cols,Eigen::Matrix3d* inverse,int* found,
                                         int* invalid,int count)
{
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=count || rows[index]!=cols[index]) return;
    const Eigen::Matrix3d m=values[index];
    const double a=m(0,0),b=m(0,1),c=m(0,2),d=m(1,0),e=m(1,1),f=m(1,2);
    const double g=m(2,0),h=m(2,1),i=m(2,2);
    const double det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
    if(!isfinite(det) || fabs(det)<1e-30) { atomicAdd(invalid,1); return; }
    Eigen::Matrix3d inv;
    inv(0,0)=(e*i-f*h)/det; inv(0,1)=(c*h-b*i)/det; inv(0,2)=(b*f-c*e)/det;
    inv(1,0)=(f*g-d*i)/det; inv(1,1)=(a*i-c*g)/det; inv(1,2)=(c*d-a*f)/det;
    inv(2,0)=(d*h-e*g)/det; inv(2,1)=(b*g-a*h)/det; inv(2,2)=(a*e-b*d)/det;
    inverse[rows[index]]=inv; found[rows[index]]=1;
}

__global__ void apply_diagonal(const Eigen::Matrix3d* inverse,const double* input,
                               double* output,int blocks)
{
    const int block=blockIdx.x*blockDim.x+threadIdx.x;
    if(block>=blocks) return;
    const Eigen::Vector3d v(input[3*block],input[3*block+1],input[3*block+2]);
    const Eigen::Vector3d z=inverse[block]*v;
    output[3*block]=z.x(); output[3*block+1]=z.y(); output[3*block+2]=z.z();
}

__global__ void update_x_r(double* x,double* r,const double* p,const double* ap,
                           double alpha,int count)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) { x[i]+=alpha*p[i]; r[i]-=alpha*ap[i]; }
}

__global__ void update_p(double* p,const double* z,double beta,int count)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) p[i]=z[i]+beta*p[i];
}

__global__ void prolongate_mixed(const double* coarse,double* fine,
                                 const int* fine_to_coarse,const int* bases,
                                 const int* basis_masks,const double3* rest,int prefix,int fine_blocks)
{
    const int block=blockIdx.x*blockDim.x+threadIdx.x;
    if(block>=fine_blocks) return;
    int base,width; double phi[4];
    transform_for_block(block,prefix,fine_to_coarse,bases,basis_masks,rest,base,width,phi);
    for(int component=0;component<3;++component)
    {
        double value=0;
        for(int a=0;a<width;++a) value+=phi[a]*coarse[3*(base+a)+component];
        fine[3*block+component]=value;
    }
}

__global__ void form_residual(const double* rhs,const double* ax,double* residual,int count)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) residual[i]=rhs[i]-ax[i];
}

double device_dot(const double* a,const double* b,int count)
{
    return thrust::inner_product(thrust::device_ptr<const double>(a),
        thrust::device_ptr<const double>(a)+count,thrust::device_ptr<const double>(b),0.0);
}

struct ResidualDots
{
    double rr,rz;
};

struct ResidualDotTerm
{
    __host__ __device__ ResidualDots operator()(const thrust::tuple<double,double>& values) const
    {
        const double r=thrust::get<0>(values);
        return {r*r,r*thrust::get<1>(values)};
    }
};

struct ResidualDotSum
{
    __host__ __device__ ResidualDots operator()(ResidualDots a,ResidualDots b) const
    {
        return {a.rr+b.rr,a.rz+b.rz};
    }
};

ResidualDots device_residual_dots(const double* residual,const double* preconditioned,int count)
{
    // One reduction returns both scalars needed by a host-driven PCG step.
    const auto first=thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_ptr<const double>(residual),
        thrust::device_ptr<const double>(preconditioned)));
    return thrust::transform_reduce(first,first+count,ResidualDotTerm{},
                                    ResidualDots{0.0,0.0},ResidualDotSum{});
}

template <typename T>
void write_binary(const std::filesystem::path& path,const T* device_values,std::size_t count)
{
    std::vector<T> host(count);
    if(count>0)
        CUDA_SAFE_CALL(cudaMemcpy(host.data(),device_values,count*sizeof(T),cudaMemcpyDeviceToHost));
    std::ofstream output;
    output.exceptions(std::ios::badbit|std::ios::failbit);
    output.open(path,std::ios::binary|std::ios::trunc);
    output.write(reinterpret_cast<const char*>(host.data()),
                 static_cast<std::streamsize>(host.size()*sizeof(T)));
}

gipc::Json direction_metrics(GalerkinState& target,
                             const GIPCTripletMatrix& fine_matrix,
                             const double* fine_rhs,
                             const double* direction,
                             int fine_dofs)
{
    gipc::Spmv spmv;
    target.fine_ax.resize(fine_dofs);
    target.fine_residual.resize(fine_dofs);
    spmv.warp_reduce_sym_spmv(1.0,
        const_cast<Eigen::Matrix3d*>(fine_matrix.block_values()),
        const_cast<int*>(fine_matrix.block_row_indices()),
        const_cast<int*>(fine_matrix.block_col_indices()),fine_matrix.h_unique_key_number,
        cudatool::CDenseVectorView<double>(direction,fine_dofs),0.0,
        cudatool::DenseVectorView<double>(target.fine_ax.data(),fine_dofs));
    form_residual<<<(fine_dofs+kThreads-1)/kThreads,kThreads>>>(
        fine_rhs,target.fine_ax.data(),target.fine_residual.data(),fine_dofs);
    const double residual2=device_dot(target.fine_residual.data(),
                                      target.fine_residual.data(),fine_dofs);
    const double norm2=device_dot(direction,direction,fine_dofs);
    const double rhs_dot=device_dot(fine_rhs,direction,fine_dofs);
    const double quadratic=device_dot(direction,target.fine_ax.data(),fine_dofs);
    return {{"residual_norm",std::sqrt(std::max(0.0,residual2))},
            {"direction_norm",std::sqrt(std::max(0.0,norm2))},
            {"rhs_dot_direction",rhs_dot},{"direction_dot_A_direction",quadratic},
            {"predicted_quadratic_decrease",rhs_dot-0.5*quadratic}};
}

int quality_category(double post_reduction)
{
    if(post_reduction<=1.25) return 0;
    if(post_reduction<=2.0) return 1;
    return 2;
}

const char* quality_category_name(int category)
{
    constexpr const char* names[]={"mild","moderate","severe"};
    return names[std::clamp(category,0,2)];
}

gipc::Json freeze_coarse_system(GalerkinState& target)
{
    if(target.coarse_diagnostics_directory.empty()) return nullptr;
    const auto& solve=target.last["coarse_solve"];
    const int blocks=target.coarse_matrix->block_rows();
    int category=-1;
    if(solve.value("failure_reason",std::string{})=="iteration_cap") category=2;
    else if(solve.value("failure_reason",std::string{})=="mas_local_diagnostics_failed") category=3;
    else if(solve.value("converged",false) && blocks>1024) category=1;
    else if(solve.value("converged",false) && blocks>=257) category=0;
    else if(solve.value("converged",false) && blocks<=8) category=4;
    else if(solve.value("converged",false) && blocks<=16) category=5;
    else if(solve.value("converged",false) && blocks<=32) category=6;
    if(category<0 || target.frozen_coarse_categories[category]) return nullptr;

    const auto started=std::chrono::steady_clock::now();
    constexpr const char* names[]={"medium_converged","large_converged","iteration_cap","local_diagnostics_failure","tiny24_converged","tiny48_converged","tiny96_converged"};
    const std::filesystem::path root(target.coarse_diagnostics_directory);
    const std::string sample_name=std::string{names[category]}+"_"
        +std::to_string(target.updates);
    const auto directory=root/sample_name;
    std::filesystem::create_directories(directory);
    const auto& matrix=*target.coarse_matrix;
    const std::size_t unique=matrix.h_unique_key_number;
    write_binary(directory/"coarse_A_values.f64x9.bin",matrix.block_values(),unique);
    write_binary(directory/"coarse_A_rows.i32.bin",matrix.block_row_indices(),unique);
    write_binary(directory/"coarse_A_cols.i32.bin",matrix.block_col_indices(),unique);
    write_binary(directory/"coarse_rhs.f64.bin",target.coarse_rhs.data(),
                 target.coarse_rhs.size());
    write_binary(directory/"coarse_solution.f64.bin",target.coarse_solution.data(),
                 target.coarse_solution.size());
    gipc::Json metadata={
        {"format","agipc_coarse_system_v1"},{"sample_name",sample_name},
        {"category",names[category]},{"update_index",target.updates},
        {"coarse_block_nodes",blocks},{"coarse_dofs",3*blocks},
        {"coarse_unique_blocks",unique},{"coarse_matrix_half_storage",true},
        {"matrix_block_layout","Eigen column-major FP64 3x3"},
        {"preconditioner",solve.value("preconditioner",std::string{"block_jacobi"})},{"coarse_solve",solve},
        {"reference_solution_valid",solve.value("converged",false)},
        {"coarse_solution_role","current iterate; zero or partial for rejected solves"},
        {"fine_block_nodes",target.last["fine_block_nodes"]},
        {"coarse_fem_nodes",target.last["coarse_fem_nodes"]},
        {"candidate_ready",target.candidate_ready},
        {"candidate_failure_reason",target.candidate_failure_reason},
        {"binary_freeze_ms",std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-started).count()}};
    auto samples=target.frozen_coarse_samples;
    samples.push_back(metadata);
    std::ofstream output;
    output.exceptions(std::ios::badbit|std::ios::failbit);
    output.open(directory/"metadata.json");
    output<<metadata.dump(2)<<'\n';
    output.close();
    output.open(root/"manifest.json");
    output<<gipc::Json{{"format","agipc_coarse_samples_v1"},
                      {"samples",samples}}.dump(2)<<'\n';
    output.close();
    target.frozen_coarse_categories[category]=true;
    target.frozen_coarse_samples=std::move(samples);
    return metadata;
}

gipc::Json post_correct_shadow(GalerkinState& target,const GIPCTripletMatrix& fine,
                               const double* fine_rhs,int fine_dofs)
{
    const int blocks=fine.block_rows(),unique=fine.h_unique_key_number;
    const int max_iterations=target.post_max_iterations;
    target.fine_solution.resize(fine_dofs);
    CUDA_SAFE_CALL(cudaMemcpy(target.fine_solution.data(),target.prolonged.data(),
                              fine_dofs*sizeof(double),cudaMemcpyDeviceToDevice));
    const double rhs2=device_dot(fine_rhs,fine_rhs,fine_dofs);
    const double initial2=device_dot(target.fine_residual.data(),
                                     target.fine_residual.data(),fine_dofs);
    const double initial_solution2=device_dot(target.fine_solution.data(),
                                              target.fine_solution.data(),fine_dofs);
    const double initial_rhs_dot=device_dot(fine_rhs,target.fine_solution.data(),fine_dofs);
    std::vector<double> residual_history;
    if(collect_full_diagnostics) residual_history.push_back(std::sqrt(std::max(0.0,initial2)));
    std::vector<double> curvature_history;
    if(max_iterations==0)
    {
        const double reduction=initial2>0?1.0:0.0;
        const bool candidate_valid=std::isfinite(initial2) && std::isfinite(initial_solution2)
            && std::isfinite(initial_rhs_dot)
            && reduction<=1.0+kResidualGrowthTolerance
            && (initial_rhs_dot>0 || rhs2==0);
        return {{"attempted",true},{"converged",initial2<=1e-6*rhs2},
                {"stop_reason","cap_zero_ablation"},
                {"iterations",0},{"max_iterations",0},{"relative_tolerance",1e-3},
                {"initial_solution_norm",std::sqrt(std::max(0.0,initial_solution2))},
                {"nonzero_initial_guess",initial_solution2>0},
                {"initial_residual_norm",std::sqrt(std::max(0.0,initial2))},
                {"final_residual_norm",std::sqrt(std::max(0.0,initial2))},
                {"residual_reduction_ratio",1.0},{"residual_history",residual_history},
                {"curvature_history",curvature_history},{"rhs_dot_direction",initial_rhs_dot},
                {"candidate_valid",candidate_valid},{"controls_solver",false}};
    }

    target.fine_z.resize(fine_dofs); target.fine_p.resize(fine_dofs);
    target.fine_ap.resize(fine_dofs);
    target.fine_inverse_diagonal.resize(blocks);
    target.fine_diagonal_found.resize(blocks);
    target.fine_invalid_entries.resize(1); target.fine_invalid_entries.reset_zero();
    initialize_inverse_diagonal<<<(blocks+kThreads-1)/kThreads,kThreads>>>(
        target.fine_inverse_diagonal.data(),target.fine_diagonal_found.data(),blocks);
    extract_inverse_diagonal<<<(unique+kThreads-1)/kThreads,kThreads>>>(
        fine.block_values(),fine.block_row_indices(),fine.block_col_indices(),
        target.fine_inverse_diagonal.data(),target.fine_diagonal_found.data(),
        target.fine_invalid_entries.data(),unique);
    const int found=thrust::reduce(thrust::device_ptr<int>(target.fine_diagonal_found.data()),
        thrust::device_ptr<int>(target.fine_diagonal_found.data())+blocks,0);
    std::vector<int> invalid;
    target.fine_invalid_entries.copy_to_host(invalid);
    if(found!=blocks || invalid[0]!=0)
        return {{"attempted",true},{"converged",false},
                {"stop_reason","missing_or_singular_diagonal"},{"iterations",0},
                {"max_iterations",max_iterations},{"diagonal_blocks_found",found},
                {"diagonal_blocks_expected",blocks},{"invalid_diagonal_blocks",invalid[0]},
                {"initial_solution_norm",std::sqrt(std::max(0.0,initial_solution2))},
                {"nonzero_initial_guess",initial_solution2>0},
                {"initial_residual_norm",std::sqrt(std::max(0.0,initial2))},
                {"final_residual_norm",std::sqrt(std::max(0.0,initial2))},
                {"residual_reduction_ratio",1.0},{"residual_history",residual_history},
                {"curvature_history",curvature_history},{"candidate_valid",false},
                {"controls_solver",false}};

    const double tolerance2=1e-6*rhs2;
    double residual2=initial2;
    int iterations=0;
    std::string stop_reason;
    if(residual2<=tolerance2)
        stop_reason="residual_tolerance";
    else
    {
        apply_diagonal<<<(blocks+kThreads-1)/kThreads,kThreads>>>(
            target.fine_inverse_diagonal.data(),target.fine_residual.data(),
            target.fine_z.data(),blocks);
        CUDA_SAFE_CALL(cudaMemcpy(target.fine_p.data(),target.fine_z.data(),
                                  fine_dofs*sizeof(double),cudaMemcpyDeviceToDevice));
        double rz=device_dot(target.fine_residual.data(),target.fine_z.data(),fine_dofs);
        gipc::Spmv spmv;
        while(iterations<max_iterations && residual2>tolerance2)
        {
            spmv.warp_reduce_sym_spmv(1.0,
                const_cast<Eigen::Matrix3d*>(fine.block_values()),
                const_cast<int*>(fine.block_row_indices()),
                const_cast<int*>(fine.block_col_indices()),unique,
                cudatool::CDenseVectorView<double>(target.fine_p.data(),fine_dofs),0.0,
                cudatool::DenseVectorView<double>(target.fine_ap.data(),fine_dofs));
            const double curvature=device_dot(target.fine_p.data(),target.fine_ap.data(),fine_dofs);
            if(collect_full_diagnostics) curvature_history.push_back(curvature);
            if(!std::isfinite(curvature) || curvature<=0 || !std::isfinite(rz))
            {
                stop_reason=!std::isfinite(curvature)?"nonfinite_curvature":"nonpositive_curvature";
                break;
            }
            const double alpha=rz/curvature;
            update_x_r<<<(fine_dofs+kThreads-1)/kThreads,kThreads>>>(
                target.fine_solution.data(),target.fine_residual.data(),target.fine_p.data(),
                target.fine_ap.data(),alpha,fine_dofs);
            ++iterations;
            apply_diagonal<<<(blocks+kThreads-1)/kThreads,kThreads>>>(
                target.fine_inverse_diagonal.data(),target.fine_residual.data(),
                target.fine_z.data(),blocks);
            const auto next_dots=device_residual_dots(target.fine_residual.data(),
                                                       target.fine_z.data(),fine_dofs);
            residual2=next_dots.rr;
            if(collect_full_diagnostics) residual_history.push_back(std::sqrt(std::max(0.0,residual2)));
            if(!std::isfinite(residual2)) { stop_reason="nonfinite_residual"; break; }
            if(residual2<=tolerance2) { stop_reason="residual_tolerance"; break; }
            const double next_rz=next_dots.rz;
            if(!std::isfinite(next_rz) || next_rz<=0 || !std::isfinite(rz) || rz<=0)
            { stop_reason="invalid_preconditioned_residual"; break; }
            update_p<<<(fine_dofs+kThreads-1)/kThreads,kThreads>>>(target.fine_p.data(),
                target.fine_z.data(),next_rz/rz,fine_dofs);
            rz=next_rz;
        }
        if(stop_reason.empty()) stop_reason="iteration_cap";
    }
    const double recursive_residual2=residual2;
    gipc::Spmv validation_spmv;
    validation_spmv.warp_reduce_sym_spmv(1.0,
        const_cast<Eigen::Matrix3d*>(fine.block_values()),
        const_cast<int*>(fine.block_row_indices()),
        const_cast<int*>(fine.block_col_indices()),unique,
        cudatool::CDenseVectorView<double>(target.fine_solution.data(),fine_dofs),0.0,
        cudatool::DenseVectorView<double>(target.fine_ax.data(),fine_dofs));
    form_residual<<<(fine_dofs+kThreads-1)/kThreads,kThreads>>>(fine_rhs,
        target.fine_ax.data(),target.fine_residual.data(),fine_dofs);
    residual2=device_dot(target.fine_residual.data(),target.fine_residual.data(),fine_dofs);
    const double raw_final_residual2=residual2;
    const double raw_reduction=initial2>0
        ? std::sqrt(std::max(0.0,raw_final_residual2)/initial2)
        : (raw_final_residual2==0?0.0:std::numeric_limits<double>::infinity());
    std::string selected_solution="post_corrected";
    if(!std::isfinite(raw_reduction)
       || raw_reduction>1.0+kResidualGrowthTolerance)
    {
        CUDA_SAFE_CALL(cudaMemcpy(target.fine_solution.data(),target.prolonged.data(),
                                  fine_dofs*sizeof(double),cudaMemcpyDeviceToDevice));
        residual2=initial2;
        selected_solution="prolongated_residual_guard";
        ++target.residual_guard_restores;
    }
    const bool converged=residual2<=tolerance2;
    const double final_solution2=device_dot(target.fine_solution.data(),
                                            target.fine_solution.data(),fine_dofs);
    const double rhs_dot_direction=device_dot(fine_rhs,target.fine_solution.data(),fine_dofs);
    const double reduction=initial2>0?std::sqrt(std::max(0.0,residual2)/initial2):0.0;
    const bool accepted_stop=stop_reason=="residual_tolerance" || stop_reason=="iteration_cap";
    const bool candidate_valid=accepted_stop && std::isfinite(residual2)
        && std::isfinite(final_solution2) && std::isfinite(rhs_dot_direction)
        && reduction<=1.0+kResidualGrowthTolerance
        && (rhs_dot_direction>0 || rhs2==0);
    return {{"attempted",true},{"converged",converged},{"stop_reason",stop_reason},
            {"iterations",iterations},{"max_iterations",max_iterations},
            {"relative_tolerance",1e-3},{"tolerance_reference","fine_rhs_norm"},
            {"initial_solution_norm",std::sqrt(std::max(0.0,initial_solution2))},
            {"final_solution_norm",std::sqrt(std::max(0.0,final_solution2))},
            {"nonzero_initial_guess",initial_solution2>0},
            {"initial_residual_norm",std::sqrt(std::max(0.0,initial2))},
            {"recursive_final_residual_norm",std::sqrt(std::max(0.0,recursive_residual2))},
            {"raw_final_residual_norm",std::sqrt(std::max(0.0,raw_final_residual2))},
            {"raw_residual_reduction_ratio",raw_reduction},
            {"final_residual_norm",std::sqrt(std::max(0.0,residual2))},
            {"residual_reduction_ratio",reduction},
            {"residual_reduced",std::isfinite(reduction)
                && reduction<=1.0+kResidualGrowthTolerance},
            {"selected_solution",selected_solution},
            {"rhs_dot_direction",rhs_dot_direction},{"residual_history",residual_history},
            {"curvature_history",curvature_history},{"candidate_valid",candidate_valid},
            {"controls_solver",false}};
}

gipc::Json solve_coarse_shadow(GalerkinState& target,const GIPCTripletMatrix& fine,
                               const double* fine_rhs,int fine_dofs,MappingDeviceView mapping,
                               int prefix_blocks)
{
    auto& coarse=*target.coarse_matrix;
    const int blocks=coarse.block_rows(),dofs=3*blocks,unique=coarse.h_unique_key_number;
    target.coarse_solution.resize(dofs); target.coarse_solution.reset_zero();
    gipc::Json dense_attempt=nullptr;
    if(target.tiny_dense_max_dofs>0 && dofs<=target.tiny_dense_max_dofs)
    {
        ++target.tiny_dense_attempts;
        const auto started=std::chrono::steady_clock::now();
        try {
            if(!target.tiny_dense) target.tiny_dense=std::make_shared<TinyDenseSolver>();
            dense_attempt=target.tiny_dense->solve(coarse,target.coarse_rhs.data(),target.coarse_solution.data());
        } catch(const std::exception& error) {
            dense_attempt={{"attempted",true},{"converged",false},{"failure_reason","dense_exception"},{"error",error.what()}};
            target.tiny_dense.reset();
        }
        dense_attempt["dense_wall_ms"]=collect_timing?gipc::Json(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()):gipc::Json(nullptr);
        if(dense_attempt.value("converged",false))
        {
            ++target.tiny_dense_successes;
            target.prolonged.resize(fine_dofs);
            prolongate_mixed<<<(fine.block_rows()+kThreads-1)/kThreads,kThreads>>>(
                target.coarse_solution.data(),target.prolonged.data(),mapping.fine_to_coarse,
                mapping.coarse_block_bases,mapping.basis_masks,mapping.rest_positions,prefix_blocks,fine.block_rows());
            target.fine_ax.resize(fine_dofs); target.fine_residual.resize(fine_dofs);
            gipc::Spmv spmv;
            spmv.warp_reduce_sym_spmv(1.0,const_cast<Eigen::Matrix3d*>(fine.block_values()),
                const_cast<int*>(fine.block_row_indices()),const_cast<int*>(fine.block_col_indices()),fine.h_unique_key_number,
                cudatool::CDenseVectorView<double>(target.prolonged.data(),fine_dofs),0.0,
                cudatool::DenseVectorView<double>(target.fine_ax.data(),fine_dofs));
            form_residual<<<(fine_dofs+kThreads-1)/kThreads,kThreads>>>(fine_rhs,target.fine_ax.data(),target.fine_residual.data(),fine_dofs);
            const double initial2=device_dot(fine_rhs,fine_rhs,fine_dofs);
            const double final2=device_dot(target.fine_residual.data(),target.fine_residual.data(),fine_dofs);
            dense_attempt["fine_initial_residual_norm"]=std::sqrt(initial2);
            dense_attempt["fine_prolongated_residual_norm"]=std::sqrt(std::max(0.0,final2));
            dense_attempt["fine_residual_ratio"]=initial2>0?std::sqrt(final2/initial2):0.0;
            dense_attempt["rhs_dot_direction"]=device_dot(fine_rhs,target.prolonged.data(),fine_dofs);
            dense_attempt["controls_solver"]=false;
            return dense_attempt;
        }
        ++target.tiny_dense_rejections;
        target.coarse_solution.reset_zero();
    }
    const int storage_dofs=direct_mas_apply && target.use_coarse_mas32
        ?3*((blocks+31)/32*32):dofs;
    target.coarse_r.resize(storage_dofs); target.coarse_z.resize(storage_dofs);
    // resize may reuse a formerly larger system: never retain stale real RHS
    // values in nodes that have become isolated padding after a shrink.
    if(storage_dofs>dofs) CUDA_SAFE_CALL(cudaMemset(target.coarse_r.data()+dofs,
                                                  0,(storage_dofs-dofs)*sizeof(double)));
    target.coarse_p.resize(dofs); target.coarse_ap.resize(dofs);
    target.inverse_diagonal.resize(blocks); target.diagonal_found.resize(blocks);
    CUDA_SAFE_CALL(cudaMemcpy(target.coarse_r.data(),target.coarse_rhs.data(),
                              dofs*sizeof(double),cudaMemcpyDeviceToDevice));
    initialize_inverse_diagonal<<<(blocks+kThreads-1)/kThreads,kThreads>>>(
        target.inverse_diagonal.data(),target.diagonal_found.data(),blocks);
    extract_inverse_diagonal<<<(unique+kThreads-1)/kThreads,kThreads>>>(
        coarse.block_values(),coarse.block_row_indices(),coarse.block_col_indices(),
        target.inverse_diagonal.data(),target.diagonal_found.data(),
        target.invalid_entries.data(),unique);
    const int found=thrust::reduce(thrust::device_ptr<int>(target.diagonal_found.data()),
        thrust::device_ptr<int>(target.diagonal_found.data())+blocks,0);
    if(found!=blocks)
        return {{"attempted",true},{"converged",false},{"failure_reason","missing_or_singular_diagonal"},
                {"diagonal_blocks_found",found},{"diagonal_blocks_expected",blocks}};
    std::shared_ptr<RuntimeCoarseMas> mas;
    gipc::Json mas_diagnostics=nullptr;
    double mas_setup_ms=0,mas_validation_ms=0;
    if(target.use_coarse_mas32)
    {
        ++target.mas_attempts;
        const auto begin=std::chrono::steady_clock::now();
        try
        {
            mas=target.mas_reuse_enabled?target.runtime_mas:nullptr;
            if(!mas) mas=std::make_shared<RuntimeCoarseMas>();
            mas->setup(coarse,target.mas_reuse_enabled,
                       mapping.paper_affine_basis && !target.use_factorized_mas32,
                       target.use_factorized_mas32);
            if(target.mas_reuse_enabled) target.runtime_mas=mas;
            if(mas->last_reused) ++target.mas_graph_reuses;
        }
        catch(const std::exception& error)
        {
            ++target.mas_local_failures;
            target.runtime_mas.reset();
            return {{"attempted",true},{"converged",false},{"failure_reason","mas_setup_failed"},
                    {"preconditioner","mas32"},{"setup_error",error.what()}};
        }
        mas_setup_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        target.total_mas_setup_ms+=mas_setup_ms;
    }
    int mas_applications=0;
    const auto apply_preconditioner=[&](const double* input,double* output) {
        if(mas) { ++mas_applications; mas->apply(input,output,direct_mas_apply); }
        else apply_diagonal<<<(blocks+kThreads-1)/kThreads,kThreads>>>(target.inverse_diagonal.data(),input,output,blocks);
    };
    apply_preconditioner(target.coarse_r.data(),target.coarse_z.data());
    // Local inverse diagnostics are a development gate, not an AGIPC solve
    // stage. Production still checks positive r^T M^-1 r, curvature, the
    // recursive residual, and the explicit true residual below.
    if(mas && target.mas_validation!="off")
    {
        const auto begin=std::chrono::steady_clock::now();
        mas_diagnostics=mas->diagnostics(target.mas_validation);
        mas_validation_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        target.total_mas_validation_ms+=mas_validation_ms;
        if(!mas_diagnostics.value("passed",false))
        {
            ++target.mas_local_failures;
            if(target.mas_failure_samples.size()<4)
            {
                const auto probe_started=std::chrono::steady_clock::now();
                auto paired=mas_diagnostics;
                if(target.mas_validation=="gpu")
                {
                    const auto cpu=mas->diagnostics("cpu");
                    paired["cpu_gpu_agreement"]=gpu_mas32::local_diagnostics_agree(paired,cpu);
                    paired["cpu_reference"]=cpu;
                }
                else if(target.mas_validation=="cpu")
                {
                    const auto gpu=mas->diagnostics("gpu");
                    paired["cpu_gpu_agreement"]=gpu_mas32::local_diagnostics_agree(gpu,paired);
                    paired["gpu_reference"]=gpu;
                }
                const double probe_ms=std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-probe_started).count();
                target.total_mas_validation_ms+=probe_ms;mas_validation_ms+=probe_ms;
                mas_diagnostics=std::move(paired);
                target.mas_failure_samples.push_back({{"update_index",target.updates+1},
                    {"coarse_block_nodes",blocks},{"coarse_unique_blocks",unique},
                    {"mas_graph_reused",mas->last_reused},{"validation_mode",target.mas_validation},
                    {"rejection_probe_wall_ms",probe_ms},{"diagnostic",mas_diagnostics}});
            }
            return {{"attempted",true},{"converged",false},{"failure_reason","mas_local_diagnostics_failed"},
                    {"preconditioner","mas32"},{"mas_local_diagnostics",mas_diagnostics},
                    {"max_iterations",std::min(512,std::max(1,dofs))},{"relative_tolerance",1e-3},
                    {"mas_setup_wall_ms",mas_setup_ms},{"mas_validation_wall_ms",mas_validation_ms}};
        }
    }
    const auto pcg_started=std::chrono::steady_clock::now();
    CUDA_SAFE_CALL(cudaMemcpy(target.coarse_p.data(),target.coarse_z.data(),
                              dofs*sizeof(double),cudaMemcpyDeviceToDevice));
    const auto initial_dots=device_residual_dots(target.coarse_r.data(),
                                                target.coarse_z.data(),dofs);
    const double initial2=initial_dots.rr;
    double residual2=initial2,rz=initial_dots.rz;
    const double tolerance2=1e-6*initial2;
    int iterations=0; std::string failure;
    gipc::Spmv spmv;
    const int max_iterations=std::min(512,std::max(1,dofs));
    int state_downloads=0,queued_iterations=0;
    if(coarse_pcg_batch)
    {
        auto& w=target.device_coarse_pcg;w.resize(dofs);
        auto control=device_pcg::initial(initial2,rz,max_iterations,bool(mas));
        w.control.copy_from_host(std::vector<device_pcg::Control>{control});
        while(control.status==device_pcg::Running)
        {
            const int count=std::min(coarse_pcg_batch,max_iterations-control.iterations);
            for(int queued=0;queued<count;++queued)
            {
                spmv.warp_reduce_sym_spmv(1.0,coarse.block_values(),coarse.block_row_indices(),
                    coarse.block_col_indices(),unique,
                    cudatool::CDenseVectorView<double>(target.coarse_p.data(),dofs),0.0,
                    cudatool::DenseVectorView<double>(target.coarse_ap.data(),dofs));
                device_pcg::reduce(w,target.coarse_p.data(),target.coarse_ap.data(),dofs,false);
                device_pcg::update_x_r<<<(dofs+kThreads-1)/kThreads,kThreads>>>(target.coarse_solution.data(),
                    target.coarse_r.data(),target.coarse_p.data(),target.coarse_ap.data(),dofs,w.control.data());
                apply_preconditioner(target.coarse_r.data(),target.coarse_z.data());
                device_pcg::reduce(w,target.coarse_r.data(),target.coarse_z.data(),dofs,true);
                device_pcg::update_p<<<(dofs+kThreads-1)/kThreads,kThreads>>>(target.coarse_p.data(),
                    target.coarse_z.data(),dofs,w.control.data());
                ++queued_iterations;
            }
            CUDA_SAFE_CALL(cudaMemcpy(&control,w.control.data(),sizeof(control),cudaMemcpyDeviceToHost));
            ++state_downloads;
        }
        iterations=control.iterations;residual2=control.rr;rz=control.rz;
        failure=device_pcg::failure(control.status);
    }
    else
    for(;iterations<max_iterations && residual2>tolerance2;)
    {
        if(mas && (!std::isfinite(rz) || rz<=0))
        { failure="invalid_preconditioned_residual"; break; }
        spmv.warp_reduce_sym_spmv(1.0,coarse.block_values(),coarse.block_row_indices(),
            coarse.block_col_indices(),unique,
            cudatool::CDenseVectorView<double>(target.coarse_p.data(),dofs),0.0,
            cudatool::DenseVectorView<double>(target.coarse_ap.data(),dofs));
        ++queued_iterations; ++state_downloads;
        const double curvature=device_dot(target.coarse_p.data(),target.coarse_ap.data(),dofs);
        if(!std::isfinite(curvature) || curvature<=0 || !std::isfinite(rz))
        { failure=!std::isfinite(curvature)?"nonfinite_curvature":"nonpositive_curvature"; break; }
        const double alpha=rz/curvature;
        update_x_r<<<(dofs+kThreads-1)/kThreads,kThreads>>>(target.coarse_solution.data(),
            target.coarse_r.data(),target.coarse_p.data(),target.coarse_ap.data(),alpha,dofs);
        ++iterations;
        apply_preconditioner(target.coarse_r.data(),target.coarse_z.data());
        ++state_downloads;
        const auto next_dots=device_residual_dots(target.coarse_r.data(),
                                                  target.coarse_z.data(),dofs);
        residual2=next_dots.rr;
        if(!std::isfinite(residual2)) { failure="nonfinite_residual"; break; }
        if(residual2<=tolerance2) break;
        const double next_rz=next_dots.rz;
        if(!std::isfinite(next_rz) || fabs(rz)<1e-30 || (mas && next_rz<=0)) { failure="invalid_preconditioned_residual"; break; }
        update_p<<<(dofs+kThreads-1)/kThreads,kThreads>>>(target.coarse_p.data(),
            target.coarse_z.data(),next_rz/rz,dofs); rz=next_rz;
    }
    const double recursive2=residual2;
    bool converged=residual2<=tolerance2;
    if(!converged && failure.empty()) failure="iteration_cap";
    if(mas)
    {
        spmv.warp_reduce_sym_spmv(1.0,coarse.block_values(),coarse.block_row_indices(),
            coarse.block_col_indices(),unique,
            cudatool::CDenseVectorView<double>(target.coarse_solution.data(),dofs),0.0,
            cudatool::DenseVectorView<double>(target.coarse_ap.data(),dofs));
        form_residual<<<(dofs+kThreads-1)/kThreads,kThreads>>>(target.coarse_rhs.data(),
            target.coarse_ap.data(),target.coarse_r.data(),dofs);
        residual2=device_dot(target.coarse_r.data(),target.coarse_r.data(),dofs);
        converged=failure.empty() && std::isfinite(residual2)
            && residual2<=tolerance2*(1.0+kResidualGrowthTolerance)*(1.0+kResidualGrowthTolerance);
        if(!converged && failure.empty()) failure="true_residual_not_converged";
    }
    const double pcg_wall_ms=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-pcg_started).count();
    target.prolonged.resize(fine_dofs);
    prolongate_mixed<<<(fine.block_rows()+kThreads-1)/kThreads,kThreads>>>(
        target.coarse_solution.data(),target.prolonged.data(),mapping.fine_to_coarse,
        mapping.coarse_block_bases,mapping.basis_masks,mapping.rest_positions,
        prefix_blocks,fine.block_rows());
    target.fine_ax.resize(fine_dofs); target.fine_residual.resize(fine_dofs);
    spmv.warp_reduce_sym_spmv(1.0,
        const_cast<Eigen::Matrix3d*>(fine.block_values()),
        const_cast<int*>(fine.block_row_indices()),
        const_cast<int*>(fine.block_col_indices()),fine.h_unique_key_number,
        cudatool::CDenseVectorView<double>(target.prolonged.data(),fine_dofs),0.0,
        cudatool::DenseVectorView<double>(target.fine_ax.data(),fine_dofs));
    form_residual<<<(fine_dofs+kThreads-1)/kThreads,kThreads>>>(fine_rhs,
        target.fine_ax.data(),target.fine_residual.data(),fine_dofs);
    const double fine_initial2=device_dot(fine_rhs,fine_rhs,fine_dofs);
    const double fine_final2=device_dot(target.fine_residual.data(),target.fine_residual.data(),fine_dofs);
    const double rhs_dot_direction=device_dot(fine_rhs,target.prolonged.data(),fine_dofs);
    return {{"attempted",true},{"converged",converged},{"failure_reason",failure},
            {"solver","pcg"},{"tiny_dense_attempt",dense_attempt},
            {"preconditioner",mas?(target.use_factorized_mas32?"mas32-factor":"mas32"):"block_jacobi"},
            {"mas_local_diagnostics",mas_diagnostics},{"mas_setup_wall_ms",mas_setup_ms},
            {"mas_validation_wall_ms",mas_validation_ms},{"true_residual_checked",bool(mas)},
            {"pcg_wall_ms",pcg_wall_ms},
            {"coarse_pcg_batch",coarse_pcg_batch},
            {"pcg_iteration_state_downloads",state_downloads},
            {"pcg_queued_iterations",queued_iterations},
            {"pcg_masked_iterations",std::max(0,queued_iterations-iterations)},
            {"device_pcg_workspace_bytes",target.device_coarse_pcg.bytes()},
            {"mas_apply_mode",direct_mas_apply?"direct":"copy"},
            {"mas_applications",mas_applications},
            {"mas_adapter_d2d_copies",direct_mas_apply?0:2*mas_applications},
            {"pcg_logical_dofs",dofs},{"pcg_storage_dofs",storage_dofs},
            {"mas_validation_mode",target.mas_validation},{"mas_graph_reused",mas && mas->last_reused},
            {"mas_nullspace_regularization",mas && mas->nullspace_regularization},
            {"recursive_final_residual_norm",std::sqrt(std::max(0.0,recursive2))},
            {"iterations",iterations},{"max_iterations",max_iterations},
            {"relative_tolerance",1e-3},{"initial_residual_norm",std::sqrt(initial2)},
            {"final_residual_norm",std::sqrt(std::max(0.0,residual2))},
            {"fine_initial_residual_norm",std::sqrt(fine_initial2)},
            {"fine_prolongated_residual_norm",std::sqrt(std::max(0.0,fine_final2))},
            {"fine_residual_ratio",fine_initial2>0?std::sqrt(fine_final2/fine_initial2):0.0},
            {"rhs_dot_direction",rhs_dot_direction},{"controls_solver",false}};
}

gipc::Json assemble_shadow(GalerkinState& target,
                           const GIPCTripletMatrix& fine_matrix,
                           const double* fine_rhs,
                           std::size_t fine_rhs_dofs,
                           MappingDeviceView mapping)
{
    target.candidate_ready=false;
    target.candidate_failure_reason="mapping_incomplete";
    target.candidate_dofs=0;
    target.candidate_iterations=0;
    if(!mapping.ready) return nullptr;
    if(fine_rhs_dofs%3!=0)
        throw std::runtime_error("Gate C requires a 3-DoF block-aligned right-hand side");
    const int fine_blocks=static_cast<int>(fine_rhs_dofs/3);
    const int prefix_blocks=fine_blocks-mapping.fine_nodes;
    if(prefix_blocks<0 || fine_matrix.block_rows()!=fine_blocks
       || fine_matrix.block_cols()!=fine_blocks)
        throw std::runtime_error("Gate C fine matrix/mapping domain mismatch");
    if(mapping.coarse_nodes<0 || (mapping.fine_nodes>0 && mapping.coarse_nodes==0))
        throw std::runtime_error("Gate C mapping has no coarse owner");
    const int fine_unique=fine_matrix.h_unique_key_number;
    if(fine_unique>std::numeric_limits<int>::max()/16)
        throw std::runtime_error("AGIPC expanded counts exceed signed index capacity");
    if(fine_unique<1)
    {
        target.candidate_failure_reason="fine_matrix_empty";
        return nullptr;
    }
    const int coarse_blocks=prefix_blocks+mapping.coarse_block_nodes;

    cudaEvent_t start=nullptr,assembly_end=nullptr,coarse_end=nullptr,end=nullptr;
    if(collect_timing) CUDA_SAFE_CALL(cudaEventCreate(&start));
    if(collect_timing) CUDA_SAFE_CALL(cudaEventCreate(&assembly_end));
    if(collect_timing) CUDA_SAFE_CALL(cudaEventCreate(&coarse_end));
    if(collect_timing) CUDA_SAFE_CALL(cudaEventCreate(&end));
    if(collect_timing) CUDA_SAFE_CALL(cudaEventRecord(start));
    if(!target.coarse_matrix)
    {
        target.coarse_matrix=std::make_unique<GIPCTripletMatrix>();
        target.coarse_matrix->init_var();
    }
    target.expansion_counts.resize(fine_unique);
    target.expansion_offsets.resize(fine_unique);
    count_expanded_blocks<<<(fine_unique+kThreads-1)/kThreads,kThreads>>>(
        fine_matrix.block_row_indices(),fine_matrix.block_col_indices(),
        target.expansion_counts.data(),mapping.fine_to_coarse,mapping.coarse_block_bases,
        mapping.basis_masks,mapping.rest_positions,prefix_blocks,fine_unique);
    thrust::exclusive_scan(thrust::device_ptr<int>(target.expansion_counts.data()),
        thrust::device_ptr<int>(target.expansion_counts.data())+fine_unique,
        thrust::device_ptr<int>(target.expansion_offsets.data()));
    const int expanded=thrust::reduce(thrust::device_ptr<int>(target.expansion_counts.data()),
        thrust::device_ptr<int>(target.expansion_counts.data())+fine_unique,0);
    auto& coarse=*target.coarse_matrix;
    if(expanded < 0 || expanded > std::numeric_limits<int>::max()/2)
        throw std::runtime_error("AGIPC expanded matrix exceeds signed index capacity");
    std::size_t memory_free=0,memory_total=0;
    if(collect_full_diagnostics) CUDA_SAFE_CALL(cudaMemGetInfo(&memory_free,&memory_total));
    const std::size_t triplet_required=static_cast<std::size_t>(expanded)*2
        *(sizeof(Eigen::Matrix3d)+2*sizeof(int));
    coarse.resize(coarse_blocks,coarse_blocks,static_cast<std::size_t>(expanded)*2);
    coarse.resize_collision_hash_size(expanded);
    coarse.global_triplet_offset=expanded;
    target.coarse_rhs.resize(static_cast<std::size_t>(coarse_blocks)*3);
    target.coarse_rhs.reset_zero();
    target.invalid_entries.resize(1);
    target.invalid_entries.reset_zero();

    emit_expanded_blocks<<<(fine_unique+kThreads-1)/kThreads,kThreads>>>(
        fine_matrix.block_values(),fine_matrix.block_row_indices(),
        fine_matrix.block_col_indices(),coarse.block_values(),
        coarse.block_row_indices(),coarse.block_col_indices(),
        target.expansion_offsets.data(),mapping.fine_to_coarse,
        mapping.coarse_block_bases,mapping.basis_masks,mapping.rest_positions,
        prefix_blocks,fine_blocks,target.invalid_entries.data(),fine_unique);
    reduce_mixed_rhs<<<(fine_blocks+kThreads-1)/kThreads,kThreads>>>(
        fine_rhs,target.coarse_rhs.data(),mapping.fine_to_coarse,
        mapping.coarse_block_bases,mapping.basis_masks,mapping.rest_positions,
        prefix_blocks,target.invalid_entries.data(),fine_blocks);
    CUDA_SAFE_CALL(cudaGetLastError());

    gipc::Converter converter;
    converter.convert(coarse,0,expanded,expanded);
    std::vector<int> invalid;
    target.invalid_entries.copy_to_host(invalid);
    if(invalid[0]!=0)
        throw std::runtime_error("Gate C encountered invalid matrix or right-hand-side entries");
    if(collect_timing) CUDA_SAFE_CALL(cudaEventRecord(assembly_end));
    auto coarse_solve=solve_coarse_shadow(target,fine_matrix,fine_rhs,
        static_cast<int>(fine_rhs_dofs),mapping,prefix_blocks);
    if(collect_timing) CUDA_SAFE_CALL(cudaEventRecord(coarse_end));
    target.coarse_solve_distribution.push_back({{"update_index",target.updates+1},
        {"frame",target.direction_frame},
        {"frame_newton_ordinal",target.direction_newton_ordinal},
        {"active_self_or_body_pairs",target.direction_pairs},
        {"active_ground_pairs",target.direction_ground_pairs},
        {"has_active_contact",target.direction_pairs>0 || target.direction_ground_pairs>0},
        {"fine_block_nodes",fine_blocks},
        {"fine_dofs",fine_rhs_dofs},
        {"block_nodes",target.coarse_matrix->block_rows()},
        {"dofs",3*target.coarse_matrix->block_rows()},
        {"coarse_to_fine_block_ratio",fine_blocks>0
            ? double(target.coarse_matrix->block_rows())/double(fine_blocks) : 0.0},
        {"unique_blocks",target.coarse_matrix->h_unique_key_number},
        {"stored_scalar_entries",9*std::size_t(target.coarse_matrix->h_unique_key_number)},
        {"dense_matrix_bytes_if_selected",sizeof(double)*std::size_t(3*target.coarse_matrix->block_rows())*std::size_t(3*target.coarse_matrix->block_rows())},
        {"solver",coarse_solve.value("solver",std::string("pcg"))},
        {"coarse_iterations",coarse_solve.value("iterations",0)},
        {"pcg_queued_iterations",coarse_solve.value("pcg_queued_iterations",0)},
        {"pcg_masked_iterations",coarse_solve.value("pcg_masked_iterations",0)},
        {"pcg_iteration_state_downloads",coarse_solve.value("pcg_iteration_state_downloads",0)},
        {"mas_setup_wall_ms",coarse_solve.value("mas_setup_wall_ms",0.0)},
        {"pcg_wall_ms",coarse_solve.value("pcg_wall_ms",0.0)},
        {"initial_residual_norm",coarse_solve.value("initial_residual_norm",0.0)},
        {"final_residual_norm",coarse_solve.value("final_residual_norm",0.0)},
        {"fine_prolongated_residual_ratio",coarse_solve.value("fine_residual_ratio",0.0)},
        {"converged",coarse_solve.value("converged",false)},
        {"failure_reason",coarse_solve.value("failure_reason",std::string{})},
        {"dense_allocated_bytes",coarse_solve.value("tracked_allocated_bytes",std::size_t(0))}});
    auto post_correction=coarse_solve["converged"].get<bool>()
        ? post_correct_shadow(target,fine_matrix,fine_rhs,static_cast<int>(fine_rhs_dofs))
        : gipc::Json{{"attempted",false},{"stop_reason","coarse_solve_failed"},
                     {"controls_solver",false}};
    target.candidate_ready=post_correction.value("candidate_valid",false);
    if(target.candidate_ready)
        target.candidate_failure_reason.clear();
    else if(!coarse_solve.value("converged",false))
        target.candidate_failure_reason="coarse_"
            +coarse_solve.value("failure_reason",std::string{"solve_failed"});
    else
    {
        const auto stop=post_correction.value("stop_reason",std::string{"candidate_invalid"});
        const auto reduction=post_correction.value("residual_reduction_ratio",1.0);
        const auto rhs_dot=post_correction.value("rhs_dot_direction",0.0);
        if(stop!="residual_tolerance" && stop!="iteration_cap"
           && stop!="cap_zero_ablation")
            target.candidate_failure_reason="post_"+stop;
        else if(!std::isfinite(reduction) || reduction>1.0+1e-12)
            target.candidate_failure_reason="post_residual_not_reduced";
        else if(!std::isfinite(rhs_dot) || rhs_dot<=0)
            target.candidate_failure_reason="post_non_descent_direction";
        else
            target.candidate_failure_reason="post_candidate_invalid";
    }
    target.candidate_dofs=fine_rhs_dofs;
    target.candidate_iterations=coarse_solve.value("iterations",0)
        +post_correction.value("iterations",0);
    auto& solve_record=target.coarse_solve_distribution.back();
    solve_record["post_iterations"]=post_correction.value("iterations",0);
    solve_record["candidate_iterations"]=target.candidate_iterations;
    solve_record["post_stop_reason"]=post_correction.value("stop_reason",std::string{});
    solve_record["post_initial_residual_norm"]=post_correction.value("initial_residual_norm",0.0);
    solve_record["post_final_residual_norm"]=post_correction.value("final_residual_norm",0.0);
    solve_record["post_residual_reduction_ratio"]=post_correction.value("residual_reduction_ratio",0.0);
    solve_record["candidate_ready"]=target.candidate_ready;
    solve_record["candidate_failure_reason"]=target.candidate_failure_reason;
    if(collect_timing) CUDA_SAFE_CALL(cudaEventRecord(end));
    if(collect_timing) CUDA_SAFE_CALL(cudaEventSynchronize(end));
    float elapsed=0,assembly_elapsed=0,coarse_elapsed=0,post_elapsed=0;
    if(collect_timing) CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed,start,end));
    if(collect_timing) CUDA_SAFE_CALL(cudaEventElapsedTime(&assembly_elapsed,start,assembly_end));
    if(collect_timing) CUDA_SAFE_CALL(cudaEventElapsedTime(&coarse_elapsed,assembly_end,coarse_end));
    if(collect_timing) CUDA_SAFE_CALL(cudaEventElapsedTime(&post_elapsed,coarse_end,end));
    if(collect_timing) CUDA_SAFE_CALL(cudaEventDestroy(start));
    if(collect_timing) CUDA_SAFE_CALL(cudaEventDestroy(assembly_end));
    if(collect_timing) CUDA_SAFE_CALL(cudaEventDestroy(coarse_end));
    if(collect_timing) CUDA_SAFE_CALL(cudaEventDestroy(end));
    coarse_solve["solve_ms"]=collect_timing?gipc::Json(coarse_elapsed):gipc::Json(nullptr);
    post_correction["solve_ms"]=collect_timing?gipc::Json(post_elapsed):gipc::Json(nullptr);
    target.invalid_entries.copy_to_host(invalid);
    ++target.updates;
    target.total_ms+=elapsed;
    target.total_assembly_ms+=assembly_elapsed;
    target.total_coarse_solve_ms+=coarse_elapsed;
    target.total_post_correction_ms+=post_elapsed;
    const auto capacity_bytes=[](const auto& buffer) {
        return buffer.capacity()*sizeof(typename std::decay_t<decltype(buffer)>::value_type);
    };
    const auto matrix_bytes=capacity_bytes(coarse.m_block_values)
        +capacity_bytes(coarse.m_block_row_indices)+capacity_bytes(coarse.m_block_col_indices);
    const auto sort_bytes=capacity_bytes(coarse.m_block_hash_value)
        +capacity_bytes(coarse.m_block_sort_hash_value)+capacity_bytes(coarse.m_block_index)
        +capacity_bytes(coarse.m_block_sort_index)+capacity_bytes(coarse.m_block_temp_buffer);
    const auto vector_bytes=capacity_bytes(target.coarse_rhs)+capacity_bytes(target.coarse_solution)
        +capacity_bytes(target.coarse_r)+capacity_bytes(target.coarse_z)
        +capacity_bytes(target.coarse_p)+capacity_bytes(target.coarse_ap)
        +capacity_bytes(target.expansion_counts)+capacity_bytes(target.expansion_offsets);
    const auto device_pcg_bytes=target.device_coarse_pcg.bytes();
    const auto tracked_bytes=matrix_bytes+sort_bytes+vector_bytes+device_pcg_bytes;
    target.tracked_coarse_peak_bytes=std::max(target.tracked_coarse_peak_bytes,tracked_bytes);
    target.last={{"stage","galerkin_mixed_shadow"},{"formula","Ac=PT*A*P; bc=PT*b"},
                 {"fine_block_nodes",fine_blocks},{"prefix_identity_blocks",prefix_blocks},
                 {"fine_fem_nodes",mapping.fine_nodes},{"coarse_fem_nodes",mapping.coarse_nodes},
                 {"translational_nodes",mapping.translational_nodes},
                 {"affine_nodes",mapping.affine_nodes},
                 {"coarse_block_nodes",coarse_blocks},{"fine_unique_blocks",fine_unique},
                 {"expanded_blocks_before_reduction",expanded},
                 {"coarse_unique_blocks",coarse.h_unique_key_number},
                 {"invalid_entries",invalid[0]},
                 {"collect_timing",collect_timing},
                 {"shadow_pipeline_ms",collect_timing?gipc::Json(elapsed):gipc::Json(nullptr)},
                 {"memory",{{"coarse_triplets_required_bytes",triplet_required},
                    {"coarse_triplets_capacity_bytes",matrix_bytes},
                    {"coarse_hash_sort_capacity_bytes",sort_bytes},
                    {"coarse_vectors_expansion_capacity_bytes",vector_bytes},
                    {"device_pcg_workspace_capacity_bytes",device_pcg_bytes},
                    {"tracked_coarse_allocated_bytes",tracked_bytes},
                    {"tracked_coarse_peak_bytes",target.tracked_coarse_peak_bytes},
                    {"device_free_before_expansion_bytes",collect_full_diagnostics?gipc::Json(memory_free):gipc::Json(nullptr)},
                    {"device_total_bytes",collect_full_diagnostics?gipc::Json(memory_total):gipc::Json(nullptr)},
                    {"scope","coarse matrix/hash/vector buffers; excludes MAS and converter scratch"}}},
                 {"stage_timing_ms",collect_timing?gipc::Json{{"assembly",assembly_elapsed},
                     {"coarse_solve",coarse_elapsed},{"post_correction",post_elapsed}}:gipc::Json(nullptr)},
                 {"coarse_solve",coarse_solve},{"post_correction",post_correction},
                 {"controls_solver",false}};
    const auto frozen=freeze_coarse_system(target);
    if(!frozen.is_null()) target.last["coarse_system_freeze"]=frozen;
    return target.last;
}

gipc::Json adopt_candidate(GalerkinState& target,double* destination,
                           std::size_t destination_dofs)
{
    if(!target.adoption_enabled)
        return {{"attempted",false},{"adopted",false},{"reason","disabled"}};
    ++target.adoption_attempts;
    if(!target.candidate_ready || target.candidate_dofs!=destination_dofs)
    {
        ++target.fallbacks;
        const std::string reason=target.candidate_ready
            ? "dof_mismatch"
            : (target.candidate_failure_reason.empty()
                ? "candidate_gate_failed" : target.candidate_failure_reason);
        target.fallback_reason_counts[reason]=
            target.fallback_reason_counts.value(reason,std::size_t{0})+1;
        target.last_adoption={{"attempted",true},{"adopted",false},
            {"reason",reason},
            {"candidate_dofs",target.candidate_dofs},{"destination_dofs",destination_dofs}};
        return target.last_adoption;
    }
    CUDA_SAFE_CALL(cudaMemcpy(destination,target.fine_solution.data(),
                              destination_dofs*sizeof(double),cudaMemcpyDeviceToDevice));
    ++target.adoptions;
    target.last_adoption={{"attempted",true},{"adopted",true},{"reason","candidate_valid"},
                          {"iterations",target.candidate_iterations},
                          {"candidate_dofs",target.candidate_dofs}};
    return target.last_adoption;
}

}

#include "agipc_device_pcg_tests.inl"

gipc::Json benchmark_production_coarse(const GIPCTripletMatrix& matrix,
                                       const double* device_rhs,int repetitions)
{
    const int blocks=matrix.block_rows(),dofs=3*blocks,unique=matrix.h_unique_key_number;
    if(blocks<1 || unique<1 || repetitions<1 || repetitions>16)
        throw std::runtime_error("Invalid production coarse benchmark dimensions");
    GalerkinState target;
    target.coarse_matrix=std::make_unique<GIPCTripletMatrix>();
    auto& coarse=*target.coarse_matrix;
    coarse.init_var(); coarse.reshape(blocks,blocks);
    coarse.h_unique_key_number=unique;
    coarse.m_block_values.resize(unique);
    coarse.m_block_row_indices.resize(unique);
    coarse.m_block_col_indices.resize(unique);
    CUDA_SAFE_CALL(cudaMemcpy(coarse.block_values(),matrix.block_values(),
                              unique*sizeof(Eigen::Matrix3d),cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(coarse.block_row_indices(),matrix.block_row_indices(),
                              unique*sizeof(int),cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(coarse.block_col_indices(),matrix.block_col_indices(),
                              unique*sizeof(int),cudaMemcpyDeviceToDevice));
    target.coarse_rhs.resize(dofs);
    CUDA_SAFE_CALL(cudaMemcpy(target.coarse_rhs.data(),device_rhs,
                              dofs*sizeof(double),cudaMemcpyDeviceToDevice));
    target.invalid_entries.resize(1); target.invalid_entries.reset_zero();
    target.tiny_dense_max_dofs=state.tiny_dense_max_dofs;
    target.use_coarse_mas32=true;
    target.mas_validation="gpu";
    target.mas_reuse_enabled=true;
    gipc::Json runs=gipc::Json::array();
    std::vector<double> first_solution;
    bool passed=true;
    for(int repeat=0;repeat<repetitions;++repeat)
    {
        const auto started=std::chrono::steady_clock::now();
        // Every fine block is an identity-prefix block, so prolongation is an
        // identity map. The timed solve is the production coarse path itself.
        auto result=solve_coarse_shadow(target,matrix,device_rhs,dofs,
                                        MappingDeviceView{},blocks);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        const double wall_ms=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-started).count();
        std::vector<double> solution;
        target.coarse_solution.copy_to_host(solution);
        double difference2=0,reference2=0;
        if(repeat==0) first_solution=solution;
        for(int i=0;i<dofs;++i)
        {
            difference2+=(solution[i]-first_solution[i])*(solution[i]-first_solution[i]);
            reference2+=first_solution[i]*first_solution[i];
        }
        const double relative_difference=std::sqrt(difference2/std::max(reference2,1e-300));
        const double initial=result.value("initial_residual_norm",0.0);
        const double final=result.value("final_residual_norm",
                                         std::numeric_limits<double>::infinity());
        const bool valid=result.value("converged",false)
            && result.value("true_residual_checked",false)
            && result.value("failure_reason",std::string{}).empty()
            && std::isfinite(final) && final<=initial*1e-3*(1.0+2e-6)
            && std::isfinite(relative_difference);
        passed&=valid;
        runs.push_back({{"repeat",repeat},{"wall_ms",wall_ms},
            {"solver",result.value("solver",std::string("pcg"))},
            {"tracked_dense_bytes",result.value("tracked_allocated_bytes",std::size_t(0))},
            {"mas_setup_wall_ms",result.value("mas_setup_wall_ms",0.0)},
            {"mas_validation_wall_ms",result.value("mas_validation_wall_ms",0.0)},
            {"pcg_wall_ms",result.value("pcg_wall_ms",0.0)},
            {"iterations",result.value("iterations",0)},
            {"mas_apply_mode",result.value("mas_apply_mode",std::string{})},
            {"coarse_pcg_batch",result.value("coarse_pcg_batch",0)},
            {"pcg_iteration_state_downloads",result.value("pcg_iteration_state_downloads",0)},
            {"pcg_queued_iterations",result.value("pcg_queued_iterations",0)},
            {"pcg_masked_iterations",result.value("pcg_masked_iterations",0)},
            {"device_pcg_workspace_bytes",result.value("device_pcg_workspace_bytes",std::size_t(0))},
            {"mas_applications",result.value("mas_applications",0)},
            {"mas_adapter_d2d_copies",result.value("mas_adapter_d2d_copies",0)},
            {"pcg_storage_dofs",result.value("pcg_storage_dofs",0)},
            {"mas_graph_reused",result.value("mas_graph_reused",false)},
            {"initial_residual_norm",initial},{"final_residual_norm",final},
            {"relative_solution_difference_from_first",relative_difference},
            {"failure_reason",result.value("failure_reason",std::string{})},
            {"passed",valid}});
    }
    return {{"test","agipc_production_coarse_fixed_input"},
            {"tiny_dense_max_dofs",target.tiny_dense_max_dofs},
            {"first_solution_for_small_system",dofs<=96?gipc::Json(first_solution):gipc::Json(nullptr)},
            {"first_solution",first_solution},{"mas_apply_mode",direct_mas_apply?"direct":"copy"},
            {"coarse_block_nodes",blocks},{"coarse_unique_blocks",unique},
            {"preconditioner","mas32"},{"validation","gpu"},
            {"precise_graph_reuse",true},{"runs",runs},{"passed",passed},
            {"timing_scope","production solve_coarse_shadow; setup, validation, PCG and identity fine metrics"},
            {"performance_claim",false}};
}

void configure_coarse_pcg_batch(int batch) { coarse_pcg_batch=batch; }

void configure_direct_mas_apply(bool enabled) { direct_mas_apply=enabled; }

gipc::Json mas_apply_self_test()
{
    RuntimeCoarseMas adapter;
    cudatool::CudaDeviceBuffer<double> input,output,reference;
    gipc::Json cases=gipc::Json::array();
    for(int blocks:{65,33,33,32,1,65})
    {
        GIPCTripletMatrix matrix; matrix.init_var(); matrix.reshape(blocks,blocks);
        std::vector<Eigen::Matrix3d> values;
        std::vector<int> rows,cols;
        for(int i=0;i<blocks;++i)
        {
            rows.push_back(i);cols.push_back(i);
            values.push_back((4.0+0.01*i)*Eigen::Matrix3d::Identity());
            if(i+1<blocks) { rows.push_back(i);cols.push_back(i+1);
                values.push_back(-0.2*Eigen::Matrix3d::Identity()); }
        }
        matrix.h_unique_key_number=static_cast<int>(values.size());
        matrix.m_block_values.copy_from_host(values);
        matrix.m_block_row_indices.copy_from_host(rows);
        matrix.m_block_col_indices.copy_from_host(cols);
        adapter.setup(matrix,true,false);
        const int dofs=3*blocks,storage=3*adapter.padded;
        for(bool zero:{false,true})
        {
            // Reusing these allocations is intentional, including a 65->33
            // shrink. The input beyond the true nodes must be reset each time.
            std::vector<double> rhs(storage+6,0.0);
            for(int i=0;i<dofs;++i) rhs[i]=zero?0.0:std::sin(0.17*i)+0.3;
            for(int i=storage;i<storage+6;++i) rhs[i]=123456.0;
            input.copy_from_host(rhs);
            output.copy_from_host(std::vector<double>(storage+6,123456.0));
            reference.resize(dofs);
            adapter.apply(input.data(),reference.data(),false);
            const auto cpu_copy=adapter.diagnostics("cpu");
            adapter.apply(input.data(),output.data(),true);
            const auto cpu_direct=adapter.diagnostics("cpu");
            std::vector<double> actual,expected,unchanged;
            output.copy_to_host(actual);reference.copy_to_host(expected);input.copy_to_host(unchanged);
            double error=0;bool padding_zero=true,sentinel=true;
            for(int i=0;i<dofs;++i) error=std::max(error,std::abs(actual[i]-expected[i]));
            for(int i=dofs;i<storage;++i) padding_zero&=actual[i]==0.0;
            for(int i=storage;i<storage+6;++i) sentinel&=actual[i]==123456.0;
            // The direct kernel and the copy-based reference may differ by one
            // FP64 rounding step because their accumulation orders differ.
            const bool passed=std::isfinite(error) && error<=1e-15 && padding_zero && sentinel
                && unchanged==rhs && cpu_copy==cpu_direct && cpu_direct.value("passed",false);
            cases.push_back({{"blocks",blocks},{"storage_dofs",storage},{"zero_rhs",zero},
                {"graph_reused",adapter.last_reused},{"max_abs_output_error",error},
                {"padding_zero",padding_zero},{"sentinel_untouched",sentinel},
                {"input_unchanged",unchanged==rhs},{"cpu_diagnostics_identical",cpu_copy==cpu_direct},{"passed",passed}});
            if(!passed) throw std::runtime_error("Direct coarse MAS apply fixture failed: "+cases.back().dump());
        }
    }
    {
        const int blocks=33;
        GIPCTripletMatrix matrix; matrix.init_var(); matrix.reshape(blocks,blocks);
        std::vector<Eigen::Matrix3d> values;
        std::vector<int> rows,cols;
        for(int i=0;i<blocks;++i)
        {
            rows.push_back(i);cols.push_back(i);
            values.push_back((4.0+0.01*i)*Eigen::Matrix3d::Identity());
            if(i+1<blocks) { rows.push_back(i);cols.push_back(i+1);
                values.push_back(-0.2*Eigen::Matrix3d::Identity()); }
        }
        matrix.h_unique_key_number=static_cast<int>(values.size());
        matrix.m_block_values.copy_from_host(values);
        matrix.m_block_row_indices.copy_from_host(rows);
        matrix.m_block_col_indices.copy_from_host(cols);
        adapter.setup(matrix,false,false,true);
        const int dofs=3*blocks,storage=3*adapter.padded;
        std::vector<double> rhs(storage+3,0.0);
        for(int i=0;i<dofs;++i) rhs[i]=std::cos(0.11*i)-0.2;
        for(int i=storage;i<storage+3;++i) rhs[i]=654321.0;
        input.copy_from_host(rhs);
        output.copy_from_host(std::vector<double>(storage+3,654321.0));
        reference.resize(dofs);
        adapter.apply(input.data(),reference.data(),false);
        adapter.apply(input.data(),output.data(),true);
        const auto diagnostics=adapter.diagnostics("gpu");
        std::vector<double> actual,expected;
        output.copy_to_host(actual);reference.copy_to_host(expected);
        double error=0;bool padding_zero=true,sentinel=true;
        for(int i=0;i<dofs;++i) error=std::max(error,std::abs(actual[i]-expected[i]));
        for(int i=dofs;i<storage;++i) padding_zero&=actual[i]==0.0;
        for(int i=storage;i<storage+3;++i) sentinel&=actual[i]==654321.0;
        const bool passed=std::isfinite(error) && error<=1e-15 && padding_zero
            && sentinel && diagnostics.value("passed",false);
        cases.push_back({{"blocks",blocks},{"storage_dofs",storage},
            {"backend","fp64_cholesky_inverse_apply"},{"max_abs_output_error",error},
            {"padding_zero",padding_zero},{"sentinel_untouched",sentinel},
            {"diagnostics",diagnostics},{"passed",passed}});
        if(!passed) throw std::runtime_error("Factorized coarse MAS apply fixture failed: "+cases.back().dump());
    }
    return {{"test","agipc_direct_mas_apply"},{"passed",true},{"cases",cases}};
}

void configure_galerkin(int fine_correction_max_iterations,
                        bool adoption_enabled,
                        std::string fallback_diagnostics_directory,
                        std::string coarse_diagnostics_directory,
                        bool use_coarse_mas32,bool use_factorized_mas32,
                        std::string mas_validation,bool mas_reuse_enabled)
{
    state.runtime_mas.reset();state.mas_graph_reuses=0;
    state.mas_validation=std::move(mas_validation);state.mas_reuse_enabled=mas_reuse_enabled;
    state.use_coarse_mas32=use_coarse_mas32;
    state.use_factorized_mas32=use_factorized_mas32;
    state.mas_attempts=state.mas_local_failures=0;
    state.mas_failure_samples=gipc::Json::array();
    state.total_mas_setup_ms=state.total_mas_validation_ms=0;
    state.post_max_iterations=std::max(0,fine_correction_max_iterations);
    state.adoption_enabled=adoption_enabled;
    state.fallback_diagnostics_directory=std::move(fallback_diagnostics_directory);
    state.frozen_quality_categories={false,false,false};
    state.direction_quality_evaluations=0;
    state.residual_guard_restores=0;
    state.frozen_fallback_samples=gipc::Json::array();
    state.coarse_diagnostics_directory=std::move(coarse_diagnostics_directory);
    state.frozen_coarse_categories={false,false,false,false,false,false,false};
    state.frozen_coarse_samples=gipc::Json::array();
}

void configure_tiny_dense_limit(int max_dofs)
{
    if(max_dofs!=0 && max_dofs!=24 && max_dofs!=48 && max_dofs!=96)
        throw std::runtime_error("Invalid tiny dense candidate boundary");
    state.tiny_dense_max_dofs=max_dofs;
    state.tiny_dense.reset();
    state.tiny_dense_attempts=state.tiny_dense_successes=state.tiny_dense_rejections=0;
    state.coarse_solve_distribution=gipc::Json::array();
}

bool galerkin_adoption_enabled()
{
    return state.adoption_enabled;
}

void configure_instrumentation(bool timing, bool diagnostics)
{
    collect_timing = timing;
    collect_full_diagnostics = diagnostics;
}

bool collect_timing_enabled() { return collect_timing; }
bool collect_full_diagnostics_enabled() { return collect_full_diagnostics; }

void configure_direction_freeze(std::string directory,std::size_t after_update,
                                bool continue_motion,std::vector<int> frames,int late_after_newton)
{
    state.direction_freeze_directory=std::move(directory);
    state.direction_freeze_after_update=after_update;
    state.direction_frozen=false;
    state.direction_continue=continue_motion;
    state.direction_capture_frames=std::move(frames);
    state.direction_captured_frames.clear();
    state.direction_captured_late_frames.clear();
    state.direction_late_after_newton=late_after_newton;
    state.direction_frame=state.direction_newton_ordinal=0;
}

void set_direction_capture_context(int frame,int active_pairs,int active_ground_pairs,const int4* collision_pairs)
{
    state.direction_newton_ordinal=state.direction_frame==frame?state.direction_newton_ordinal+1:1;
    state.direction_collision_pairs=collision_pairs;
    state.direction_frame=frame;
    state.direction_pairs=active_pairs;
    state.direction_ground_pairs=active_ground_pairs;
}

gipc::Json direction_capture_summary()
{
    return {{"enabled",!state.direction_freeze_directory.empty()},
            {"continue_motion",state.direction_continue},
            {"requested_frames",state.direction_capture_frames},
            {"captured_frames",state.direction_captured_frames},
            {"late_after_newton",state.direction_late_after_newton},
            {"captured_late_frames",state.direction_captured_late_frames},
            {"capture_overhead_excluded_from_performance_claim",true}};
}

bool direction_freeze_complete() { return state.direction_frozen && !state.direction_continue; }

std::string freeze_accepted_direction(const GIPCTripletMatrix& matrix,
                               const double* rhs,std::size_t dofs)
{
    if(state.direction_freeze_directory.empty()
       || (state.direction_frozen && !state.direction_continue)
       || state.updates<state.direction_freeze_after_update) return {};
    const bool early_done=std::find(state.direction_captured_frames.begin(),state.direction_captured_frames.end(),
                                    state.direction_frame)!=state.direction_captured_frames.end();
    const bool late_done=std::find(state.direction_captured_late_frames.begin(),state.direction_captured_late_frames.end(),
                                   state.direction_frame)!=state.direction_captured_late_frames.end();
    const bool late_sample=early_done;
    if(state.direction_continue
       && ((state.direction_pairs<=0 && state.direction_ground_pairs<=0)
           || !std::binary_search(state.direction_capture_frames.begin(),
                                  state.direction_capture_frames.end(),state.direction_frame)
           || (early_done && (state.direction_late_after_newton==0 || late_done
                               || state.direction_newton_ordinal<state.direction_late_after_newton)))) return {};
    // Diagnostic sampling only: skip the tiny zero-increment system at step
    // entry. This does not affect mapping, solver convergence, or adoption.
    constexpr double capture_minimum_coarse_fraction=0.05;
    if(state.direction_continue
       && (late_sample?3*state.coarse_matrix->block_rows()<=96
                      :state.coarse_matrix->block_rows()<capture_minimum_coarse_fraction*matrix.block_rows())) return {};
    if(!state.last_adoption.value("adopted",false) || state.candidate_dofs!=dofs)
        throw std::runtime_error("accepted direction capture has incompatible state");
    std::filesystem::path directory(state.direction_freeze_directory);
    if(state.direction_continue)
        directory/=std::string("frame_")+std::to_string(state.direction_frame)+(late_sample?"_late":"");
    // Never overwrite an earlier frozen system with a different trajectory.
    if(std::filesystem::exists(directory) && !std::filesystem::is_empty(directory))
        throw std::runtime_error("direction capture directory must be empty");
    std::filesystem::create_directories(directory);
    const auto mapping=mapping_device_view();
    const std::size_t unique=matrix.h_unique_key_number;
    write_binary(directory/"fine_A_values.f64x9.bin",matrix.block_values(),unique);
    write_binary(directory/"fine_A_rows.i32.bin",matrix.block_row_indices(),unique);
    write_binary(directory/"fine_A_cols.i32.bin",matrix.block_col_indices(),unique);
    write_binary(directory/"fine_rhs.f64.bin",rhs,dofs);
    write_binary(directory/"agipc_candidate.f64.bin",state.fine_solution.data(),dofs);
    write_binary(directory/"prolongated.f64.bin",state.prolonged.data(),dofs);
    write_binary(directory/"fine_to_coarse.i32.bin",mapping.fine_to_coarse,mapping.fine_nodes);
    write_binary(directory/"coarse_block_bases.i32.bin",mapping.coarse_block_bases,mapping.coarse_nodes);
    write_binary(directory/"coarse_basis_masks.i32.bin",mapping.basis_masks,mapping.coarse_nodes);
    write_binary(directory/"fine_rest_positions.f64x3.bin",mapping.rest_positions,mapping.fine_nodes);
    const auto& coarse=*state.coarse_matrix;
    const std::size_t coarse_unique=coarse.h_unique_key_number;
    write_binary(directory/"coarse_A_values.f64x9.bin",coarse.block_values(),coarse_unique);
    write_binary(directory/"coarse_A_rows.i32.bin",coarse.block_row_indices(),coarse_unique);
    write_binary(directory/"coarse_A_cols.i32.bin",coarse.block_col_indices(),coarse_unique);
    write_binary(directory/"coarse_rhs.f64.bin",state.coarse_rhs.data(),state.coarse_rhs.size());
    write_binary(directory/"coarse_solution.f64.bin",state.coarse_solution.data(),state.coarse_solution.size());
    if(state.direction_pairs>0 && !state.direction_collision_pairs)
        throw std::runtime_error("native collision context missing at capture");
    write_binary(directory/"native_collision_pairs.i32x4.bin",state.direction_collision_pairs,state.direction_pairs);
    const auto criterion_snapshot=capture_criterion_snapshot(directory.string());
    const gipc::Json metadata={
        {"format","agipc_accepted_direction_v3"},{"update_index",state.updates},
        {"frame",state.direction_frame},{"frame_newton_ordinal",state.direction_newton_ordinal},
        {"sample_kind",late_sample?"late":"early"},
        {"native_matrix_segments",{{"abd_abd_start",matrix.h_abd_abd_contact_start_id},
            {"abd_fem_start",matrix.h_abd_fem_contact_start_id},{"fem_abd_start",matrix.h_fem_abd_contact_start_id},
            {"fem_fem_start",matrix.h_fem_fem_contact_start_id},{"abd_abd_count",matrix.abd_abd_contact_num},
            {"abd_fem_count",matrix.abd_fem_contact_num},{"fem_abd_count",matrix.fem_abd_contact_num},
            {"fem_fem_count",matrix.fem_fem_contact_num}}},
        {"active_contact_pairs",state.direction_pairs},
        {"active_ground_pairs",state.direction_ground_pairs},
        {"continue_motion",state.direction_continue},{"galerkin_state",state.last},
        {"sampling_minimum_coarse_fraction",state.direction_continue && !late_sample?capture_minimum_coarse_fraction:0.0},
        {"sampling_excluded_tiny_max_dofs",late_sample?96:0},
        {"requested_after_update",state.direction_freeze_after_update},
        {"fine_dofs",dofs},{"fine_block_nodes",matrix.block_rows()},
        {"fine_unique_blocks",unique},{"fine_matrix_half_storage",true},
        {"block_values_layout","Eigen column-major 3x3 FP64"},
        {"mapping_fine_nodes",mapping.fine_nodes},{"mapping_coarse_nodes",mapping.coarse_nodes},
        {"coarse_block_nodes",coarse.block_rows()},{"coarse_unique_blocks",coarse_unique},
        {"coarse_solve",state.last["coarse_solve"]},
        {"post_correction",state.last["post_correction"]},{"adoption",state.last_adoption},
        {"guarded_fine_correction",state.last.value("guarded_fine_correction",gipc::Json(nullptr))},
        {"coarse_preconditioner",state.use_coarse_mas32?
            (state.use_factorized_mas32?"mas32-factor":"mas32"):"block_jacobi"},
        {"mas_validation",state.mas_validation},{"mas_reuse_enabled",state.mas_reuse_enabled},
        {"criterion_snapshot",criterion_snapshot},
        {"captured_before_ccd_line_search_state_update",true},
        {"stop_before_ccd_line_search_state_update",!state.direction_continue},{"performance_claim",false}};
    std::ofstream output(directory/"metadata.json");
    output<<metadata.dump(2)<<'\n';output.close();
    if(!output) throw std::runtime_error("failed to write direction capture metadata");
    (late_sample?state.direction_captured_late_frames:state.direction_captured_frames).push_back(state.direction_frame);
    state.direction_frozen=true;
    return directory.string();
}

gipc::Json update_galerkin_shadow(const GIPCTripletMatrix& fine_matrix,
                                  const double* fine_rhs,
                                  std::size_t fine_rhs_dofs)
{
    return assemble_shadow(state,fine_matrix,fine_rhs,fine_rhs_dofs,mapping_device_view());
}

gipc::Json adopt_galerkin_candidate(double* destination, std::size_t destination_dofs)
{
    return adopt_candidate(state,destination,destination_dofs);
}

void configure_guarded_fine_correction(bool enabled){guarded_fine_correction=enabled;}
bool guarded_fine_correction_enabled(){return guarded_fine_correction;}
void configure_guarded_fine_warm_start(bool enabled){guarded_fine_warm_start=enabled;}
bool guarded_fine_warm_start_enabled(){return guarded_fine_warm_start;}
double galerkin_prolongated_residual_squared(){
    const double norm=state.last.at("post_correction").at("initial_residual_norm");
    return norm*norm;
}
const double* galerkin_prolongated(std::size_t dofs){
    if(!state.candidate_ready || state.candidate_dofs!=dofs)throw std::runtime_error("guarded trial candidate dimensions differ");
    return state.prolonged.data();
}
void record_guarded_fine_correction(gipc::Json result,const double* selected,std::size_t dofs){
    state.last["guarded_fine_correction"]=result;
    for(const char* key:{"iterations","preconditioner_setup_wall_ms","total_wall_ms"})
        state.guarded_totals[key]=state.guarded_totals.value(key,0.0)+result.value(key,0.0);
    state.guarded_totals["attempts"]=state.guarded_totals.value("attempts",0)+1;
    const std::string reason=result.value("selection_reason",std::string("unknown"));
    auto& counts=state.guarded_totals["selection_reason_counts"];
    if(counts.is_null())counts=gipc::Json::object();
    counts[reason]=counts.value(reason,0)+1;
    state.guarded_totals["selected"]=state.guarded_totals.value("selected",0)+(result.value("selected",false)?1:0);
    if(result.value("selected",false))CUDA_SAFE_CALL(cudaMemcpy(state.fine_solution.data(),selected,dofs*sizeof(double),cudaMemcpyDeviceToDevice));
}

void record_linear_solve_timing(gipc::Json timing)
{
    if(state.last.is_null()) return;
    state.last["linear_system_timing_ms"]=timing;
    for(const char* key: {"fine_assembly","adaptive_pipeline","fine_preconditioner",
                          "candidate_decision","fallback_fine_solve","guarded_fine_correction","distribute",
                          "build_total","solve_total"})
    {
        const double value=timing.value(key,0.0);
        state.total_linear_timing[key]=state.total_linear_timing.value(key,0.0)+value;
    }
}

void record_fallback_direction_quality(const GIPCTripletMatrix& fine_matrix,
                                       const double* fine_rhs,
                                       const double* fine_direction,
                                       std::size_t fine_dofs)
{
    if(state.fallback_diagnostics_directory.empty() || state.last.is_null()
       || state.last_adoption.value("reason",std::string{})
              !="post_residual_not_reduced"
       || state.candidate_dofs!=fine_dofs || fine_dofs==0)
        return;

    const auto comparison_start=std::chrono::steady_clock::now();
    const int dofs=static_cast<int>(fine_dofs);
    auto candidate=direction_metrics(state,fine_matrix,fine_rhs,
                                     state.fine_solution.data(),dofs);
    auto fine=direction_metrics(state,fine_matrix,fine_rhs,fine_direction,dofs);
    const double candidate_dot_fine=device_dot(state.fine_solution.data(),
                                                fine_direction,dofs);
    const double candidate_norm=candidate.value("direction_norm",0.0);
    const double fine_norm=fine.value("direction_norm",0.0);
    const double cosine_denominator=candidate_norm*fine_norm;
    const double cosine=cosine_denominator>0
        ? std::clamp(candidate_dot_fine/cosine_denominator,-1.0,1.0) : 0.0;
    const double difference2=std::max(0.0,candidate_norm*candidate_norm
        +fine_norm*fine_norm-2.0*candidate_dot_fine);
    const double candidate_residual=candidate.value("residual_norm",0.0);
    const double fine_residual=fine.value("residual_norm",0.0);
    const double post_reduction=
        state.last["post_correction"].value("residual_reduction_ratio",1.0);
    const int category=quality_category(post_reduction);
    const auto comparison_end=std::chrono::steady_clock::now();
    gipc::Json quality={
        {"failure_reason","post_residual_not_reduced"},
        {"post_residual_reduction_ratio",post_reduction},
        {"candidate",candidate},{"fine_fallback",fine},
        {"candidate_to_fine_residual_ratio",candidate_residual
            /std::max(fine_residual,1e-300)},
        {"candidate_to_fine_direction_norm_ratio",candidate_norm
            /std::max(fine_norm,1e-300)},
        {"candidate_fine_cosine",cosine},
        {"candidate_fine_angle_degrees",std::acos(cosine)*180.0/3.14159265358979323846},
        {"relative_direction_difference",std::sqrt(difference2)
            /std::max(fine_norm,1e-300)},
        {"quality_category",quality_category_name(category)},
        {"comparison_ms",std::chrono::duration<double,std::milli>(
            comparison_end-comparison_start).count()},
        {"fine_dofs",fine_dofs},
        {"fine_unique_blocks",fine_matrix.h_unique_key_number},
        {"coarse_fem_nodes",state.last.value("coarse_fem_nodes",0)},
        {"coarse_block_nodes",state.last.value("coarse_block_nodes",0)},
        {"coarse_iterations",state.last["coarse_solve"].value("iterations",0)},
        {"post_iterations",state.last["post_correction"].value("iterations",0)}};
    ++state.direction_quality_evaluations;
    quality["evaluation_index"]=state.direction_quality_evaluations;

    if(!state.frozen_quality_categories[category])
    {
        const auto freeze_start=std::chrono::steady_clock::now();
        const std::filesystem::path root(state.fallback_diagnostics_directory);
        const std::string sample_name=std::string{"post_residual_"}
            +quality_category_name(category)+"_"
            +std::to_string(state.direction_quality_evaluations);
        const auto sample_directory=root/sample_name;
        std::filesystem::create_directories(sample_directory);
        const std::size_t fine_unique=fine_matrix.h_unique_key_number;
        write_binary(sample_directory/"fine_A_values.f64x9.bin",
                     fine_matrix.block_values(),fine_unique);
        write_binary(sample_directory/"fine_A_rows.i32.bin",
                     fine_matrix.block_row_indices(),fine_unique);
        write_binary(sample_directory/"fine_A_cols.i32.bin",
                     fine_matrix.block_col_indices(),fine_unique);
        write_binary(sample_directory/"fine_rhs.f64.bin",fine_rhs,fine_dofs);
        write_binary(sample_directory/"agipc_candidate.f64.bin",
                     state.fine_solution.data(),fine_dofs);
        write_binary(sample_directory/"fine_fallback.f64.bin",fine_direction,fine_dofs);

        const auto mapping=mapping_device_view();
        write_binary(sample_directory/"fine_to_coarse.i32.bin",
                     mapping.fine_to_coarse,mapping.fine_nodes);
        write_binary(sample_directory/"coarse_block_bases.i32.bin",
                     mapping.coarse_block_bases,mapping.coarse_nodes);
        write_binary(sample_directory/"coarse_basis_masks.i32.bin",
                     mapping.basis_masks,mapping.coarse_nodes);
        write_binary(sample_directory/"fine_rest_positions.f64x3.bin",
                     mapping.rest_positions,mapping.fine_nodes);

        const std::size_t coarse_unique=state.coarse_matrix
            ? state.coarse_matrix->h_unique_key_number : 0;
        if(state.coarse_matrix)
        {
            write_binary(sample_directory/"coarse_A_values.f64x9.bin",
                         state.coarse_matrix->block_values(),coarse_unique);
            write_binary(sample_directory/"coarse_A_rows.i32.bin",
                         state.coarse_matrix->block_row_indices(),coarse_unique);
            write_binary(sample_directory/"coarse_A_cols.i32.bin",
                         state.coarse_matrix->block_col_indices(),coarse_unique);
            write_binary(sample_directory/"coarse_rhs.f64.bin",
                         state.coarse_rhs.data(),state.coarse_rhs.size());
            write_binary(sample_directory/"coarse_solution.f64.bin",
                         state.coarse_solution.data(),state.coarse_solution.size());
        }
        quality["sample_name"]=sample_name;
        quality["fine_block_nodes"]=fine_matrix.block_rows();
        quality["fine_matrix_half_storage"]=true;
        quality["coarse_unique_blocks"]=coarse_unique;
        quality["mapping_fine_nodes"]=mapping.fine_nodes;
        quality["mapping_coarse_nodes"]=mapping.coarse_nodes;
        quality["artifacts"]={
            {"fine_matrix",{"fine_A_values.f64x9.bin","fine_A_rows.i32.bin",
                             "fine_A_cols.i32.bin"}},
            {"fine_rhs","fine_rhs.f64.bin"},
            {"agipc_candidate","agipc_candidate.f64.bin"},
            {"fine_fallback","fine_fallback.f64.bin"},
            {"mapping",{"fine_to_coarse.i32.bin","coarse_block_bases.i32.bin",
                         "coarse_basis_masks.i32.bin","fine_rest_positions.f64x3.bin"}},
            {"coarse_matrix",{"coarse_A_values.f64x9.bin","coarse_A_rows.i32.bin",
                               "coarse_A_cols.i32.bin"}},
            {"coarse_rhs","coarse_rhs.f64.bin"},
            {"coarse_solution","coarse_solution.f64.bin"}};
        quality["freeze_ms"]=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-freeze_start).count();
        state.frozen_quality_categories[category]=true;
        state.frozen_fallback_samples.push_back(quality);
        std::ofstream metadata(sample_directory/"metadata.json");
        metadata<<quality.dump(2)<<'\n';
        std::ofstream manifest(root/"manifest.json");
        manifest<<gipc::Json{{"format","agipc_fallback_direction_v1"},
            {"samples",state.frozen_fallback_samples}}.dump(2)<<'\n';
    }
    state.last["fallback_direction_quality"]=quality;
}

namespace
{
gipc::Json build_galerkin_summary(const GalerkinState& snapshot,bool history)
{
    const auto& state=snapshot;
    if(state.last.is_null()) return nullptr;
    auto result=state.last;
    result["updates"]=state.updates;
    result["total_shadow_pipeline_ms"]=collect_timing?gipc::Json(state.total_ms):gipc::Json(nullptr);
    result["total_stage_timing_ms"]=collect_timing?gipc::Json{{"assembly",state.total_assembly_ms},
        {"coarse_solve",state.total_coarse_solve_ms},
        {"post_correction",state.total_post_correction_ms}}:gipc::Json(nullptr);
    result["total_linear_system_timing_ms"]=collect_timing?state.total_linear_timing:gipc::Json(nullptr);
    result["direction_quality_evaluations"]=state.direction_quality_evaluations;
    result["residual_guard_restores"]=state.residual_guard_restores;
    result["coarse_preconditioner_requested"]=state.use_coarse_mas32?
        (state.use_factorized_mas32?"mas32-factor":"mas32"):"block_jacobi";
    result["mas_attempts"]=state.mas_attempts;
    result["tiny_dense_max_dofs"]=state.tiny_dense_max_dofs;
    result["tiny_dense_attempts"]=state.tiny_dense_attempts;
    result["tiny_dense_successes"]=state.tiny_dense_successes;
    result["tiny_dense_rejections"]=state.tiny_dense_rejections;
    if(history) result["coarse_solve_distribution"]=state.coarse_solve_distribution;
    else {
        result["step_statistics_scope"]="current solve and cumulative counters; complete history in final metrics";
        result["coarse_solve_distribution_records_so_far"]=state.coarse_solve_distribution.size();
        result["latest_coarse_size"]=state.coarse_solve_distribution.empty()
            ?gipc::Json(nullptr):state.coarse_solve_distribution.back();
    }
    result["mas_local_failures"]=state.mas_local_failures;
    result["total_mas_setup_wall_ms"]=state.total_mas_setup_ms;
    result["total_mas_validation_wall_ms"]=state.total_mas_validation_ms;
    result["mas_validation_mode"]=state.mas_validation;
    result["mas_reuse_enabled"]=state.mas_reuse_enabled;
    result["mas_graph_reuses"]=state.mas_graph_reuses;
    if(history) {
        result["mas_failure_samples"]=state.mas_failure_samples;
        result["frozen_fallback_samples"]=state.frozen_fallback_samples;
        result["frozen_coarse_samples"]=state.frozen_coarse_samples;
    } else {
        result["mas_failure_samples_count"]=state.mas_failure_samples.size();
        result["frozen_fallback_samples_count"]=state.frozen_fallback_samples.size();
        result["frozen_coarse_samples_count"]=state.frozen_coarse_samples.size();
    }
    result["adoption_enabled"]=state.adoption_enabled;
    result["adoption_attempts"]=state.adoption_attempts;
    result["adoptions"]=state.adoptions;
    result["fallbacks"]=state.fallbacks;
    result["fallback_reason_counts"]=state.fallback_reason_counts;
    result["guarded_fine_correction_enabled"]=guarded_fine_correction;
    result["guarded_fine_warm_start_enabled"]=guarded_fine_warm_start;
    result["guarded_fine_correction_totals"]=state.guarded_totals;
    if(state.last.contains("guarded_fine_correction"))result["guarded_fine_correction"]=state.last["guarded_fine_correction"];
    result["last_adoption"]=state.last_adoption;
    result["controls_solver"]=!state.last_adoption.is_null()
                              && state.last_adoption.value("adopted",false);
    return result;
}
}

void configure_step_statistics_mode(StepStatisticsMode mode) { step_statistics_mode=mode; }
gipc::Json galerkin_summary() { return build_galerkin_summary(state,true); }
gipc::Json galerkin_step_summary() { return build_galerkin_summary(state,step_history_enabled()); }

gipc::Json step_statistics_self_test()
{
    GalerkinState fixture;
    fixture.last={{"coarse_solve",{{"converged",true},{"iterations",7},{"true_relative_residual",1e-5}}},
                  {"post_correction",{{"attempted",true},{"failure_reason",""}}}};
    fixture.adoptions=16; fixture.fallbacks=2; fixture.mas_attempts=18;
    fixture.mas_failure_samples.push_back({{"update_index",3},{"diagnostic",{{"passed",false}}}});
    auto require=[](bool ok,const char* message) { if(!ok) throw std::runtime_error(message); };
    std::size_t minimum_bytes=0,maximum_bytes=0;
    for(int count:{0,1,2048}) {
        fixture.coarse_solve_distribution=gipc::Json::array();
        for(int i=0;i<count;++i) fixture.coarse_solve_distribution.push_back(
            {{"update_index",i+1},{"dofs",96},{"failure_reason",""},{"converged",true}});
        fixture.updates=count;
        const auto original=fixture.coarse_solve_distribution.dump();
        const auto compact=build_galerkin_summary(fixture,false);
        const auto complete=build_galerkin_summary(fixture,true);
        require(!compact.contains("coarse_solve_distribution"),"step summary copied the growing history");
        require(compact.at("coarse_solve_distribution_records_so_far")==count,"step history count lost");
        require(complete.at("coarse_solve_distribution").size()==std::size_t(count),"final history lost");
        require(compact.at("latest_coarse_size")==
            (count?fixture.coarse_solve_distribution.back():gipc::Json(nullptr)),"latest size lost");
        require(compact.at("coarse_solve")==fixture.last.at("coarse_solve"),"current solve guard changed");
        require(compact.at("post_correction")==fixture.last.at("post_correction"),"post guard changed");
        for(const char* key:{"adoptions","fallbacks","mas_attempts","updates"})
            require(compact.at(key)==complete.at(key),"cumulative solve counter changed");
        require(compact.at("mas_failure_samples_count")==1 && complete.at("mas_failure_samples").size()==1,
                "failure sample retention changed");
        require(original==fixture.coarse_solve_distribution.dump(),"summary mutated its input history");
        const auto bytes=compact.dump().size();
        if(count==0) minimum_bytes=bytes;
        maximum_bytes=std::max(maximum_bytes,bytes);
    }
    require(maximum_bytes<=minimum_bytes+256,"step summary grew with history length");
    const auto saved_mode=step_statistics_mode;
    const bool saved_diagnostics=collect_full_diagnostics;
    step_statistics_mode=StepStatisticsMode::Automatic;
    collect_full_diagnostics=false; const bool automatic_compact=!step_history_enabled();
    collect_full_diagnostics=true; const bool automatic_history=step_history_enabled();
    step_statistics_mode=StepStatisticsMode::Compact; const bool forced_compact=!step_history_enabled();
    collect_full_diagnostics=false; step_statistics_mode=StepStatisticsMode::History;
    const bool forced_history=step_history_enabled();
    step_statistics_mode=saved_mode; collect_full_diagnostics=saved_diagnostics;
    require(automatic_compact && automatic_history && forced_compact && forced_history,"statistics mode resolution failed");
    return {{"passed",true},{"history_lengths",{0,1,2048}},
            {"empty_compact_bytes",minimum_bytes},{"maximum_compact_bytes",maximum_bytes},
            {"final_history_retained",true},{"current_guards_preserved",true},
            {"input_history_unchanged",true},{"mode_resolution_cases",4}};
}

gipc::Json galerkin_self_test()
{
    constexpr int fine_blocks=7, prefix_blocks=1, fine_fem_nodes=6, coarse_fem_nodes=2;
    std::vector<int> rows,cols;
    std::vector<Eigen::Matrix3d> values;
    Eigen::MatrixXd q(3*fine_blocks,3*fine_blocks);
    for(int row=0;row<q.rows();++row) for(int col=0;col<q.cols();++col)
        q(row,col)=std::sin(0.31*(row+1)*(col+2))+0.2*std::cos(0.17*(row+3+col));
    const Eigen::MatrixXd dense=q.transpose()*q+Eigen::MatrixXd::Identity(q.cols(),q.cols());
    auto add=[&](int row,int col,const Eigen::Matrix3d& value)
    {
        rows.push_back(row); cols.push_back(col); values.push_back(value);
    };
    const Eigen::Matrix3d identity=Eigen::Matrix3d::Identity();
    for(int row=0;row<fine_blocks;++row) for(int col=row;col<fine_blocks;++col)
        add(row,col,dense.block<3,3>(3*row,3*col));

    GIPCTripletMatrix fine;
    fine.init_var();
    fine.resize(fine_blocks,fine_blocks,values.size()*2);
    fine.resize_collision_hash_size(values.size());
    fine.global_triplet_offset=static_cast<int>(values.size());
    CUDA_SAFE_CALL(cudaMemcpy(fine.block_values(),values.data(),
        values.size()*sizeof(Eigen::Matrix3d),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(fine.block_row_indices(),rows.data(),
        rows.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(fine.block_col_indices(),cols.data(),
        cols.size()*sizeof(int),cudaMemcpyHostToDevice));
    gipc::Converter converter;
    converter.convert(fine,0,fine.global_triplet_offset,fine.global_triplet_offset);

    // Old coarse id 0 is affine and is sorted after translational old id 1.
    cudatool::CudaDeviceBuffer<int> mapping(std::vector<int>{0,0,0,0,1,1});
    cudatool::CudaDeviceBuffer<int> coarse_bases(std::vector<int>{1,0});
    cudatool::CudaDeviceBuffer<int> basis_masks(std::vector<int>{15,1});
    std::vector<double3> rest={make_double3(0,0,0),make_double3(1,0,0),
                               make_double3(0,1,0),make_double3(0,0,1),
                               make_double3(-0.1,0.8,-0.6),make_double3(0.9,0.4,0.2)};
    cudatool::CudaDeviceBuffer<double3> device_rest(rest);
    Eigen::VectorXd b(3*fine_blocks);
    for(int i=0;i<b.size();++i) b[i]=0.25+0.125*i;
    cudatool::CudaDeviceBuffer<double> device_b(
        std::vector<double>(b.data(),b.data()+b.size()));
    GalerkinState local;
    local.adoption_enabled=true;
    const auto stats=assemble_shadow(local,fine,device_b.data(),b.size(),
        {mapping.data(),coarse_bases.data(),basis_masks.data(),device_rest.data(),
         fine_fem_nodes,coarse_fem_nodes,1,1,5,true,true});

    Eigen::MatrixXd prolongation=Eigen::MatrixXd::Zero(3*fine_blocks,
        3*(prefix_blocks+5));
    prolongation.block<3,3>(0,0)=identity;
    for(int local_node=0;local_node<fine_fem_nodes;++local_node)
    {
        const int old=local_node<4?0:1;
        const int base=prefix_blocks+(old==0?1:0);
        const int width=old==0?4:1;
        const double phi[4]={1.0,rest[local_node].x,rest[local_node].y,rest[local_node].z};
        for(int a=0;a<width;++a)
            prolongation.block<3,3>(3*(prefix_blocks+local_node),3*(base+a))=phi[a]*identity;
    }
    const Eigen::MatrixXd expected=prolongation.transpose()*dense*prolongation;
    const Eigen::VectorXd expected_b=prolongation.transpose()*b;
    const Eigen::VectorXd expected_coarse_solution=expected.ldlt().solve(expected_b);
    const Eigen::VectorXd expected_fine_direction=prolongation*expected_coarse_solution;

    const int unique=local.coarse_matrix->h_unique_key_number;
    std::vector<int> coarse_rows(unique),coarse_cols(unique);
    std::vector<Eigen::Matrix3d> coarse_values(unique);
    CUDA_SAFE_CALL(cudaMemcpy(coarse_rows.data(),local.coarse_matrix->block_row_indices(),
        unique*sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(coarse_cols.data(),local.coarse_matrix->block_col_indices(),
        unique*sizeof(int),cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(coarse_values.data(),local.coarse_matrix->block_values(),
        unique*sizeof(Eigen::Matrix3d),cudaMemcpyDeviceToHost));
    Eigen::MatrixXd actual=Eigen::MatrixXd::Zero(expected.rows(),expected.cols());
    for(int i=0;i<unique;++i)
    {
        const int row=coarse_rows[i],col=coarse_cols[i];
        actual.block<3,3>(3*row,3*col)+=coarse_values[i];
        if(row!=col) actual.block<3,3>(3*col,3*row)+=coarse_values[i].transpose();
    }
    std::vector<double> host_b;
    local.coarse_rhs.copy_to_host(host_b);
    const Eigen::Map<const Eigen::VectorXd> actual_b(host_b.data(),host_b.size());
    const double matrix_error=(actual-expected).norm()/std::max(1.0,expected.norm());
    const double rhs_error=(actual_b-expected_b).norm()/std::max(1.0,expected_b.norm());
    const double symmetry_error=(actual-actual.transpose()).norm()/std::max(1.0,actual.norm());
    std::vector<double> host_direction;
    local.prolonged.copy_to_host(host_direction);
    const Eigen::Map<const Eigen::VectorXd> actual_direction(host_direction.data(),host_direction.size());
    const double direction_error=(actual_direction-expected_fine_direction).norm()
        /std::max(1.0,expected_fine_direction.norm());
    std::vector<double> host_corrected_direction;
    local.fine_solution.copy_to_host(host_corrected_direction);
    const Eigen::Map<const Eigen::VectorXd> corrected_direction(
        host_corrected_direction.data(),host_corrected_direction.size());
    const Eigen::VectorXd exact_fine_direction=dense.ldlt().solve(b);
    const double initial_exact_error=(actual_direction-exact_fine_direction).norm()
        /std::max(1.0,exact_fine_direction.norm());
    const double corrected_exact_error=(corrected_direction-exact_fine_direction).norm()
        /std::max(1.0,exact_fine_direction.norm());
    cudatool::CudaDeviceBuffer<double> adopted_direction(b.size());
    const auto fallback=adopt_candidate(local,adopted_direction.data(),b.size()-3);
    const auto adoption=adopt_candidate(local,adopted_direction.data(),b.size());
    std::vector<double> host_adopted_direction;
    adopted_direction.copy_to_host(host_adopted_direction);
    const Eigen::Map<const Eigen::VectorXd> actual_adopted_direction(
        host_adopted_direction.data(),host_adopted_direction.size());
    const double adoption_copy_error=(actual_adopted_direction-corrected_direction).norm();
    Eigen::VectorXd coarse_probe=Eigen::VectorXd::LinSpaced(expected.rows(),-0.7,0.9);
    Eigen::VectorXd fine_probe=Eigen::VectorXd::LinSpaced(dense.rows(),0.3,-0.5);
    const double adjoint_error=std::abs((prolongation*coarse_probe).dot(fine_probe)
        -coarse_probe.dot(prolongation.transpose()*fine_probe));
    const bool positive_definite=Eigen::LLT<Eigen::MatrixXd>(actual).info()==Eigen::Success;
    Eigen::Matrix<double,5,4> planar,collinear;
    for(int i=0;i<5;++i)
    {
        planar.row(i)<<1.0,double(i),double(i*i),0.0;
        collinear.row(i)<<1.0,double(i),0.0,0.0;
    }
    const int planar_rank=planar.fullPivLu().rank();
    const int collinear_rank=collinear.fullPivLu().rank();
    const bool coarse_converged=stats["coarse_solve"]["converged"].get<bool>();
    const double residual_ratio=stats["coarse_solve"]["fine_residual_ratio"].get<double>();
    const double projected_residual_ratio=
        stats["coarse_solve"]["final_residual_norm"].get<double>()
        /stats["coarse_solve"]["initial_residual_norm"].get<double>();
    const double rhs_dot_direction=stats["coarse_solve"]["rhs_dot_direction"].get<double>();
    const bool post_nonzero=stats["post_correction"]["nonzero_initial_guess"].get<bool>();
    const int post_iterations=stats["post_correction"]["iterations"].get<int>();
    const double post_reduction=stats["post_correction"]["residual_reduction_ratio"].get<double>();
    const bool post_residual_reduced=stats["post_correction"]["residual_reduced"].get<bool>();
    const double post_rhs_dot=stats["post_correction"]["rhs_dot_direction"].get<double>();
    if(matrix_error>1e-12 || rhs_error>1e-12 || symmetry_error>1e-14
       || direction_error>5e-3 || adjoint_error>1e-12 || !positive_definite
       || !coarse_converged || residual_ratio>=1.0 || rhs_dot_direction<=0
       || projected_residual_ratio>1e-3
       || !post_nonzero || post_iterations>10 || !post_residual_reduced
       || post_reduction>=1.0 || post_rhs_dot<=0
       || corrected_exact_error>=initial_exact_error
       || fallback.value("adopted",true) || !adoption.value("adopted",false)
       || local.fallbacks!=1 || local.adoptions!=1
       || local.fallback_reason_counts.value("dof_mismatch",std::size_t{0})!=1
       || adoption_copy_error>1e-15
       || planar_rank!=3 || collinear_rank!=2)
        throw std::runtime_error("Gate D mixed GPU Galerkin mismatch: matrix="
            +std::to_string(matrix_error)+", rhs="+std::to_string(rhs_error)
            +", symmetry="+std::to_string(symmetry_error)+", adjoint="
            +std::to_string(adjoint_error)+", direction="+std::to_string(direction_error)
            +", residual_ratio="+std::to_string(residual_ratio)
            +", projected_ratio="+std::to_string(projected_residual_ratio)
            +", post_reduction="+std::to_string(post_reduction)
            +", initial_exact_error="+std::to_string(initial_exact_error)
            +", corrected_exact_error="+std::to_string(corrected_exact_error)
            +", spd="+(positive_definite?"true":"false")
            +", planar_rank="+std::to_string(planar_rank)
            +", collinear_rank="+std::to_string(collinear_rank));
    return {{"test","agipc_galerkin_mixed"},{"passed",true},
            {"relative_matrix_error",matrix_error},{"relative_rhs_error",rhs_error},
            {"relative_symmetry_error",symmetry_error},{"tolerance",1e-12},
            {"adjoint_error",adjoint_error},{"positive_definite",positive_definite},
            {"coarse_pcg_converged",coarse_converged},
            {"prolongated_direction_relative_error",direction_error},
            {"projected_residual_ratio",projected_residual_ratio},
            {"fine_residual_ratio",residual_ratio},{"rhs_dot_direction",rhs_dot_direction},
            {"post_pcg_nonzero_initial_guess",post_nonzero},
            {"post_pcg_iterations",post_iterations},{"post_pcg_iteration_cap",10},
            {"post_pcg_residual_reduction_ratio",post_reduction},
            {"coarse_direction_exact_error",initial_exact_error},
            {"corrected_direction_exact_error",corrected_exact_error},
            {"adoption_copy_error",adoption_copy_error},
            {"adoption_gate_case","dimension_mismatch_fallback_then_valid_adoption"},
            {"planar_affine_basis_rank",planar_rank},{"collinear_affine_basis_rank",collinear_rank},
            {"block_shapes","1x1/1x4/4x1/4x4"},
            {"fine_block_nodes",fine_blocks},{"coarse_block_nodes",prefix_blocks+5},
            {"fine_unique_blocks",fine.h_unique_key_number},
            {"coarse_unique_blocks",stats["coarse_unique_blocks"]},
            {"gpu_kernels",true},{"controls_solver",false}};
}
}
