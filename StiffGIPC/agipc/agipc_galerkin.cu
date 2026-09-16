#include "agipc_criterion.cuh"

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

struct RuntimeCoarseMas;
struct GalerkinState
{
    std::unique_ptr<GIPCTripletMatrix> coarse_matrix;
    cudatool::CudaDeviceBuffer<double> coarse_rhs;
    cudatool::CudaDeviceBuffer<int> invalid_entries;
    cudatool::CudaDeviceBuffer<int> expansion_counts;
    cudatool::CudaDeviceBuffer<int> expansion_offsets;
    cudatool::CudaDeviceBuffer<double> coarse_solution,coarse_r,coarse_z,coarse_p,coarse_ap;
    cudatool::CudaDeviceBuffer<double> prolonged,fine_solution,fine_ax,fine_residual;
    cudatool::CudaDeviceBuffer<double> fine_z,fine_p,fine_ap;
    cudatool::CudaDeviceBuffer<Eigen::Matrix3d> inverse_diagonal;
    cudatool::CudaDeviceBuffer<int> diagonal_found;
    cudatool::CudaDeviceBuffer<Eigen::Matrix3d> fine_inverse_diagonal;
    cudatool::CudaDeviceBuffer<int> fine_diagonal_found,fine_invalid_entries;
    int post_max_iterations=10;
    bool use_coarse_mas32=false;
    std::size_t mas_attempts=0,mas_local_failures=0;
    double total_mas_setup_ms=0,total_mas_validation_ms=0;
    std::string mas_validation="gpu";
    bool mas_reuse_enabled=false;
    std::size_t mas_graph_reuses=0;
    gipc::Json mas_failure_samples=gipc::Json::array();
    std::shared_ptr<RuntimeCoarseMas> runtime_mas;
    bool adoption_enabled=false;
    bool candidate_ready=false;
    std::string candidate_failure_reason="not_assembled";
    std::size_t candidate_dofs=0;
    int candidate_iterations=0;
    std::size_t adoption_attempts=0,adoptions=0,fallbacks=0;
    gipc::Json fallback_reason_counts=gipc::Json::object();
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
    std::array<bool,4> frozen_coarse_categories={false,false,false,false};
    gipc::Json frozen_coarse_samples=gipc::Json::array();
    std::string direction_freeze_directory;
    std::size_t direction_freeze_after_update=0;
    bool direction_frozen=false;
};

GalerkinState state;

__global__ void append_mas_padding(Eigen::Matrix3d* values,int* rows,int* cols,
                                   int unique,int blocks,int padding)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<padding)
    { values[unique+i].setIdentity(); rows[unique+i]=blocks+i; cols[unique+i]=blocks+i; }
}

// Explicit intermediate runtime adapter. Original Galerkin dimensions and PCG
// vectors stay unchanged; only MAS storage has isolated identity padding.
struct RuntimeCoarseMas
{
    gpu_mas32::TraditionalMAS32Preconditioner value;
    bool allocated=false;
    int blocks=0,padded=0;
    bool last_reused=false;
    std::vector<int> cached_rows,cached_cols;
    cudatool::CudaDeviceBuffer<Eigen::Matrix3d> values;
    cudatool::CudaDeviceBuffer<int> rows,cols;
    cudatool::CudaDeviceBuffer<uint32_t> indices;
    cudatool::CudaDeviceBuffer<double> residual,z;
    ~RuntimeCoarseMas() { if(allocated) value.FreeMAS(); }
    void setup(const GIPCTripletMatrix& matrix,bool allow_reuse)
    {
        const int old_blocks=blocks;
        blocks=matrix.block_rows(); padded=(blocks+31)/32*32;
        const int unique=matrix.h_unique_key_number,padding=padded-blocks;
        std::vector<int> host_rows(unique),host_cols(unique);
        CUDA_SAFE_CALL(cudaMemcpy(host_rows.data(),matrix.block_row_indices(),unique*sizeof(int),cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(host_cols.data(),matrix.block_col_indices(),unique*sizeof(int),cudaMemcpyDeviceToHost));
        last_reused=allocated && allow_reuse && old_blocks==blocks
            && host_rows==cached_rows && host_cols==cached_cols;
        if(last_reused)
        {
            // The graph, padding and index buffers are unchanged; refresh only
            // the numerical blocks and the zero-padded residual.
            values.resize(unique+padding);
            CUDA_SAFE_CALL(cudaMemcpy(values.data(),matrix.block_values(),
                unique*sizeof(Eigen::Matrix3d),cudaMemcpyDeviceToDevice));
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
        cached_rows=std::move(host_rows);cached_cols=std::move(host_cols);
    }
    void apply(const double* input,double* output)
    {
        CUDA_SAFE_CALL(cudaMemcpy(residual.data(),input,3*blocks*sizeof(double),cudaMemcpyDeviceToDevice));
        value.preconditioning(reinterpret_cast<const double3*>(residual.data()),reinterpret_cast<double3*>(z.data()));
        CUDA_SAFE_CALL(cudaMemcpy(output,z.data(),3*blocks*sizeof(double),cudaMemcpyDeviceToDevice));
    }
    gipc::Json diagnostics(const std::string& mode) const
    {
        if(mode=="cpu") return value.numerical_diagnostics(reinterpret_cast<const double3*>(residual.data()));
        auto result=value.local_diagnostics_gpu();
        if(mode=="crosscheck")
        {
            const auto cpu=value.numerical_diagnostics(reinterpret_cast<const double3*>(residual.data()));
            const bool agreement=gpu_mas32::local_diagnostics_agree(result,cpu);
            result["gpu_passed"]=result.value("passed",false);
            result["cpu_gpu_agreement"]=agreement;result["cpu_reference"]=cpu;
            result["passed"]=result.value("passed",false) && agreement;
        }
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
    if(category<0 || target.frozen_coarse_categories[category]) return nullptr;

    const auto started=std::chrono::steady_clock::now();
    constexpr const char* names[]={"medium_converged","large_converged","iteration_cap","local_diagnostics_failure"};
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
    std::vector<double> residual_history{std::sqrt(std::max(0.0,initial2))};
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
            curvature_history.push_back(curvature);
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
            residual_history.push_back(std::sqrt(std::max(0.0,residual2)));
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
    target.coarse_r.resize(dofs); target.coarse_z.resize(dofs);
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
            mas->setup(coarse,target.mas_reuse_enabled);
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
    const auto apply_preconditioner=[&](const double* input,double* output) {
        if(mas) mas->apply(input,output);
        else apply_diagonal<<<(blocks+kThreads-1)/kThreads,kThreads>>>(target.inverse_diagonal.data(),input,output,blocks);
    };
    apply_preconditioner(target.coarse_r.data(),target.coarse_z.data());
    if(mas)
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
    for(;iterations<max_iterations && residual2>tolerance2;)
    {
        if(mas && (!std::isfinite(rz) || rz<=0))
        { failure="invalid_preconditioned_residual"; break; }
        spmv.warp_reduce_sym_spmv(1.0,coarse.block_values(),coarse.block_row_indices(),
            coarse.block_col_indices(),unique,
            cudatool::CDenseVectorView<double>(target.coarse_p.data(),dofs),0.0,
            cudatool::DenseVectorView<double>(target.coarse_ap.data(),dofs));
        const double curvature=device_dot(target.coarse_p.data(),target.coarse_ap.data(),dofs);
        if(!std::isfinite(curvature) || curvature<=0 || !std::isfinite(rz))
        { failure=!std::isfinite(curvature)?"nonfinite_curvature":"nonpositive_curvature"; break; }
        const double alpha=rz/curvature;
        update_x_r<<<(dofs+kThreads-1)/kThreads,kThreads>>>(target.coarse_solution.data(),
            target.coarse_r.data(),target.coarse_p.data(),target.coarse_ap.data(),alpha,dofs);
        ++iterations;
        apply_preconditioner(target.coarse_r.data(),target.coarse_z.data());
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
            {"preconditioner",mas?"mas32":"block_jacobi"},
            {"mas_local_diagnostics",mas_diagnostics},{"mas_setup_wall_ms",mas_setup_ms},
            {"mas_validation_wall_ms",mas_validation_ms},{"true_residual_checked",bool(mas)},
            {"mas_validation_mode",target.mas_validation},{"mas_graph_reused",mas && mas->last_reused},
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
    if(fine_unique<1)
    {
        target.candidate_failure_reason="fine_matrix_empty";
        return nullptr;
    }
    const int coarse_blocks=prefix_blocks+mapping.coarse_block_nodes;

    cudaEvent_t start=nullptr,assembly_end=nullptr,coarse_end=nullptr,end=nullptr;
    CUDA_SAFE_CALL(cudaEventCreate(&start));
    CUDA_SAFE_CALL(cudaEventCreate(&assembly_end));
    CUDA_SAFE_CALL(cudaEventCreate(&coarse_end));
    CUDA_SAFE_CALL(cudaEventCreate(&end));
    CUDA_SAFE_CALL(cudaEventRecord(start));
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
    CUDA_SAFE_CALL(cudaEventRecord(assembly_end));
    auto coarse_solve=solve_coarse_shadow(target,fine_matrix,fine_rhs,
        static_cast<int>(fine_rhs_dofs),mapping,prefix_blocks);
    CUDA_SAFE_CALL(cudaEventRecord(coarse_end));
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
    CUDA_SAFE_CALL(cudaEventRecord(end));
    CUDA_SAFE_CALL(cudaEventSynchronize(end));
    float elapsed=0,assembly_elapsed=0,coarse_elapsed=0,post_elapsed=0;
    CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed,start,end));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&assembly_elapsed,start,assembly_end));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&coarse_elapsed,assembly_end,coarse_end));
    CUDA_SAFE_CALL(cudaEventElapsedTime(&post_elapsed,coarse_end,end));
    CUDA_SAFE_CALL(cudaEventDestroy(start));
    CUDA_SAFE_CALL(cudaEventDestroy(assembly_end));
    CUDA_SAFE_CALL(cudaEventDestroy(coarse_end));
    CUDA_SAFE_CALL(cudaEventDestroy(end));
    coarse_solve["solve_ms"]=coarse_elapsed;
    post_correction["solve_ms"]=post_elapsed;
    target.invalid_entries.copy_to_host(invalid);
    ++target.updates;
    target.total_ms+=elapsed;
    target.total_assembly_ms+=assembly_elapsed;
    target.total_coarse_solve_ms+=coarse_elapsed;
    target.total_post_correction_ms+=post_elapsed;
    target.last={{"stage","galerkin_mixed_shadow"},{"formula","Ac=PT*A*P; bc=PT*b"},
                 {"fine_block_nodes",fine_blocks},{"prefix_identity_blocks",prefix_blocks},
                 {"fine_fem_nodes",mapping.fine_nodes},{"coarse_fem_nodes",mapping.coarse_nodes},
                 {"translational_nodes",mapping.translational_nodes},
                 {"affine_nodes",mapping.affine_nodes},
                 {"coarse_block_nodes",coarse_blocks},{"fine_unique_blocks",fine_unique},
                 {"expanded_blocks_before_reduction",expanded},
                 {"coarse_unique_blocks",coarse.h_unique_key_number},
                 {"invalid_entries",invalid[0]},{"shadow_pipeline_ms",elapsed},
                 {"stage_timing_ms",{{"assembly",assembly_elapsed},
                     {"coarse_solve",coarse_elapsed},{"post_correction",post_elapsed}}},
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

void configure_galerkin(int fine_correction_max_iterations,
                        bool adoption_enabled,
                        std::string fallback_diagnostics_directory,
                        std::string coarse_diagnostics_directory,
                        bool use_coarse_mas32,std::string mas_validation,bool mas_reuse_enabled)
{
    state.runtime_mas.reset();state.mas_graph_reuses=0;
    state.mas_validation=std::move(mas_validation);state.mas_reuse_enabled=mas_reuse_enabled;
    state.use_coarse_mas32=use_coarse_mas32;
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
    state.frozen_coarse_categories={false,false,false,false};
    state.frozen_coarse_samples=gipc::Json::array();
}

bool galerkin_adoption_enabled()
{
    return state.adoption_enabled;
}

void configure_direction_freeze(std::string directory,std::size_t after_update)
{
    state.direction_freeze_directory=std::move(directory);
    state.direction_freeze_after_update=after_update;
    state.direction_frozen=false;
}

bool direction_freeze_complete() { return state.direction_frozen; }

void freeze_accepted_direction(const GIPCTripletMatrix& matrix,
                               const double* rhs,std::size_t dofs)
{
    if(state.direction_freeze_directory.empty() || state.direction_frozen
       || state.updates<state.direction_freeze_after_update) return;
    if(!state.last_adoption.value("adopted",false) || state.candidate_dofs!=dofs)
        throw std::runtime_error("accepted direction capture has incompatible state");
    const std::filesystem::path directory(state.direction_freeze_directory);
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
    const auto criterion_snapshot=capture_criterion_snapshot(directory.string());
    const gipc::Json metadata={
        {"format","agipc_accepted_direction_v1"},{"update_index",state.updates},
        {"requested_after_update",state.direction_freeze_after_update},
        {"fine_dofs",dofs},{"fine_block_nodes",matrix.block_rows()},
        {"fine_unique_blocks",unique},{"fine_matrix_half_storage",true},
        {"block_values_layout","Eigen column-major 3x3 FP64"},
        {"mapping_fine_nodes",mapping.fine_nodes},{"mapping_coarse_nodes",mapping.coarse_nodes},
        {"coarse_block_nodes",coarse.block_rows()},{"coarse_unique_blocks",coarse_unique},
        {"coarse_solve",state.last["coarse_solve"]},
        {"post_correction",state.last["post_correction"]},{"adoption",state.last_adoption},
        {"coarse_preconditioner",state.use_coarse_mas32?"mas32":"block_jacobi"},
        {"mas_validation",state.mas_validation},{"mas_reuse_enabled",state.mas_reuse_enabled},
        {"criterion_snapshot",criterion_snapshot},
        {"stop_before_ccd_line_search_state_update",true},{"performance_claim",false}};
    std::ofstream output(directory/"metadata.json");
    output<<metadata.dump(2)<<'\n';output.close();
    if(!output) throw std::runtime_error("failed to write direction capture metadata");
    state.direction_frozen=true;
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

void record_linear_solve_timing(gipc::Json timing)
{
    if(state.last.is_null()) return;
    state.last["linear_system_timing_ms"]=timing;
    for(const char* key: {"fine_assembly","adaptive_pipeline","fine_preconditioner",
                          "candidate_decision","fallback_fine_solve","distribute",
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

gipc::Json galerkin_summary()
{
    if(state.last.is_null()) return nullptr;
    auto result=state.last;
    result["updates"]=state.updates;
    result["total_shadow_pipeline_ms"]=state.total_ms;
    result["total_stage_timing_ms"]={{"assembly",state.total_assembly_ms},
        {"coarse_solve",state.total_coarse_solve_ms},
        {"post_correction",state.total_post_correction_ms}};
    result["total_linear_system_timing_ms"]=state.total_linear_timing;
    result["direction_quality_evaluations"]=state.direction_quality_evaluations;
    result["residual_guard_restores"]=state.residual_guard_restores;
    result["coarse_preconditioner_requested"]=state.use_coarse_mas32?"mas32":"block_jacobi";
    result["mas_attempts"]=state.mas_attempts;
    result["mas_local_failures"]=state.mas_local_failures;
    result["total_mas_setup_wall_ms"]=state.total_mas_setup_ms;
    result["total_mas_validation_wall_ms"]=state.total_mas_validation_ms;
    result["mas_validation_mode"]=state.mas_validation;
    result["mas_reuse_enabled"]=state.mas_reuse_enabled;
    result["mas_graph_reuses"]=state.mas_graph_reuses;
    result["mas_failure_samples"]=state.mas_failure_samples;
    result["frozen_fallback_samples"]=state.frozen_fallback_samples;
    result["frozen_coarse_samples"]=state.frozen_coarse_samples;
    result["adoption_enabled"]=state.adoption_enabled;
    result["adoption_attempts"]=state.adoption_attempts;
    result["adoptions"]=state.adoptions;
    result["fallbacks"]=state.fallbacks;
    result["fallback_reason_counts"]=state.fallback_reason_counts;
    result["last_adoption"]=state.last_adoption;
    result["controls_solver"]=!state.last_adoption.is_null()
                              && state.last_adoption.value("adopted",false);
    return result;
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
         fine_fem_nodes,coarse_fem_nodes,1,1,5,true});

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
