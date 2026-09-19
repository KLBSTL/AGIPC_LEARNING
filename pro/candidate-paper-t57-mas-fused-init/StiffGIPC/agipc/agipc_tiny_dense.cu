#include "agipc_tiny_dense.cuh"
#include <cuda_tools/cuda_device_buffer.h>
#include <linear_system/linear_system/global_matrix.h>
#include <linear_system/utils/spmv.h>
#include <cusolverDn.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <Eigen/Dense>
#include <cmath>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace agipc {
namespace {
__global__ void expand_dense(const Eigen::Matrix3d* values,const int* rows,const int* cols,
                             double* dense,int unique,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=unique) return;
    const int br=rows[i],bc=cols[i];
    for(int r=0;r<3;++r) for(int c=0;c<3;++c) {
        const double value=values[i](r,c);
        dense[(3*bc+c)*n+3*br+r]=value;
        if(br!=bc) dense[(3*br+r)*n+3*bc+c]=value;
    }
}
__global__ void check_dense(const double* dense,int n,int* flags) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n*n) return;
    const double value=dense[i],other=dense[(i%n)*n+i/n];
    if(!isfinite(value)) atomicOr(flags,1);
    if(fabs(value-other)>1e-12*fmax(1.0,fmax(fabs(value),fabs(other)))) atomicOr(flags,2);
}
__global__ void residual_kernel(const double* rhs,const double* product,double* residual,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) residual[i]=rhs[i]-product[i];
}
double dot(const double* a,const double* b,int n) {
    return thrust::inner_product(thrust::device_ptr<const double>(a),
        thrust::device_ptr<const double>(a)+n,thrust::device_ptr<const double>(b),0.0);
}
}
struct TinyDenseSolver::Impl {
    cusolverDnHandle_t handle=nullptr;
    cudatool::CudaDeviceBuffer<double> dense,workspace,product,residual;
    cudatool::CudaDeviceBuffer<int> info{1},flags{1};
    ~Impl() { if(handle) cusolverDnDestroy(handle); }
};
TinyDenseSolver::TinyDenseSolver():impl(std::make_unique<Impl>()) {}
TinyDenseSolver::~TinyDenseSolver()=default;
gipc::Json TinyDenseSolver::solve(const GIPCTripletMatrix& matrix,const double* rhs,double* solution) {
    auto& p=*impl;
    const int n=3*matrix.block_rows(),unique=matrix.h_unique_key_number;
    gipc::Json result={{"attempted",true},{"converged",false},{"solver","fp64_gpu_cholesky"},
        {"preconditioner","none"},{"dofs",n},{"iterations",0},{"relative_tolerance",1e-3},
        {"regularization",false},{"true_residual_checked",false},{"spd_checked",false}};
    auto fail=[&](const char* reason) { result["failure_reason"]=reason; return result; };
    if(n<1 || n>96 || unique<1) return fail("dense_dimension_out_of_range");
    auto api=[&](cusolverStatus_t status) { result["cusolver_status"]=static_cast<int>(status); return status==CUSOLVER_STATUS_SUCCESS; };
    if(!p.handle) {
        if(!api(cusolverDnCreate(&p.handle))) return fail("dense_handle_failed");
    }
    if(!api(cusolverDnSetStream(p.handle,cudaStreamPerThread))) return fail("dense_stream_failed");
    p.dense.resize(n*n); p.dense.reset_zero(); p.flags.reset_zero();
    expand_dense<<<(unique+127)/128,128>>>(matrix.block_values(),matrix.block_row_indices(),matrix.block_col_indices(),p.dense.data(),unique,n);
    check_dense<<<(n*n+127)/128,128>>>(p.dense.data(),n,p.flags.data());
    CUDA_SAFE_CALL(cudaGetLastError());
    int flags=0;
    CUDA_SAFE_CALL(cudaMemcpy(&flags,p.flags.data(),sizeof(int),cudaMemcpyDeviceToHost));
    result["symmetry_relative_tolerance"]=1e-12;
    if(flags&1) return fail("dense_nonfinite_matrix");
    if(flags&2) return fail("dense_nonsymmetric_matrix");
    int size=0;
    if(!api(cusolverDnDpotrf_bufferSize(p.handle,CUBLAS_FILL_MODE_LOWER,n,p.dense.data(),n,&size))) return fail("dense_workspace_query_failed");
    if(size<0) return fail("dense_invalid_workspace");
    p.workspace.resize(size); p.product.resize(n); p.residual.resize(n);
    result["workspace_bytes"]=p.workspace.capacity()*sizeof(double);
    result["tracked_allocated_bytes"]=(p.dense.capacity()+p.workspace.capacity()+p.product.capacity()+p.residual.capacity())*sizeof(double)
        +(p.info.capacity()+p.flags.capacity())*sizeof(int);
    result["memory_scope"]="dense matrix, cuSolver workspace, product, residual and flags; excludes cuSolver handle/internal allocations";
    if(!api(cusolverDnDpotrf(p.handle,CUBLAS_FILL_MODE_LOWER,n,p.dense.data(),n,p.workspace.data(),size,p.info.data()))) return fail("dense_factor_api_failed");
    int info=0;
    CUDA_SAFE_CALL(cudaMemcpy(&info,p.info.data(),sizeof(int),cudaMemcpyDeviceToHost));
    result["factor_info"]=info;
    if(info!=0) return fail(info>0?"dense_not_spd":"dense_factor_invalid_argument");
    result["spd_checked"]=true;
    CUDA_SAFE_CALL(cudaMemcpy(solution,rhs,n*sizeof(double),cudaMemcpyDeviceToDevice));
    if(!api(cusolverDnDpotrs(p.handle,CUBLAS_FILL_MODE_LOWER,n,1,p.dense.data(),n,solution,n,p.info.data()))) return fail("dense_solve_api_failed");
    CUDA_SAFE_CALL(cudaMemcpy(&info,p.info.data(),sizeof(int),cudaMemcpyDeviceToHost));
    result["solve_info"]=info;
    if(info!=0) return fail("dense_solve_invalid_argument");
    gipc::Spmv spmv;
    spmv.warp_reduce_sym_spmv(1.0,const_cast<Eigen::Matrix3d*>(matrix.block_values()),
        const_cast<int*>(matrix.block_row_indices()),const_cast<int*>(matrix.block_col_indices()),unique,
        cudatool::CDenseVectorView<double>(solution,n),0.0,cudatool::DenseVectorView<double>(p.product.data(),n));
    residual_kernel<<<(n+127)/128,128>>>(rhs,p.product.data(),p.residual.data(),n);
    CUDA_SAFE_CALL(cudaGetLastError());
    const double initial2=dot(rhs,rhs,n),final2=dot(p.residual.data(),p.residual.data(),n),descent=dot(rhs,solution,n);
    result["true_residual_checked"]=true;
    result["initial_residual_norm"]=std::sqrt(initial2);
    result["final_residual_norm"]=std::sqrt(final2);
    result["rhs_dot_direction"]=descent;
    if(!std::isfinite(initial2) || !std::isfinite(final2) || !std::isfinite(descent)) return fail("dense_nonfinite_solution_or_rhs");
    if(final2>initial2*1e-6*(1.0+1e-6)*(1.0+1e-6)) return fail("dense_true_residual_not_converged");
    if(initial2>0 && descent<=0) return fail("dense_nonpositive_descent");
    result["converged"]=true; result["failure_reason"]="";
    return result;
}

gipc::Json tiny_dense_self_test() {
    using cudatool::CudaDeviceBuffer;
    TinyDenseSolver solver;
    gipc::Json cases=gipc::Json::array();
    auto test=[&](const Eigen::MatrixXd& a,const Eigen::VectorXd& rhs,const char* label,bool expect_success) {
        const int n=a.rows(),blocks=n/3;
        std::vector<Eigen::Matrix3d> values; std::vector<int> rows,cols;
        for(int r=0;r<blocks;++r) for(int c=0;c<=r;++c) { values.push_back(a.block<3,3>(3*r,3*c)); rows.push_back(r); cols.push_back(c); }
        GIPCTripletMatrix matrix; matrix.init_var(); matrix.reshape(blocks,blocks);
        matrix.h_unique_key_number=values.size();
        matrix.m_block_values=values; matrix.m_block_row_indices=rows; matrix.m_block_col_indices=cols;
        CudaDeviceBuffer<double> device_rhs(std::vector<double>(rhs.data(),rhs.data()+n)),solution(n);
        auto result=solver.solve(matrix,device_rhs.data(),solution.data());
        bool passed=result.value("converged",false)==expect_success;
        if(expect_success && result.value("converged",false)) {
            std::vector<double> x; solution.copy_to_host(x);
            Eigen::Map<const Eigen::VectorXd> actual(x.data(),n);
            const Eigen::VectorXd oracle=a.llt().solve(rhs);
            const double error=(actual-oracle).norm()/std::max(1.0,oracle.norm());
            const double residual=(a*actual-rhs).norm()/std::max(1.0,rhs.norm());
            passed=passed && std::isfinite(error) && error<1e-10 && residual<1e-10;
            result["cpu_llt_relative_solution_error"]=error; result["cpu_true_residual_ratio"]=residual;
        }
        result["case"]=label; result["passed"]=passed; cases.push_back(result);
        if(!passed) throw std::runtime_error(std::string("tiny dense CPU oracle test failed: ")+label);
    };
    for(int n:{3,21,24,48,96}) {
        Eigen::MatrixXd m(n,n);
        for(int r=0;r<n;++r) for(int c=0;c<n;++c) m(r,c)=std::sin(.71*r+.33*c+.2*r*c)/n;
        const Eigen::MatrixXd a=m.transpose()*m+Eigen::MatrixXd::Identity(n,n)*.5;
        const Eigen::VectorXd rhs=Eigen::VectorXd::LinSpaced(n,-1,2);
        test(a,rhs,"coupled_spd",true); test(a,Eigen::VectorXd::Zero(n),"zero_rhs_spd",true);
    }
    Eigen::MatrixXd a=Eigen::MatrixXd::Identity(3,3); Eigen::VectorXd rhs=Eigen::VectorXd::Ones(3);
    a(0,0)=-1; test(a,rhs,"indefinite",false);
    a(0,0)=0; test(a,rhs,"singular",false);
    a(0,0)=std::numeric_limits<double>::infinity(); test(a,rhs,"nonfinite_matrix",false);
    a.setIdentity(); a(0,1)=.1; test(a,rhs,"asymmetric_diagonal_block",false);
    a.setIdentity(); rhs[0]=std::numeric_limits<double>::quiet_NaN(); test(a,rhs,"nonfinite_rhs",false);
    return {{"passed",true},{"cases",cases},{"candidate_default_enabled",false}};
}
}
