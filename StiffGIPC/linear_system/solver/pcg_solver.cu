#include <linear_system/solver/pcg_solver.h>
#include <gipc/utils/timer.h>
#include <gipc/statistics.h>
#include <cuda_tools/cuda_tools.h>
#include <cmath>
#include <limits>



__global__ void PCG_vdv_Reduction(double* squeue, const double* a, const double* b, int numbers)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;

    extern __shared__ double tep[];

    if(idx >= numbers)
        return;

    double temp = a[idx] * b[idx];

    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    //double nextTp;
    int    warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        temp = tep[threadIdx.x];
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        squeue[blockIdx.x] = temp;
    }
}



__global__ void add_reduction(double* mem, int numbers)
{
    int idof = blockIdx.x * blockDim.x;
    int idx  = threadIdx.x + idof;
    extern __shared__ double tep[];
    if(idx >= numbers)
        return;
    double temp = mem[idx];
    int    warpTid = threadIdx.x % 32;
    int    warpId  = (threadIdx.x >> 5);
    int    warpNum;
    if(blockIdx.x == gridDim.x - 1)
    {
        warpNum = ((numbers - idof + 31) >> 5);
    }
    else
    {
        warpNum = ((blockDim.x) >> 5);
    }
    for(int i = 1; i < 32; i = (i << 1))
    {
        temp += __shfl_down_sync(0xffffffff, temp, i);
    }
    if(warpTid == 0)
    {
        tep[warpId] = temp;
    }
    __syncthreads();
    if(threadIdx.x >= warpNum)
        return;
    if(warpNum > 1)
    {
        temp = tep[threadIdx.x];
        for(int i = 1; i < warpNum; i = (i << 1))
        {
            temp += __shfl_down_sync(0xffffffff, temp, i);
        }
    }
    if(threadIdx.x == 0)
    {
        mem[blockIdx.x] = temp;
    }
}



__global__ void update_vector_dx_r(
    double* dx, double* r, const double* c, const double* q, double alpha, int numbers)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numbers)
        return;
    dx[idx] = dx[idx] + alpha * c[idx];
    r[idx]  = r[idx] - alpha * q[idx];
}

__global__ void update_vector_c(
    double* c, const double* s, double beta, int numbers)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= numbers)
        return;
    c[idx] = s[idx] + beta * c[idx];
}


double My_PCG_General_v_v_Reduction_Algorithm(double* temp, double* A, double* B, int vertexNum)
{

    int numbers = vertexNum;
    if(numbers < 1)
        return 0;
    const unsigned int threadNum = 256;
    int                blockNum  = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(double) * (threadNum >> 5);
    PCG_vdv_Reduction<<<blockNum, threadNum, sharedMsize>>>(temp, A, B, numbers);


    numbers  = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while(numbers > 1)
    {
        add_reduction<<<blockNum, threadNum, sharedMsize>>>(temp, numbers);
        numbers  = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    double result;
    cudaMemcpy(&result, temp, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

namespace gipc
{
PCGSolver::PCGSolver(const PCGSolverConfig& cfg)
    : m_config(cfg)
{
}
SizeT PCGSolver::solve(cudatool::DenseVectorView<Float> x, cudatool::CDenseVectorView<Float> b)
{
    Timer timer{"pcg"};

    x.buffer_view().fill(0);
    z.resize(b.size());
    p.resize(b.size());
    r.resize(b.size());
    //temp.resize(b.size());
    Ap.resize(b.size());
    auto iter = pcg(x, b, m_config.max_iter_ratio * b.size());

    return iter;
}


Json PCGSolver::trace(cudatool::DenseVectorView<Float>  x,
                      cudatool::CDenseVectorView<Float> b,
                      SizeT                             iteration_count)
{
    x.buffer_view().fill(0);
    z.resize(b.size());
    p.resize(b.size());
    r.resize(b.size());
    Ap.resize(b.size());
    Json result;
    result["iterations"] = Json::array();
    result["requested_iterations"] = iteration_count;
    result["completed_iterations"] = pcg(x, b, iteration_count + 1, &result);
    return result;
}

SizeT PCGSolver::pcg(cudatool::DenseVectorView<Float>  x,
                     cudatool::CDenseVectorView<Float> b,
                     SizeT                             max_iter,
                     Json*                             trace)
{
    SizeT k = 0;

    r.buffer_view().copy_from(b.buffer_view());

    Float alpha, beta, rz, rz0;

    {
        //Timer timer{"preconditioner"};
        apply_preconditioner(z, r);
    }

    {
        //Timer timer{"dot"};
        rz = My_PCG_General_v_v_Reduction_Algorithm(p.buffer_view().data(),
                                                    r.buffer_view().data(),
                                                    z.buffer_view().data(),
                                                    z.size());
    }

    p   = z;
    rz0 = rz;

    auto residual_norm = [&]() {
        std::vector<Float> host_r;
        r.copy_to(host_r);
        long double sum = 0.0;
        for(const auto value : host_r)
            sum += static_cast<long double>(value) * value;
        return std::sqrt(static_cast<double>(sum));
    };
    if(trace)
    {
        (*trace)["initial_residual_norm"] = residual_norm();
        (*trace)["initial_rtz"] = rz;
    }

    for(k = 1; k < max_iter; ++k)
    {
        Float dot_res = 0;
        {
            //Timer timer{"spmv"};
            // Ap = A * p
            spmv(p.cview(), Ap.view());
        }

        {
            //Timer timer{"dot"};

            dot_res = My_PCG_General_v_v_Reduction_Algorithm(
                z.buffer_view().data(),
                p.buffer_view().data(),
                Ap.buffer_view().data(),
                z.size());

            alpha = rz / dot_res;

            if(trace)
                (*trace)["iterations"].push_back({{"iteration", k},
                                                    {"rtz_before", rz},
                                                    {"p_t_a_p", dot_res},
                                                    {"alpha", alpha}});
        }

        {
            //Timer timer{"axpby"};
            LaunchCudaKernal_default(z.size(),
                                     256,
                                     0,
                                     update_vector_dx_r,
                                     x.buffer_view().data(),
                                     r.buffer_view().data(),
                                     (const double*)p.buffer_view().data(),
                                     (const double*)Ap.buffer_view().data(),
                                     alpha,
                                     (int)z.size());
        }

        if(trace)
            (*trace)["iterations"].back()["residual_norm"] = residual_norm();

        if(std::abs(rz) <= m_config.global_tol_rate * rz0)
        {
            if(trace)
                (*trace)["iterations"].back()["existing_convergence_gate"] = true;
            break;
        }

        {
            //Timer timer{"preconditioner"};
            apply_preconditioner(z, r);
        }

        Float rz_new = 0;
        {
            //Timer timer{"dot"};
            rz_new = My_PCG_General_v_v_Reduction_Algorithm(Ap.buffer_view().data(),
                                                            r.buffer_view().data(),
                                                            z.buffer_view().data(),
                                                            z.size());
        }

        beta = rz_new / rz;

        if(trace)
        {
            auto& record = (*trace)["iterations"].back();
            record["rtz_after"] = rz_new;
            record["beta"] = beta;
            record["finite"] = std::isfinite(alpha) && std::isfinite(beta)
                               && std::isfinite(dot_res) && std::isfinite(rz_new);
            record["spd"] = dot_res > 0.0 && rz_new > 0.0;
        }

        {
            //Timer timer{"axpby"};
            LaunchCudaKernal_default(z.size(),
                                     256,
                                     0,
                                     update_vector_c,
                                     p.buffer_view().data(),
                                     (const double*)z.buffer_view().data(),
                                     beta,
                                     (int)z.size());
        }

        rz = rz_new;
    }

    return k;
}

}  // namespace gipc
