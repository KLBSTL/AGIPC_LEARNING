#include <linear_system/utils/spmv.h>
#include <cuda_tools/cuda_all.h>
#include <cuda_tools/cuda_tools.h>
#include <cub/warp/warp_reduce.cuh>

#include <algorithm>
#include <cmath>
#include <vector>

namespace gipc
{
namespace
{
__global__ void scale_y_kernel(int size, Float b, cudatool::DenseVectorViewer<Float> y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size)
        y(i) = b * y(i);
}

__global__ void fill_y_zero_kernel(int size, Float* y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size)
        y[i] = Float(0);
}

__global__ void legacy_sym_spmv_kernel(Float            a,
                                       Eigen::Matrix3d* Mats3,
                                       int*             rows,
                                       int*             cols,
                                       int              triplet_count,
                                       cudatool::CDenseVectorViewer<Float> x,
                                       cudatool::DenseVectorViewer<Float>  y)
{
    const int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index >= triplet_count)
        return;

    constexpr int N     = 3;
    const int     i     = rows[index];
    const int     j     = cols[index];
    const auto    block = Mats3[index];

    const Vector3 ij = a * block * x.segment<N>(j * N).as_eigen();
    y.segment<N>(i * N).atomic_add(ij.eval());

    if(i != j)
    {
        const Vector3 ji = a * block.transpose() * x.segment<N>(i * N).as_eigen();
        y.segment<N>(j * N).atomic_add(ji.eval());
    }
}

__global__ void warp_reduce_sym_spmv_kernel(Float            a,
                                            Eigen::Matrix3d* Mats3,
                                            int*             rows,
                                            int*             cols,
                                            int              triplet_count,
                                            cudatool::CDenseVectorViewer<Float> x,
                                            Float                                 b,
                                            cudatool::DenseVectorViewer<Float>  y)
{
    using WarpReduceFloat = cub::WarpReduce<Float, 32>;
    auto global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const bool is_valid   = global_thread_id < triplet_count;
    auto thread_id_in_block = threadIdx.x;
    auto warp_id            = thread_id_in_block / 32;
    auto lane_id            = thread_id_in_block & (32 - 1);

    __shared__ WarpReduceFloat::TempStorage temp_storage_float[256 / 32];

    int     prev_i = -1;
    int     i      = -1;
    char    flags;
    Vector3 vec;

    // set the previous row index
    if(is_valid && global_thread_id > 0)
        prev_i = rows[global_thread_id - 1];

    if(is_valid)
    {
        i                = rows[global_thread_id];
        auto j           = cols[global_thread_id];
        auto block_value = Mats3[global_thread_id];
        vec = block_value * x.segment<3>(j * 3).as_eigen();

        if(i != j)  // process lower triangle
        {
            Vector3 vec_ = a * block_value.transpose() * x.segment<3>(i * 3).as_eigen();
            y.segment<3>(j * 3).atomic_add(vec_);
        }
    }
    else
    {
        vec.setZero();
    }

    if((lane_id == 0) || (prev_i != i))
        flags = 1;
    else
        flags = 0;

    vec.x() = WarpReduceFloat(temp_storage_float[warp_id])
                  .HeadSegmentedReduce(vec.x(), flags, cudatool::Plus<Float>{});
    vec.y() = WarpReduceFloat(temp_storage_float[warp_id])
                  .HeadSegmentedReduce(vec.y(), flags, cudatool::Plus<Float>{});
    vec.z() = WarpReduceFloat(temp_storage_float[warp_id])
                  .HeadSegmentedReduce(vec.z(), flags, cudatool::Plus<Float>{});

    if(flags && is_valid)
    {
        auto seg_y  = y.segment<3>(i * 3);
        auto result = a * vec;
        seg_y.atomic_add(result.eval());
    }
}

__global__ void hybrid_sym_spmv_kernel(Float            a,
                                        Eigen::Matrix3d* Mats3,
                                        int*             rows,
                                        int*             cols,
                                        int              triplet_count,
                                        cudatool::CDenseVectorViewer<Float> x,
                                        cudatool::DenseVectorViewer<Float>  y,
                                        int short_row_threshold)
{
    using WarpReduceFloat = cub::WarpReduce<Float, 32>;
    const int global_thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const bool is_valid = global_thread_id < triplet_count;
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x / 32;
    const unsigned warp_mask = __activemask();

    __shared__ WarpReduceFloat::TempStorage temp_storage_float[256 / 32];

    int i = -1;
    int prev_i = -1;
    Vector3 upper;
    upper.setZero();
    if(is_valid && global_thread_id > 0)
        prev_i = rows[global_thread_id - 1];
    if(is_valid)
    {
        i = rows[global_thread_id];
        const int j = cols[global_thread_id];
        const auto block_value = Mats3[global_thread_id];
        upper = block_value * x.segment<3>(j * 3).as_eigen();
        if(i != j)
        {
            const Vector3 lower =
                a * block_value.transpose() * x.segment<3>(i * 3).as_eigen();
            y.segment<3>(j * 3).atomic_add(lower.eval());
        }
    }

    const bool is_head = is_valid && (lane_id == 0 || prev_i != i);
    const unsigned valid_mask = __ballot_sync(warp_mask, is_valid);
    const unsigned head_mask = __ballot_sync(warp_mask, is_head);

    int segment_length = 0;
    if(is_valid)
    {
        const unsigned lane_mask = lane_id == 31
                                       ? 0xffffffffu
                                       : ((1u << (lane_id + 1)) - 1u);
        const unsigned preceding_heads = head_mask & lane_mask;
        const int head_lane = 31 - __clz(preceding_heads);
        const unsigned head_lane_mask = head_lane == 31
                                            ? 0xffffffffu
                                            : ((1u << (head_lane + 1)) - 1u);
        const unsigned following_heads = head_mask & ~head_lane_mask;
        const int end_lane = following_heads != 0
                                 ? __ffs(following_heads) - 1
                                 : 32 - __clz(valid_mask);
        segment_length = end_lane - head_lane;
    }

    const bool use_reduction = is_valid && segment_length > short_row_threshold;
    if(is_valid && !use_reduction)
    {
        const Vector3 direct = a * upper;
        y.segment<3>(i * 3).atomic_add(direct.eval());
    }

    if(!__any_sync(warp_mask, use_reduction))
        return;

    Vector3 reduced = use_reduction ? upper : Vector3::Zero();
    const char flag = is_head ? 1 : 0;
    reduced.x() = WarpReduceFloat(temp_storage_float[warp_id])
                      .HeadSegmentedReduce(reduced.x(), flag, cudatool::Plus<Float>{});
    reduced.y() = WarpReduceFloat(temp_storage_float[warp_id])
                      .HeadSegmentedReduce(reduced.y(), flag, cudatool::Plus<Float>{});
    reduced.z() = WarpReduceFloat(temp_storage_float[warp_id])
                      .HeadSegmentedReduce(reduced.z(), flag, cudatool::Plus<Float>{});

    if(is_head && use_reduction)
    {
        const Vector3 result = a * reduced;
        y.segment<3>(i * 3).atomic_add(result.eval());
    }
}
}  // namespace

void Spmv::legacy_sym_spmv(Float                         a,
                           Eigen::Matrix3d*              triplet_values,
                           int*                          row_ids,
                           int*                          col_ids,
                           int                           triplet_count,
                           cudatool::CDenseVectorView<Float> x,
                           Float                         b,
                           cudatool::DenseVectorView<Float>  y)
{
    using namespace cudatool;
    if(b != 0)
        LaunchCudaKernal_default(y.size(), 256, 0, scale_y_kernel, y.size(), b, y.viewer());
    else
        LaunchCudaKernal_default(y.size(), 256, 0, fill_y_zero_kernel, y.size(), y.data());

    LaunchCudaKernal_default(triplet_count,
                             256,
                             0,
                             legacy_sym_spmv_kernel,
                             a,
                             triplet_values,
                             row_ids,
                             col_ids,
                             triplet_count,
                             x.cviewer(),
                             y.viewer());
}

void Spmv::warp_reduce_sym_spmv(Float                         a,
                                Eigen::Matrix3d*              triplet_values,
                                int*                          row_ids,
                                int*                          col_ids,
                                int                           triplet_count,
                                cudatool::CDenseVectorView<Float> x,
                                Float                         b,
                                cudatool::DenseVectorView<Float>  y)
{
    using namespace cudatool;
    constexpr int N = 3;

    if(b != 0)
    {
        LaunchCudaKernal_default(y.size(), 256, 0, scale_y_kernel, y.size(), b, y.viewer());
    }
    else
    {
        LaunchCudaKernal_default(y.size(), 256, 0, fill_y_zero_kernel, y.size(), y.data());
    }

    constexpr int block_dim = 256;
    LaunchCudaKernal_default(triplet_count,
                             block_dim,
                             0,
                             warp_reduce_sym_spmv_kernel,
                             a,
                             triplet_values,
                             row_ids,
                             col_ids,
                             triplet_count,
                             x.cviewer(),
                             b,
                             y.viewer());
}

void Spmv::hybrid_sym_spmv(Float                         a,
                            Eigen::Matrix3d*              triplet_values,
                            int*                          row_ids,
                            int*                          col_ids,
                            int                           triplet_count,
                            cudatool::CDenseVectorView<Float> x,
                            Float                         b,
                            cudatool::DenseVectorView<Float>  y,
                            int                           short_row_threshold)
{
    using namespace cudatool;
    if(b != 0)
        LaunchCudaKernal_default(y.size(), 256, 0, scale_y_kernel, y.size(), b, y.viewer());
    else
        LaunchCudaKernal_default(y.size(), 256, 0, fill_y_zero_kernel, y.size(), y.data());

    LaunchCudaKernal_default(triplet_count,
                             256,
                             0,
                             hybrid_sym_spmv_kernel,
                             a,
                             triplet_values,
                             row_ids,
                             col_ids,
                             triplet_count,
                             x.cviewer(),
                             y.viewer(),
                             short_row_threshold);
}

SpmvEquivalenceResult run_spmv_equivalence_self_test()
{
    constexpr int    block_rows = 20;
    constexpr int    dof        = block_rows * 3;
    constexpr Float  a          = 1.25;
    constexpr Float  b          = -0.5;
    constexpr double tolerance  = 1e-10;

    std::vector<Eigen::Matrix3d> values;
    std::vector<int>             rows;
    std::vector<int>             cols;
    values.reserve(10);
    rows.reserve(10);
    cols.reserve(10);

    for(int i = 0; i < block_rows; ++i)
    {
        for(int j = 0; j <= i; ++j)
        {
            Eigen::Matrix3d block;
            if(i == j)
            {
                block << 6.0 + i, 0.03, -0.02,
                         0.03, 7.0 + i, 0.01,
                        -0.02, 0.01, 8.0 + i;
            }
            else
            {
                const double s = 0.02 * (i + j + 1);
                block << s, -0.5 * s, 0.25 * s,
                         0.4 * s, -0.3 * s, 0.1 * s,
                        -0.2 * s, 0.35 * s, 0.6 * s;
            }
            values.push_back(block);
            rows.push_back(i);
            cols.push_back(j);
        }
    }

    std::vector<Float> x(dof);
    std::vector<Float> y_initial(dof);
    for(int i = 0; i < dof; ++i)
    {
        x[i]         = 0.125 * (i + 1) - 0.4;
        y_initial[i] = 0.05 * (i - 3);
    }

    std::vector<Float> reference(dof);
    for(int i = 0; i < dof; ++i)
        reference[i] = b * y_initial[i];
    for(size_t k = 0; k < values.size(); ++k)
    {
        const int i = rows[k];
        const int j = cols[k];
        Eigen::Map<const Vector3> xj(x.data() + 3 * j);
        Eigen::Map<Vector3>       yi(reference.data() + 3 * i);
        yi += a * values[k] * xj;
        if(i != j)
        {
            Eigen::Map<const Vector3> xi(x.data() + 3 * i);
            Eigen::Map<Vector3>       yj(reference.data() + 3 * j);
            yj += a * values[k].transpose() * xi;
        }
    }

    cudatool::DeviceBuffer<Eigen::Matrix3d> device_values(values);
    cudatool::DeviceBuffer<int>             device_rows(rows);
    cudatool::DeviceBuffer<int>             device_cols(cols);
    cudatool::DeviceDenseVector<Float>      device_x;
    cudatool::DeviceDenseVector<Float>      legacy_y;
    cudatool::DeviceDenseVector<Float>      srbk_y;
    cudatool::DeviceDenseVector<Float>      hybrid8_y;
    cudatool::DeviceDenseVector<Float>      hybrid16_y;
    device_x.copy_from(x);
    legacy_y.copy_from(y_initial);
    srbk_y.copy_from(y_initial);
    hybrid8_y.copy_from(y_initial);
    hybrid16_y.copy_from(y_initial);

    Spmv spmv;
    spmv.legacy_sym_spmv(a,
                         device_values.data(),
                         device_rows.data(),
                         device_cols.data(),
                         static_cast<int>(values.size()),
                         device_x.cview(),
                         b,
                         legacy_y.view());
    spmv.warp_reduce_sym_spmv(a,
                              device_values.data(),
                              device_rows.data(),
                              device_cols.data(),
                              static_cast<int>(values.size()),
                              device_x.cview(),
                              b,
                              srbk_y.view());
    spmv.hybrid_sym_spmv(a,
                         device_values.data(),
                         device_rows.data(),
                         device_cols.data(),
                         static_cast<int>(values.size()),
                         device_x.cview(),
                         b,
                         hybrid8_y.view(),
                         8);
    spmv.hybrid_sym_spmv(a,
                         device_values.data(),
                         device_rows.data(),
                         device_cols.data(),
                         static_cast<int>(values.size()),
                         device_x.cview(),
                         b,
                         hybrid16_y.view(),
                         16);
    cudaDeviceSynchronize();

    std::vector<Float> legacy_result;
    std::vector<Float> srbk_result;
    std::vector<Float> hybrid8_result;
    std::vector<Float> hybrid16_result;
    legacy_y.copy_to(legacy_result);
    srbk_y.copy_to(srbk_result);
    hybrid8_y.copy_to(hybrid8_result);
    hybrid16_y.copy_to(hybrid16_result);

    SpmvEquivalenceResult result;
    result.block_rows    = block_rows;
    result.triplet_count = static_cast<int>(values.size());
    double reference_scale = 1.0;
    for(int i = 0; i < dof; ++i)
    {
        reference_scale = std::max(reference_scale, std::abs(reference[i]));
        result.legacy_reference_max_abs_error =
            std::max(result.legacy_reference_max_abs_error,
                     std::abs(legacy_result[i] - reference[i]));
        result.srbk_reference_max_abs_error =
            std::max(result.srbk_reference_max_abs_error,
                     std::abs(srbk_result[i] - reference[i]));
        result.legacy_srbk_max_abs_error =
            std::max(result.legacy_srbk_max_abs_error,
                     std::abs(legacy_result[i] - srbk_result[i]));
        result.hybrid8_reference_max_abs_error =
            std::max(result.hybrid8_reference_max_abs_error,
                     std::abs(hybrid8_result[i] - reference[i]));
        result.hybrid16_reference_max_abs_error =
            std::max(result.hybrid16_reference_max_abs_error,
                     std::abs(hybrid16_result[i] - reference[i]));
    }
    result.legacy_srbk_max_relative_error =
        result.legacy_srbk_max_abs_error / reference_scale;
    result.passed = std::isfinite(result.legacy_reference_max_abs_error)
                    && std::isfinite(result.srbk_reference_max_abs_error)
                    && result.legacy_reference_max_abs_error <= tolerance
                    && result.srbk_reference_max_abs_error <= tolerance
                    && result.hybrid8_reference_max_abs_error <= tolerance
                    && result.hybrid16_reference_max_abs_error <= tolerance
                    && result.legacy_srbk_max_abs_error <= tolerance;
    return result;
}
}  // namespace gipc
