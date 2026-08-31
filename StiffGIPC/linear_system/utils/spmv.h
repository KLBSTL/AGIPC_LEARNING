#pragma once
#include <gipc/spmv_mode.h>
#include <gipc/type_define.h>

#include <cuda_tools/cuda_all.h>

namespace gipc
{
struct SpmvEquivalenceResult
{
    bool   passed                         = false;
    int    block_rows                    = 0;
    int    triplet_count                 = 0;
    double legacy_reference_max_abs_error = 0.0;
    double srbk_reference_max_abs_error   = 0.0;
    double legacy_srbk_max_abs_error      = 0.0;
    double legacy_srbk_max_relative_error = 0.0;
    double hybrid8_reference_max_abs_error = 0.0;
    double hybrid16_reference_max_abs_error = 0.0;
};

class Spmv
{
  public:

    void legacy_sym_spmv(Float                         a,
                         Eigen::Matrix3d*              triplet_values,
                         int*                          row_ids,
                         int*                          col_ids,
                         int                           triplet_count,
                         cudatool::CDenseVectorView<Float> x,
                         Float                         b,
                         cudatool::DenseVectorView<Float>  y);

    void warp_reduce_sym_spmv(Float                         a,
                              Eigen::Matrix3d*              triplet_values,
                              int*                          row_ids,
                              int*                          col_ids,
                              int                           triplet_count,
                              cudatool::CDenseVectorView<Float> x,
                              Float                         b,
                              cudatool::DenseVectorView<Float>  y);

    void hybrid_sym_spmv(Float                         a,
                          Eigen::Matrix3d*              triplet_values,
                          int*                          row_ids,
                          int*                          col_ids,
                          int                           triplet_count,
                          cudatool::CDenseVectorView<Float> x,
                          Float                         b,
                          cudatool::DenseVectorView<Float>  y,
                          int                           short_row_threshold);
};

SpmvEquivalenceResult run_spmv_equivalence_self_test();
}  // namespace gipc
