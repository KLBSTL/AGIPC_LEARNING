//
// TraditionalMAS32Preconditioner.cuh
// Traditional GPU MAS32 adapter for the c499 StiffGIPC linear system.
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once

#include "device_fem_data.cuh"
#include "eigen_data.h"
#include <cuda_tools/cuda_all.h>
#include "linear_system/linear_system/global_matrix.h"
#include <gipc/utils/json.h>

namespace gpu_mas32
{
struct HierarchySelfTestResult
{
    bool passed = false;
    int  valid_nodes = 0;
    int  padded_nodes = 0;
    int  expected_components = 0;
    int  actual_components = 0;
    int  fine_mask_mismatches = 0;
    int  coarse_mapping_mismatches = 0;
    int  going_next_mismatches = 0;
};

HierarchySelfTestResult run_hierarchy_self_test();
gipc::Json run_local_diagnostics_gpu_self_test();
bool local_diagnostics_agree(const gipc::Json& gpu,const gipc::Json& cpu);

class TraditionalMAS32Preconditioner
{

    int totalNodes              = 0;
    int totalMapNodes           = 0;
    int levelnum                = 0;
    int collision_node_Offset   = 0;
    int totalNumberClusters     = 0;
    //int bankSize;
    int2  h_clevelSize          = {};
    int4* _collisonPairs        = nullptr;

    int2*               d_levelSize          = nullptr;
    int*                d_coarseSpaceTables  = nullptr;
    int*                d_prefixOriginal     = nullptr;
    int*                d_prefixSumOriginal  = nullptr;
    int*                d_goingNext          = nullptr;
    int*                d_denseLevel         = nullptr;
    __GEIGEN__::itable* d_coarseTable        = nullptr;
    unsigned int*       d_fineConnectMask    = nullptr;
    unsigned int*       d_nextConnectMask    = nullptr;
    unsigned int*       d_nextPrefix         = nullptr;
    unsigned int*       d_nextPrefixSum      = nullptr;


    __GEIGEN__::GPUMas32MatrixT*    d_MatMas       = nullptr;
    __GEIGEN__::GPUMas32MatrixSymT* d_inverseMatMas = nullptr;
    __GEIGEN__::GPUMas32MatrixSymf* d_precondMatMas = nullptr;
    Eigen::Vector3f*           d_multiLevelR   = nullptr;
    Precision_T3*              d_multiLevelZ   = nullptr;
    bool                        factorized_local_solve = false;
    int*                        d_factor_status = nullptr;

  public:
    int           neighborListSize    = 0;
    unsigned int* d_neighborList      = nullptr;
    unsigned int* d_neighborStart     = nullptr;
    unsigned int* d_neighborStartTemp = nullptr;
    unsigned int* d_neighborNum       = nullptr;
    unsigned int* d_neighborListInit  = nullptr;
    unsigned int* d_neighborNumInit   = nullptr;
    int*          d_partId_map_real   = nullptr;
    int*          d_real_map_partId   = nullptr;

  public:
    void initPreconditioner_Neighbor(int   vertNum,
                                     int   mCollision_node_offset,
                                     int   totalNeighborNum,
                                     int4* m_collisonPairs,
                                     int   partMapSize);
    void computeNumLevels(int vertNum);  // called in initPreconditioner_Neighbor

    void initPreconditioner_Matrix();


    int  ReorderRealtime(int cpNum);
    void BuildConnectMaskL0();           // called in ReorderRealtime
    void PreparePrefixSumL0();           // called in ReorderRealtime
    void BuildLevel1();                  // called in ReorderRealtime
    void BuildConnectMaskLx(int level);  // called in ReorderRealtime
    void NextLevelCluster(int level);    // called in ReorderRealtime
    void PrefixSumLx(int level);         // called in ReorderRealtime
    void ComputeNextLevel(int level);    // called in ReorderRealtime
    void AggregationKernel();            // called in ReorderRealtime
    void BuildCollisionConnection(unsigned int* connectionMsk,
                                  int*          coarseTableSpace,
                                  int           level,
                                  int cpNum);  // called in ReorderRealtime

    void setPreconditioner_bcoo(Eigen::Matrix3d* triplet_values,
                                int*             row_ids,
                                int*             col_ids,
                                uint32_t*        indices,
                                int              offset,
                                int              triplet_num,
                                int              cpNum);
    void PrepareHessian_bcoo(Eigen::Matrix3d* triplet_values,
                             int*             row_ids,
                             int*             col_ids,
                             uint32_t*        indices,
                             int              offset,
                             int              triplet_number);

    void preconditioning(const double3* R, double3* Z);
    void set_factorized_local_solve(bool enabled) { factorized_local_solve = enabled; }
    bool uses_factorized_local_solve() const { return factorized_local_solve; }
    gipc::Json factor_diagnostics() const;
    gipc::Json numerical_diagnostics(const double3* R) const;
    gipc::Json local_diagnostics_gpu() const;
    // Caller must prove the node count, BCOO structure and graph are unchanged.
    void refresh_fixed_graph_bcoo(Eigen::Matrix3d* values,int* rows,int* cols,
                                 uint32_t* indices,int offset,int count);
    void BuildMultiLevelR(const double3* R);  // called in preconditioning
    void SchwarzLocalXSym();                  // called in preconditioning
    void SchwarzLocalXSym_block3();                  // called in preconditioning
    void SchwarzLocalXSym_sym();           // called in preconditioning
    void CollectFinalZ(double3* Z);           // called in preconditioning

    void FreeMAS();
};
}  // namespace gpu_mas32
