//
// PCG_SOLVER.cuh
// GIPC
//
// created by Kemeng Huang on 2022/12/01
// Copyright (c) 2024 Kemeng Huang. All rights reserved.
//

#pragma once
#ifndef _PCG_SOLVER_CUH_
#define _PCG_SOLVER_CUH_
#include <cuda_runtime.h>
#include "MASPreconditioner.cuh"
#include "TraditionalMAS32Preconditioner.cuh"

class PCG_Data
{
  public:
    double*                 squeue;
    double3*                dx;
    MASPreconditioner MP;
    gpu_mas32::TraditionalMAS32Preconditioner traditional_mas32;

    int P_type = 1;

  public:
    void Malloc_DEVICE_MEM(const int& vertex_num, const int& tetradedra_num);
    void FREE_DEVICE_MEM();
};
#endif  // ! _PCG_SOLVER_CUH_
