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
#include <cuda_tools/cuda_all.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <vector>
#include <bitset>
#include <fstream>
#include <iostream>

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
    unsigned int connectMsk  = _fineConnectedMsk[idx];
    //unsigned int connectMsk = cacheMask1;
    __shared__ int unsigned cacheMask[DEFAULT_BLOCKSIZE];
    __shared__ int          prefixSum[DEFAULT_WARPNUM];
    if(laneId == 0)
    {
        prefixSum[localWarpId] = 0;
    }
    cacheMask[threadIdx.x] = connectMsk;
    unsigned int visited   = (1U << laneId);
    while(connectMsk != -1)
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
        unsigned int visited   = (1U << laneId);
        while(connectMsk != -1)
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

    __shared__ unsigned int electedMask[GPU_MAS_BANKSIZE];
    __shared__ unsigned int lanePrefix[GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE];
    if(laneId == 0)
    {
        electedMask[localWarpId] = 0;
    }
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

    //unsigned int lanePrefix2 = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
    //lanePrefix2 += _prefixSumOriginal[warpId];

    //unsigned int elected_lane = __ffs(connMsk) - 1;
    //unsigned int theLanePrefix = __shfl_sync(0xffffffff, lanePrefix2, elected_lane);

    lanePrefix[threadIdx.x] = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
    lanePrefix[threadIdx.x] += _prefixSumOriginal[warpId];

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

    __shared__ unsigned int electedMask[GPU_MAS_BANKSIZE];
    __shared__ unsigned int lanePrefix[GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE];
    if(laneId == 0)
    {
        electedMask[localWarpId] = 0;
    }
    if(tdx == number - 1)
    {
        _levelSize[1].x = _prefixSumOriginal[warpId] + _prefixOriginal[warpId];
        _levelSize[1].y = (number + GPU_MAS_BANKSIZE - 1) / GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE;
    }
    int idx = _partId_map_real[tdx];
    if(idx >= 0)
    {

        unsigned int connMsk = _fineConnectedMsk[idx];

        unsigned int electedPrefix = __popc(connMsk & _LanemaskLt(laneId));

        if(electedPrefix == 0)
        {
            atomicOr(electedMask + localWarpId, (1U << laneId));
        }

        //unsigned int lanePrefix2 = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
        //lanePrefix2 += _prefixSumOriginal[warpId];

        //unsigned int elected_lane = __ffs(connMsk) - 1;
        //unsigned int theLanePrefix = __shfl_sync(0xffffffff, lanePrefix2, elected_lane);

        lanePrefix[threadIdx.x] = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
        lanePrefix[threadIdx.x] += _prefixSumOriginal[warpId];

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

    if(__popc(prefixMsk) == GPU_MAS_BANKSIZE)
    {
        atomicOr(cacheMsk + localWarpId * GPU_MAS_BANKSIZE, connMsk);
        connMsk = cacheMsk[localWarpId * GPU_MAS_BANKSIZE];
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
        connMsk = cacheMsk[localWarpId * GPU_MAS_BANKSIZE + electedLane];
    }

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

        if(__popc(prefixMsk) == GPU_MAS_BANKSIZE)
        {
            atomicOr(cacheMsk + localWarpId * GPU_MAS_BANKSIZE, connMsk);
            connMsk = cacheMsk[localWarpId * GPU_MAS_BANKSIZE];
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
            connMsk = cacheMsk[localWarpId * GPU_MAS_BANKSIZE + electedLane];
        }

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

    __shared__ unsigned int electedMask[GPU_MAS_BANKSIZE];
    __shared__ unsigned int lanePrefix[GPU_MAS_BANKSIZE * GPU_MAS_BANKSIZE];
    if(laneId == 0)
    {
        electedMask[localWarpId] = 0;
    }

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

    lanePrefix[threadIdx.x] = __popc(electedMask[localWarpId] & _LanemaskLt(laneId));
    lanePrefix[threadIdx.x] += _nextPrefixSum[warpId];

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

    if(idx >= 0)
    {

        unsigned int connectMsk = _fineConnectMsk[idx];

        if(prefixSum[localWarpId] == 1)
        {
            auto mask_val  = __activemask();
            int  warpId    = threadIdx.x & 0x1f;
            bool bBoundary = (laneId == 0) || (warpId == 0);

            unsigned int mark     = __ballot_sync(mask_val, bBoundary);
            mark                  = __brev(mark);
            int          clzlen   = __clz(mark << (warpId + 1));
            unsigned int interval = std::min(clzlen, 31 - warpId);


            for(int iter = 1; iter < GPU_MAS_BANKSIZE; iter <<= 1)
            {
                float tmpx = __shfl_down_sync(mask_val, r[0], iter);
                float tmpy = __shfl_down_sync(mask_val, r[1], iter);
                float tmpz = __shfl_down_sync(mask_val, r[2], iter);
                if(interval >= iter)
                {
                    r[0] += tmpx;
                    r[1] += tmpy;
                    r[2] += tmpz;
                }
            }
            //int level = 0;

            if(bBoundary)
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
            atomicAdd(c_sumResidual + localWarpId * GPU_MAS_BANKSIZE + elected_lane, r[0]);
            atomicAdd(c_sumResidual + localWarpId * GPU_MAS_BANKSIZE + elected_lane + DEFAULT_BLOCKSIZE,
                      r[1]);
            atomicAdd(c_sumResidual + localWarpId * GPU_MAS_BANKSIZE + elected_lane + 2 * DEFAULT_BLOCKSIZE,
                      r[2]);

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

    if((rdx >= 0) && (cdx >= 0))
    {
        if(prefix == 1)
        {
            int warpId = threadIdx.x & 0x1f;
            bool bBoundary = (warpId == 0) || (rdx < 0) || (cdx < 0);
            unsigned int mark = __ballot_sync(0xffffffff, bBoundary);
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
                            __shfl_down_sync(0xffffffff, mat3(i, j), iter);
                    }
                }
                if(interval >= iter)
                {
                    mat3 = mat3 + matTemp;
                }
            }
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

    CUDA_SAFE_CALL(cudaMemset(d_multiLevelZ, 0, (totalNumberClusters) * sizeof(Precision_T3)));

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
    CUDA_SAFE_CALL(cudaFree(d_multiLevelR));
    CUDA_SAFE_CALL(cudaFree(d_multiLevelZ));
}
}  // namespace gpu_mas32
