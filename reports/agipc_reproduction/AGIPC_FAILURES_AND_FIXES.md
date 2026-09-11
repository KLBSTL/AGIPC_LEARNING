# AGIPC Failures and Fixes

## Empty edge set launched a zero-block kernel

The 7-isolated-node mapping fixture failed inside Thrust with `cudaErrorInvalidDevice`. The pending error came from launching the edge kernels with a zero-sized grid. Both edge launches now have explicit empty-input guards, followed by a CUDA launch check. The isolated-tail fixture passes.

## Rank-deficient affine test fixture

The first mixed test assigned only two fine nodes to a 12-DoF affine aggregate. Galerkin matrix and RHS errors were at machine precision, but Cholesky correctly reported a singular coarse system. The positive fixture now uses four noncoplanar child points. Separate planar and collinear fixtures retain rank 3 and rank 2 evidence so singular geometry is diagnosed rather than hidden.

## Symmetric-half expansion of affine diagonal blocks

Naively emitting and canonicalizing all 16 sub-blocks of an affine fine diagonal doubles off-diagonal coarse blocks. Fine diagonal blocks now emit only the upper triangular affine sub-blocks. Off-diagonal fine blocks still emit the full product, and collapsed coarse diagonals explicitly add the transposed half. Dense `P^T A P` comparison is below `1e-16` relative error.

## New CUDA source was absent from the first parallel link

After CMake detected `agipc_galerkin.cu`, the first parallel build reached the link before its object was available. A sequential rebuild compiled the new unit and linked successfully. Subsequent incremental builds are stable.
