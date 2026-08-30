#include <linear_system/preconditioner/traditional_mas32_preconditioner.h>

#include <TraditionalMAS32Preconditioner.cuh>
#include <gipc/utils/timer.h>
#include <linear_system/subsystem/fem_linear_subsystem.h>

namespace gipc
{
TraditionalMAS32_Preconditioner::TraditionalMAS32_Preconditioner(
    FEMLinearSubsystem& subsystem,
    gpu_mas32::TraditionalMAS32Preconditioner& traditional_mas,
    uint32_t* collision_pair_num)
    : Base(subsystem)
    , mas(traditional_mas)
    , cp_num(collision_pair_num)
{
    preconditioner_id = 2;
}

void TraditionalMAS32_Preconditioner::assemble()
{
    gipc::Timer timer{"precomputing traditional GPU MAS32"};
    int         triplet_number = 0;
    uint32_t*   indices        = calculate_subsystem_bcoo_indices(triplet_number);
    mas.setPreconditioner_bcoo(system_bcoo_matrix(),
                               system_bcoo_rows(),
                               system_bcoo_cols(),
                               indices,
                               get_offset(),
                               triplet_number,
                               *cp_num);
}

void TraditionalMAS32_Preconditioner::apply(cudatool::CDenseVectorView<Float> r,
                                             cudatool::DenseVectorView<Float> z)
{
    mas.preconditioning((double3*)r.data(), (double3*)z.data());
}
}  // namespace gipc
