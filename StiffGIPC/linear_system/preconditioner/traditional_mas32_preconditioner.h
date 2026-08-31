#pragma once

#include <linear_system/linear_system/i_preconditioner.h>
#include <gipc/utils/json.h>

namespace gpu_mas32
{
class TraditionalMAS32Preconditioner;
}

namespace gipc
{
class FEMLinearSubsystem;

class TraditionalMAS32_Preconditioner : public LocalPreconditioner
{
    using Base = LocalPreconditioner;
    gpu_mas32::TraditionalMAS32Preconditioner& mas;
    uint32_t*                                   cp_num;

  public:
    TraditionalMAS32_Preconditioner(FEMLinearSubsystem& subsystem,
                                    gpu_mas32::TraditionalMAS32Preconditioner& mas,
                                    uint32_t* cp_num);
    void assemble() override;
    void apply(cudatool::CDenseVectorView<Float> r,
               cudatool::DenseVectorView<Float> z) override;
    Json numerical_diagnostics(cudatool::CDenseVectorView<Float> r) const;
};
}  // namespace gipc
