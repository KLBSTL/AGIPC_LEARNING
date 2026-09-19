#pragma once
#include <gipc/utils/json.h>
#include <memory>
class GIPCTripletMatrix;
namespace agipc {
class TinyDenseSolver {
    struct Impl;
    std::unique_ptr<Impl> impl;
public:
    TinyDenseSolver();
    ~TinyDenseSolver();
    gipc::Json solve(const GIPCTripletMatrix& matrix,const double* rhs,double* solution);
};
gipc::Json tiny_dense_self_test();
}
