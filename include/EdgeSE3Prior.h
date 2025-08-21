// EdgeSE3Prior.h
#ifndef EDGE_SE3_PRIOR_H
#define EDGE_SE3_PRIOR_H

#include <Thirdparty/g2o/g2o/core/base_unary_edge.h>
#include <Thirdparty/g2o/g2o/types/types_six_dof_expmap.h>   // SE3Quat, VertexSE3Expmap
#include <Thirdparty/g2o/g2o/core/robust_kernel_impl.h>           // RobustKernelHuber

namespace g2o {

class EdgeSE3Prior : public BaseUnaryEdge<6, SE3Quat, VertexSE3Expmap> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSE3Prior() {}

    // 计算误差：log( measurement^{-1} * estimate )
    void computeError() override {
        const VertexSE3Expmap* v = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        SE3Quat est = v->estimate();
        SE3Quat d = _measurement.inverse() * est;
        _error = d.log();
    }

    // 不需要文件 I/O，直接 stub
    bool read(std::istream& )  override { return false; }
    bool write(std::ostream& ) const override { return false; }
};

} // namespace g2o

#endif // EDGE_SE3_PRIOR_H
