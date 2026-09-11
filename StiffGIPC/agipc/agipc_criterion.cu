#include "agipc_criterion.cuh"
#include <cuda_tools/cuda_device_buffer.h>
#include <device_fem_data.cuh>
#include <load_mesh.h>
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

namespace agipc
{
namespace
{
constexpr int threads = 256;
struct Element
{
    unsigned v[4] = {};
    int dim = 3;
    double inverse[9] = {}; // row-major, dim x dim, with stride 3
};
struct Strain { double g[9]; };
struct Workspace
{
    bool enabled = false;
    int fine_vertices = 0;
    int fine_offset = 0;
    int coarse_vertices = 0;
    int translational_vertices = 0;
    int affine_vertices = 0;
    int coarse_block_vertices = 0;
    int max_levels = 8;
    double threshold = 5e-5;
    cudatool::CudaDeviceBuffer<Element> elements;
    cudatool::CudaDeviceBuffer<Strain> previous;
    cudatool::CudaDeviceBuffer<double> increments;
    cudatool::CudaDeviceBuffer<uint2> edges;
    cudatool::CudaDeviceBuffer<int> offsets, adjacent, tags, reasons;
    cudatool::CudaDeviceBuffer<unsigned> direct_masks, component_masks;
    cudatool::CudaDeviceBuffer<int> representatives, component_flags;
    cudatool::CudaDeviceBuffer<int> component_prefix, current_to_next;
    cudatool::CudaDeviceBuffer<int> fine_to_coarse, fine_to_next, child_counts;
    cudatool::CudaDeviceBuffer<int> affine_flags, affine_prefix;
    cudatool::CudaDeviceBuffer<int> coarse_sorted_ids, coarse_block_bases;
    cudatool::CudaDeviceBuffer<double3> rest_positions;
    cudatool::CudaDeviceBuffer<int> remaining_edges;
    cudaEvent_t start = nullptr, end = nullptr;
    size_t updates = 0, protected_sum = 0, collapsible_sum = 0;
    double increment_min = std::numeric_limits<double>::infinity();
    double increment_max = 0, increment_sum = 0;
    size_t increment_samples = 0;
    gipc::Json last_mapping = nullptr;
};
Workspace workspace;

// Main paper Eq. (3). The identical kernel serves tet and triangle elements.
__global__ void green_increment(const double3* x, const Element* elements,
                               Strain* previous, double* increments,
                               int count, bool reset)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) return;
    const Element e = elements[i];
    double f[3][3] = {};
    const double3 origin = x[e.v[0]];
    for(int k = 0; k < e.dim; ++k)
    {
        const double3 point = x[e.v[k + 1]];
        const double ds[3] = {point.x-origin.x, point.y-origin.y, point.z-origin.z};
        for(int r = 0; r < 3; ++r)
            for(int c = 0; c < e.dim; ++c)
                f[r][c] += ds[r] * e.inverse[3*k+c];
    }
    Strain current = {};
    double norm2 = 0;
    bool valid = true;
    for(int r = 0; r < e.dim; ++r)
        for(int c = 0; c < e.dim; ++c)
        {
            double g = r == c ? -1.0 : 0.0;
            for(int k = 0; k < 3; ++k) g += f[k][r] * f[k][c];
            g *= 0.5;
            current.g[3*r+c] = g;
            const double delta = reset ? 0.0 : g-previous[i].g[3*r+c];
            valid = valid && isfinite(g) && isfinite(delta);
            norm2 += delta*delta;
        }
    previous[i] = current;
    increments[i] = valid && isfinite(norm2) ? sqrt(norm2) : INFINITY;
}

// Reasons distinguish strain, boundary and invalid input (bits 1,2,4).
__global__ void tag_edges(const uint2* edges, const int* offsets, const int* adjacent,
                          const double* increments, const int* boundary, int* tags,
                          int* reasons, double threshold, int count)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if(e >= count) return;
    int reason = boundary && (boundary[edges[e].x] || boundary[edges[e].y]) ? 2 : 0;
    for(int j = offsets[e]; j < offsets[e+1]; ++j)
    {
        const double v = increments[adjacent[j]];
        if(!isfinite(v)) reason |= 4;
        else if(v > threshold) reason |= 1;
    }
    reasons[e] = reason;
    tags[e] = reason == 0 ? 1 : 0;
}

__global__ void initialize_mapping(int* map, int count)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) map[i]=i;
}

__global__ void initialize_masks(unsigned* masks, int count)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) masks[i]=1u<<(i&31);
}

__global__ void add_direct_neighbors(const uint2* edges, const int* tags,
                                     const int* fine_map, unsigned* masks,
                                     int fine_offset, int edge_count)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=edge_count || tags[i]==0) return;
    const int left=fine_map[static_cast<int>(edges[i].x)-fine_offset];
    const int right=fine_map[static_cast<int>(edges[i].y)-fine_offset];
    if(left==right || (left>>5)!=(right>>5)) return;
    atomicOr(masks+left,1u<<(right&31));
    atomicOr(masks+right,1u<<(left&31));
}

__global__ void close_components(const unsigned* direct, unsigned* components,
                                 int* representatives, int* flags, int count)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    const int base=i&~31;
    unsigned connected=direct[i], visited=0;
    while(true)
    {
        unsigned pending=connected&~visited;
        if(!pending) break;
        const int lane=__ffs(pending)-1;
        visited|=1u<<lane;
        connected|=direct[base+lane];
    }
    const int representative=base+__ffs(connected)-1;
    components[i]=connected;
    representatives[i]=representative;
    flags[i]=representative==i?1:0;
}

__global__ void write_next_map(const int* representatives, const int* prefix,
                               int* current_to_next, int count)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) current_to_next[i]=prefix[representatives[i]];
}

__global__ void compose_map(const int* fine_map, const int* current_to_next,
                            int* next_map, int count)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) next_map[i]=current_to_next[fine_map[i]];
}

__global__ void count_children(const int* map, int* counts, int fine_count)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<fine_count) atomicAdd(counts+map[i],1);
}

__global__ void classify_coarse_nodes(const int* child_counts,int* affine_flags,int count)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count) affine_flags[i]=child_counts[i]>32?1:0;
}

__global__ void write_coarse_layout(const int* affine_flags,const int* affine_prefix,
                                    int* sorted_ids,int* block_bases,int n3,int count)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count) return;
    const int sorted=affine_flags[i] ? n3+affine_prefix[i] : i-affine_prefix[i];
    sorted_ids[i]=sorted;
    block_bases[i]=sorted<n3 ? sorted : n3+4*(sorted-n3);
}

__global__ void count_uncollapsed_allowed_edges(const uint2* edges,const int* tags,
                                                 const int* map,int fine_offset,
                                                 int* remaining,int edge_count)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<edge_count && tags[i] && map[static_cast<int>(edges[i].x)-fine_offset]
                                 !=map[static_cast<int>(edges[i].y)-fine_offset])
        atomicAdd(remaining,1);
}

void launch_increment(Workspace& w, const double3* positions, bool reset)
{
    if(w.elements.size())
        green_increment<<<(w.elements.size()+threads-1)/threads, threads>>>(
            positions, w.elements.data(), w.previous.data(), w.increments.data(),
            static_cast<int>(w.elements.size()), reset);
    CUDA_SAFE_CALL(cudaGetLastError());
}
void launch_tags(Workspace& w, const int* boundary)
{
    if(w.edges.size())
        tag_edges<<<(w.edges.size()+threads-1)/threads, threads>>>(
            w.edges.data(), w.offsets.data(), w.adjacent.data(), w.increments.data(),
            boundary, w.tags.data(), w.reasons.data(), w.threshold,
            static_cast<int>(w.edges.size()));
    CUDA_SAFE_CALL(cudaGetLastError());
}

void reserve_mapping(Workspace& w)
{
    const size_t n=w.fine_vertices;
    w.direct_masks.resize(n); w.component_masks.resize(n);
    w.representatives.resize(n); w.component_flags.resize(n);
    w.component_prefix.resize(n); w.current_to_next.resize(n);
    w.fine_to_coarse.resize(n); w.fine_to_next.resize(n); w.child_counts.resize(n);
    w.affine_flags.resize(n); w.affine_prefix.resize(n);
    w.coarse_sorted_ids.resize(n); w.coarse_block_bases.resize(n);
    w.remaining_edges.resize(1);
}

gipc::Json build_mapping(Workspace& w)
{
    if(!w.fine_vertices)
    {
        w.coarse_vertices=0;
        return {{"fine_nodes",0},{"coarse_nodes",0},{"complete",true}};
    }
    const int fine_blocks=(w.fine_vertices+threads-1)/threads;
    const int edge_blocks=(static_cast<int>(w.edges.size())+threads-1)/threads;
    initialize_mapping<<<fine_blocks,threads>>>(w.fine_to_coarse.data(),w.fine_vertices);
    int current=w.fine_vertices;
    gipc::Json levels=gipc::Json::array();
    bool fixed_point=false;
    for(int level=0;level<w.max_levels && current>1;++level)
    {
        const int blocks=(current+threads-1)/threads;
        initialize_masks<<<blocks,threads>>>(w.direct_masks.data(),current);
        if(edge_blocks>0)
        {
            add_direct_neighbors<<<edge_blocks,threads>>>(w.edges.data(),w.tags.data(),
                w.fine_to_coarse.data(),w.direct_masks.data(),w.fine_offset,
                static_cast<int>(w.edges.size()));
        }
        close_components<<<blocks,threads>>>(w.direct_masks.data(),w.component_masks.data(),
            w.representatives.data(),w.component_flags.data(),current);
        thrust::exclusive_scan(thrust::device_ptr<int>(w.component_flags.data()),
            thrust::device_ptr<int>(w.component_flags.data())+current,
            thrust::device_ptr<int>(w.component_prefix.data()));
        const int next=thrust::reduce(thrust::device_ptr<int>(w.component_flags.data()),
            thrust::device_ptr<int>(w.component_flags.data())+current,0);
        write_next_map<<<blocks,threads>>>(w.representatives.data(),w.component_prefix.data(),
                                          w.current_to_next.data(),current);
        compose_map<<<fine_blocks,threads>>>(w.fine_to_coarse.data(),w.current_to_next.data(),
                                             w.fine_to_next.data(),w.fine_vertices);
        CUDA_SAFE_CALL(cudaMemcpy(w.fine_to_coarse.data(),w.fine_to_next.data(),
                                  w.fine_vertices*sizeof(int),cudaMemcpyDeviceToDevice));
        levels.push_back({{"level",level+1},{"input_nodes",current},{"output_nodes",next}});
        if(next==current) { fixed_point=true; break; }
        current=next;
    }
    w.child_counts.reset_zero();
    count_children<<<fine_blocks,threads>>>(w.fine_to_coarse.data(),w.child_counts.data(),
                                            w.fine_vertices);
    w.remaining_edges.reset_zero();
    if(edge_blocks>0)
    {
        count_uncollapsed_allowed_edges<<<edge_blocks,threads>>>(w.edges.data(),w.tags.data(),
            w.fine_to_coarse.data(),w.fine_offset,w.remaining_edges.data(),
            static_cast<int>(w.edges.size()));
    }
    CUDA_SAFE_CALL(cudaGetLastError());
    std::vector<int> children,remaining;
    w.child_counts.copy_to_host(children); w.remaining_edges.copy_to_host(remaining);
    int child_sum=0,min_children=std::numeric_limits<int>::max(),max_children=0;
    for(int i=0;i<current;++i)
    {
        child_sum+=children[i]; min_children=std::min(min_children,children[i]);
        max_children=std::max(max_children,children[i]);
    }
    const bool valid=child_sum==w.fine_vertices && min_children>0;
    if(!valid) throw std::runtime_error("AGIPC mapping ownership invariant failed");
    const bool complete=remaining[0]==0;
    w.coarse_vertices=current;
    const int coarse_blocks=(current+threads-1)/threads;
    classify_coarse_nodes<<<coarse_blocks,threads>>>(w.child_counts.data(),
                                                     w.affine_flags.data(),current);
    thrust::exclusive_scan(thrust::device_ptr<int>(w.affine_flags.data()),
        thrust::device_ptr<int>(w.affine_flags.data())+current,
        thrust::device_ptr<int>(w.affine_prefix.data()));
    const int n12=thrust::reduce(thrust::device_ptr<int>(w.affine_flags.data()),
        thrust::device_ptr<int>(w.affine_flags.data())+current,0);
    const int n3=current-n12;
    write_coarse_layout<<<coarse_blocks,threads>>>(w.affine_flags.data(),
        w.affine_prefix.data(),w.coarse_sorted_ids.data(),w.coarse_block_bases.data(),n3,current);
    CUDA_SAFE_CALL(cudaGetLastError());
    w.translational_vertices=n3;
    w.affine_vertices=n12;
    w.coarse_block_vertices=n3+4*n12;
    return {{"method","paper_warp_hash"},{"fine_nodes",w.fine_vertices},
            {"coarse_nodes",current},{"mapping_levels",levels.size()},{"levels",levels},
            {"translational_nodes",n3},{"affine_nodes",n12},
            {"coarse_block_nodes",w.coarse_block_vertices},
            {"min_children",min_children},{"max_children",max_children},
            {"child_sum",child_sum},{"remaining_collapsible_edges",remaining[0]},
            {"fixed_point",fixed_point},{"complete",complete},
            {"stop_reason",complete?"all_collapsible_components_resolved":
                (fixed_point?"fixed_point_with_remaining_edges":"level_cap")}};
}
}

void initialize_criterion(const tetrahedra_obj& mesh, double threshold, int max_levels)
{
    if(!std::isfinite(threshold) || threshold <= 0)
        throw std::invalid_argument("AGIPC threshold must be finite and positive");
    auto& w = workspace;
    w.threshold = threshold;
    if(max_levels<1 || max_levels>16) throw std::invalid_argument("AGIPC max levels must be 1..16");
    w.max_levels=max_levels;
    const auto& counts = mesh.abd_fem_count_info;
    w.fine_vertices = static_cast<int>(counts.fem_point_num);
    w.fine_offset = static_cast<int>(counts.fem_point_offset);
    std::vector<Element> elements;
    for(size_t t = 0; t < counts.fem_tet_num; ++t)
    {
        const size_t id = counts.fem_tet_offset+t;
        const auto tet = mesh.tetrahedras.at(id);
        Element e{{tet.x,tet.y,tet.z,tet.w},3,{}};
        for(int r=0;r<3;++r) for(int c=0;c<3;++c)
            e.inverse[3*r+c] = mesh.DM_inverse.at(id).m[r][c];
        elements.push_back(e);
    }
    for(size_t t = 0; t < counts.fem_tri_num; ++t)
    {
        const auto tri = mesh.triangles.at(t);
        Element e{{tri.x,tri.y,tri.z,0},2,{}};
        for(int r=0;r<2;++r) for(int c=0;c<2;++c)
            e.inverse[3*r+c] = mesh.tri_DM_inverse.at(t).m[r][c];
        elements.push_back(e);
    }
    // Topology is immutable. Deterministic edge order also simplifies dumps.
    std::map<std::pair<unsigned,unsigned>,std::vector<int>> incident;
    for(size_t i=0;i<elements.size();++i)
        for(int a=0;a<=elements[i].dim;++a)
            for(int b=a+1;b<=elements[i].dim;++b)
            {
                const auto lo = std::min(elements[i].v[a],elements[i].v[b]);
                const auto hi = std::max(elements[i].v[a],elements[i].v[b]);
                if(lo < counts.fem_point_offset || hi >= counts.fem_point_offset+counts.fem_point_num)
                    throw std::runtime_error("AGIPC element escapes FEM vertex domain");
                incident[{lo,hi}].push_back(static_cast<int>(i));
            }
    std::vector<uint2> edges;
    std::vector<int> offsets{0}, adjacent;
    for(const auto& item: incident)
    {
        edges.push_back(make_uint2(item.first.first,item.first.second));
        adjacent.insert(adjacent.end(),item.second.begin(),item.second.end());
        offsets.push_back(static_cast<int>(adjacent.size()));
    }
    w.elements.copy_from_host(elements);
    w.previous.resize(elements.size()); w.increments.resize(elements.size());
    std::vector<double3> rest;
    rest.reserve(counts.fem_point_num);
    for(size_t i=0;i<counts.fem_point_num;++i)
        rest.push_back(mesh.vertexes.at(counts.fem_point_offset+i));
    w.rest_positions.copy_from_host(rest);
    w.edges.copy_from_host(edges); w.offsets.copy_from_host(offsets);
    w.adjacent.copy_from_host(adjacent);
    w.tags.resize(edges.size()); w.reasons.resize(edges.size());
    reserve_mapping(w);
    if(!w.start) { CUDA_SAFE_CALL(cudaEventCreate(&w.start)); CUDA_SAFE_CALL(cudaEventCreate(&w.end)); }
    w.enabled = true;
    w.updates = w.protected_sum = w.collapsible_sum = w.increment_samples = 0;
    w.increment_min = std::numeric_limits<double>::infinity();
    w.increment_max = w.increment_sum = 0;
    w.last_mapping = nullptr;
}

gipc::Json update_mapping()
{
    if(!workspace.enabled) return nullptr;
    workspace.last_mapping=build_mapping(workspace);
    return workspace.last_mapping;
}

MappingDeviceView mapping_device_view()
{
    const auto& w=workspace;
    return {w.fine_to_coarse.data(),w.coarse_block_bases.data(),w.affine_flags.data(),
            w.rest_positions.data(),w.fine_vertices,w.coarse_vertices,
            w.translational_vertices,w.affine_vertices,w.coarse_block_vertices,
            w.enabled && !w.last_mapping.is_null()};
}

void begin_criterion_step(device_TetraData& mesh)
{
    if(workspace.enabled) launch_increment(workspace,mesh.vertexes,true);
}

gipc::Json update_criterion(device_TetraData& mesh)
{
    auto& w=workspace;
    if(!w.enabled) return nullptr;
    CUDA_SAFE_CALL(cudaEventRecord(w.start));
    launch_increment(w,mesh.vertexes,false);
    launch_tags(w,mesh.BoundaryType);
    CUDA_SAFE_CALL(cudaEventRecord(w.end));
    // This is opt-in diagnostic readback, excluded from formal timing runs.
    std::vector<double> increments;
    std::vector<int> reasons;
    w.increments.copy_to_host(increments); w.reasons.copy_to_host(reasons);
    CUDA_SAFE_CALL(cudaEventSynchronize(w.end));
    float elapsed=0; CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed,w.start,w.end));
    int protected_count=0, strain_count=0, boundary_count=0, invalid_count=0;
    for(int r:reasons) { protected_count+=r!=0; strain_count+=(r&1)!=0; boundary_count+=(r&2)!=0; }
    double low=std::numeric_limits<double>::infinity(), high=0, sum=0;
    for(double v:increments)
    {
        if(!std::isfinite(v)) { ++invalid_count; continue; }
        low=std::min(low,v); high=std::max(high,v); sum+=v;
    }
    if(invalid_count) throw std::runtime_error("AGIPC criterion nonfinite element; Gate A failed");
    ++w.updates;
    w.protected_sum += protected_count;
    w.collapsible_sum += reasons.size()-protected_count;
    if(!increments.empty())
    {
        w.increment_min=std::min(w.increment_min,low);
        w.increment_max=std::max(w.increment_max,high);
        w.increment_sum+=sum;
        w.increment_samples+=increments.size();
    }
    return {{"stage","criterion_only"},{"fine_vertex_count",w.fine_vertices},
            {"edge_count",reasons.size()},{"protected_edge_count",protected_count},
            {"collapsible_edge_count",reasons.size()-protected_count},
            {"protected_edge_ratio",reasons.empty()?0.0:double(protected_count)/reasons.size()},
            {"strain_protected_edges",strain_count},{"boundary_protected_edges",boundary_count},
            {"green_increment_min",increments.empty()?0.0:low},
            {"green_increment_mean",increments.empty()?0.0:sum/increments.size()},
            {"green_increment_max",high},{"threshold",w.threshold},
            {"criterion_ms",elapsed},{"invalid_elements",invalid_count},
            {"history_convention","reset_at_subIP_initial_positions"}};
}

gipc::Json criterion_summary()
{
    const auto& w=workspace;
    if(!w.enabled) return nullptr;
    auto result=gipc::Json{{"stage","criterion_and_mapping_shadow"},{"newton_updates",w.updates},
            {"edge_samples",w.protected_sum+w.collapsible_sum},
            {"protected_edge_samples",w.protected_sum},
            {"collapsible_edge_samples",w.collapsible_sum},
            {"protected_edge_ratio",w.protected_sum+w.collapsible_sum
                ? double(w.protected_sum)/(w.protected_sum+w.collapsible_sum):0.0},
            {"green_increment_min",w.increment_samples?w.increment_min:0.0},
            {"green_increment_mean",w.increment_samples?w.increment_sum/w.increment_samples:0.0},
            {"green_increment_max",w.increment_max},{"threshold",w.threshold},
            {"history_convention","reset_at_subIP_initial_positions"}};
    if(!w.last_mapping.is_null()) result["latest_mapping"]=w.last_mapping;
    const auto galerkin=galerkin_summary();
    if(!galerkin.is_null()) result["galerkin_shadow"]=galerkin;
    return result;
}

gipc::Json criterion_self_test()
{
    // Same CUDA kernels as production; independent Eigen tensor reference.
    Workspace w;
    std::vector<Element> elements;
    Element tet{{0,1,2,3},3,{1,0,0,0,1,0,0,0,1}};
    Element tri{{0,1,2,0},2,{1,0,0,0,1,0,0,0,0}};
    elements={tet,tri}; w.elements.copy_from_host(elements);
    w.previous.resize(2); w.increments.resize(2);
    w.edges.copy_from_host(std::vector<uint2>{make_uint2(0,1),make_uint2(2,3),make_uint2(1,2)});
    w.offsets.copy_from_host(std::vector<int>{0,2,3,4});
    w.adjacent.copy_from_host(std::vector<int>{0,1,0,1});
    w.tags.resize(3); w.reasons.resize(3);
    cudatool::CudaDeviceBuffer<double3> x;
    double max_error=0; int cases=0;
    auto require=[](bool ok,const char* message) { if(!ok) throw std::runtime_error(message); };
    auto upload=[&](const Eigen::Matrix3d& f) {
        x.copy_from_host(std::vector<double3>{make_double3(0,0,0),
            make_double3(f(0,0),f(1,0),f(2,0)),make_double3(f(0,1),f(1,1),f(2,1)),
            make_double3(f(0,2),f(1,2),f(2,2))});
    };
    const Eigen::Matrix3d identity=Eigen::Matrix3d::Identity();
    for(int test=0;test<5;++test)
    {
        if(test==4)
        {
            for(auto& e:elements) { e.inverse[0]=2; e.inverse[1]=0.25; e.inverse[4]=0.5; }
            w.elements.copy_from_host(elements);
        }
        upload(identity); launch_increment(w,x.data(),true);
        Eigen::Matrix3d f=identity;
        if(test==1) f(0,0)=1.001;
        if(test==2) f(0,1)=0.003;
        if(test==3) { f(0,0)=0;f(0,1)=-1;f(1,0)=1;f(1,1)=0; }
        if(test==4) { f(0,1)=0.03; f(2,0)=0.02; }
        upload(f); launch_increment(w,x.data(),false);
        std::vector<double> actual; w.increments.copy_to_host(actual);
        for(int i=0;i<2;++i)
        {
            const int dim=elements[i].dim;
            Eigen::MatrixXd inv(dim,dim);
            for(int r=0;r<dim;++r) for(int c=0;c<dim;++c) inv(r,c)=elements[i].inverse[3*r+c];
            const Eigen::MatrixXd now=f.leftCols(dim)*inv;
            const Eigen::MatrixXd before=identity.leftCols(dim)*inv;
            const double expected=(0.5*(now.transpose()*now-before.transpose()*before)).norm();
            max_error=std::max(max_error,std::abs(expected-actual[i]));
        }
        require(max_error<=1e-13,"Gate A tensor increment differs from Eigen reference");
        launch_increment(w,x.data(),false); w.increments.copy_to_host(actual);
        require(actual[0]==0 && actual[1]==0,"Gate A did not advance Newton history");
        ++cases;
    }
    // Shared-edge max, local protection, strict equality and threshold monotonicity.
    w.increments.copy_from_host(std::vector<double>{0,5e-5});
    int previous_protected=4;
    for(double threshold: {1e-6,5e-5,5e-4})
    {
        w.threshold=threshold; launch_tags(w,nullptr);
        std::vector<int> tags; w.tags.copy_to_host(tags);
        const int count=3-tags[0]-tags[1]-tags[2];
        require(count<=previous_protected,"Gate A threshold monotonicity"); previous_protected=count;
        require(tags== (threshold<5e-5 ? std::vector<int>{0,1,0}:std::vector<int>{1,1,1}),
                "Gate A incident element/equality/locality failure");
        launch_tags(w,nullptr); std::vector<int> repeated;w.tags.copy_to_host(repeated);
        require(tags==repeated,"Gate A nondeterministic tags"); ++cases;
    }
    w.increments.copy_from_host(std::vector<double>{std::numeric_limits<double>::quiet_NaN(),0});
    launch_tags(w,nullptr);std::vector<int> tags;w.tags.copy_to_host(tags);
    require(tags==std::vector<int>({0,0,1}),"Gate A NaN must protect affected edges");
    cudatool::CudaDeviceBuffer<int> boundary(std::vector<int>{1,0,0,0});
    w.increments.copy_from_host(std::vector<double>{0,0}); launch_tags(w,boundary.data());
    w.tags.copy_to_host(tags);
    require(tags==std::vector<int>({0,1,1}),"Gate A boundary protection failure");
    upload(identity); launch_increment(w,x.data(),true); launch_increment(w,x.data(),false);
    std::vector<double> reset;w.increments.copy_to_host(reset);
    require(reset[0]==0 && reset[1]==0,"Gate A timestep reset failure");
    return {{"test","agipc_criterion"},{"passed",true},{"cases",cases+2},
            {"max_absolute_tensor_error",max_error},{"gpu_kernels",true},
            {"scope","tensor/history/edge-tags only; not mapping or solver"}};
}

gipc::Json mapping_self_test()
{
    int cases=0;
    auto run=[&](int nodes,const std::vector<uint2>& edges,const std::vector<int>& tags,
                 const std::vector<int>& expected,int expected_affine) {
        Workspace w; w.fine_vertices=nodes; w.fine_offset=0; w.max_levels=16;
        w.edges.copy_from_host(edges); w.tags.copy_from_host(tags); reserve_mapping(w);
        const auto stats=build_mapping(w);
        std::vector<int> actual; w.fine_to_coarse.copy_to_host(actual);
        if(actual.size()!=expected.size()) throw std::runtime_error("Gate B map size mismatch");
        for(int i=0;i<nodes;++i) for(int j=0;j<nodes;++j)
            if((actual[i]==actual[j])!=(expected[i]==expected[j]))
                throw std::runtime_error("Gate B partition differs from CPU oracle");
        if(!stats["complete"].get<bool>())
            throw std::runtime_error("Gate B left a collapsible edge unresolved");
        if(stats["affine_nodes"].get<int>()!=expected_affine)
            throw std::runtime_error("Gate D child-count 32/33 classification failure");
        const auto first=actual;
        build_mapping(w); w.fine_to_coarse.copy_to_host(actual);
        if(first!=actual) throw std::runtime_error("Gate B mapping is nondeterministic");
        ++cases;
    };
    run(7,{}, {},{0,1,2,3,4,5,6},0);
    run(5,{make_uint2(0,2),make_uint2(2,4),make_uint2(1,3)},
          {1,1,1},{0,1,0,1,0},0);
    run(4,{make_uint2(0,1),make_uint2(1,2),make_uint2(2,3)},
          {1,0,1},{0,0,1,1},0);
    std::vector<uint2> chain;
    std::vector<int> tags(31,1), expected(32,0);
    for(int i=0;i<31;++i) chain.push_back(make_uint2(i,i+1));
    run(32,chain,tags,expected,0);
    chain.clear(); tags.assign(32,1); expected.assign(33,0);
    for(int i=0;i<32;++i) chain.push_back(make_uint2(i,i+1));
    run(33,chain,tags,expected,1);
    return {{"test","agipc_mapping"},{"passed",true},{"cases",cases},
            {"gpu_kernels",true},{"covered","indirect/protected/7-tail/32-vs-33-affine/hierarchy/determinism"}};
}
}
