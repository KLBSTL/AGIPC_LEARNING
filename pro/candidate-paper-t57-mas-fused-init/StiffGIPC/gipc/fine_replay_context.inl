// Diagnostic-only restoration of the exact dynamic CEMAS collision graph.
namespace {
template<class T> std::vector<T> fine_context_read(const std::filesystem::path& path) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    if(!in)throw std::runtime_error("missing fine context: "+path.string());
    const auto bytes=in.tellg();if(bytes<0 || bytes%sizeof(T))throw std::runtime_error("invalid context size");
    std::vector<T> data(static_cast<std::size_t>(bytes)/sizeof(T));in.seekg(0);
    if(bytes>0)in.read(reinterpret_cast<char*>(data.data()),bytes);
    if(!in)throw std::runtime_error("context read failed");return data;
}
}
gipc::Json GIPC::replay_frozen_fine(const std::string& directory,const std::string& output,device_TetraData& mesh)
{
    const std::filesystem::path dir=directory;
    const auto meta=gipc::Json::parse(std::ifstream(dir/"metadata.json"));
    if(meta.value("format",std::string{})!="agipc_accepted_direction_v3")
        throw std::runtime_error("native fine replay requires V3 collision/segment context; V1/V2 cannot recover it");
    const int count=meta.at("mapping_fine_nodes"),offset=abd_fem_count_info.fem_point_offset;
    const int prefix=meta.at("fine_block_nodes").get<int>()-count;
    if(prefix!=4*abd_fem_count_info.abd_body_num
       || offset!=meta.at("criterion_snapshot").at("fine_offset").get<int>())
        throw std::runtime_error("native ABD/FEM layout mismatch");
    // init_headless has not assembled a physical gradient yet. Establish its
    // original layout without recomputing or replacing the captured RHS.
    m_abd_system->system_gradient.resize(3*prefix);
    if(count!=abd_fem_count_info.fem_point_num || pcg_data.P_type!=1)
        throw std::runtime_error("fine replay scene/CEMAS16 mismatch");
    const auto rest=fine_context_read<double3>(dir/"fine_rest_positions.f64x3.bin");
    std::vector<double3> initial(count);
    CUDA_SAFE_CALL(cudaMemcpy(initial.data(),mesh.vertexes+offset,count*sizeof(double3),cudaMemcpyDeviceToHost));
    if(rest.size()!=count)throw std::runtime_error("rest dimension mismatch");
    double mismatch=0;
    for(int i=0;i<count;++i) {
        mismatch=std::max(mismatch,std::abs(initial[i].x-rest[i].x));
        mismatch=std::max(mismatch,std::abs(initial[i].y-rest[i].y));
        mismatch=std::max(mismatch,std::abs(initial[i].z-rest[i].z));
    }
    if(!std::isfinite(mismatch) || mismatch>1e-12)throw std::runtime_error("native fine node ordering/rest positions mismatch");
    const auto positions=fine_context_read<double3>(dir/"criterion_current_positions.f64x3.bin");
    const auto boundary=fine_context_read<int>(dir/"criterion_boundary.i32.bin");
    const auto pairs=fine_context_read<int4>(dir/"native_collision_pairs.i32x4.bin");
    const int cp=meta.at("active_contact_pairs");
    if(positions.size()!=count || boundary.size()!=count || cp<0 || pairs.size()!=cp)
        throw std::runtime_error("native context dimension mismatch");
    ensure_collision_pair_capacity(cp,0);
    if(cp>0)CUDA_SAFE_CALL(cudaMemcpy(_collisonPairs,pairs.data(),cp*sizeof(int4),cudaMemcpyHostToDevice));
    h_cpNum[0]=cp;h_gpNum=meta.at("active_ground_pairs");
    CUDA_SAFE_CALL(cudaMemcpy(mesh.vertexes+offset,positions.data(),count*sizeof(double3),cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(mesh.BoundaryType+offset,boundary.data(),count*sizeof(int),cudaMemcpyHostToDevice));
    auto result=m_global_linear_system->replay_frozen_fine(directory,output);
    result["native_rest_order_max_abs_error"]=mismatch;
    result["native_collision_pairs_restored"]=cp;
    std::ofstream out(output);out<<result.dump(2)<<'\n';out.close();
    if(!out)throw std::runtime_error("context report write failed");
    return result;
}
