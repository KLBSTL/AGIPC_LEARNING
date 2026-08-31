#include <GIPC.cuh>
#include <gipc/gipc.h>
#include <gipc/utils/timer.h>
#include <gipc/utils/json.h>
#include <filesystem>
#include <fstream>

void GIPC::build_gipc_system(device_TetraData& tet)
{
    std::cout << "* Building GIPC system:" << std::endl;
    gipc::Timer::disable_all();
    
    // set up debug
    cudatool::Debug::debug_sync_all(false);

    std::cout << "- create ABD system..." << std::endl;
    m_abd_sim_data            = std::make_unique<gipc::ABDSimData>(*this, tet);
    m_abd_system              = std::make_unique<gipc::ABDSystem>();
    m_abd_system->parms.kappa = 1e8;
    m_abd_system->parms.dt    = IPC_dt;

    std::filesystem::path config_dir =
        std::filesystem::path(GIPC_ASSETS_DIR) / "scene/abd_system_config.json";
    if(!std::filesystem::exists(config_dir))
        config_dir = std::filesystem::current_path() / "Assets/scene/abd_system_config.json";

    gipc::Json json = gipc::Json::parse(std::ifstream(config_dir));
    

    m_abd_system->parms.motor_speed = json["motor_speed"].get<double>();
    m_abd_system->parms.motor_strength = json["motor_strength"].get<double>();

    std::cout << "- create Global Linear System ..." << std::endl;

    m_global_linear_system = std::make_unique<gipc::GlobalLinearSystem>();

    std::cout << "* Finished building GIPC system." << std::endl;
}

void GIPC::set_spmv_mode(gipc::SpmvMode mode)
{
    CT_ASSERT(m_global_linear_system, "build_gipc_system() must precede set_spmv_mode()");
    m_global_linear_system->spmv_mode(mode);
}

void GIPC::set_frozen_linear_diagnostics_path(const std::string& path)
{
    CT_ASSERT(m_global_linear_system,
              "build_gipc_system() must precede frozen diagnostics setup");
    m_global_linear_system->frozen_linear_diagnostics_path(path);
}

bool GIPC::frozen_linear_diagnostics_complete() const
{
    return m_global_linear_system
           && m_global_linear_system->frozen_linear_diagnostics_complete();
}

void GIPC::init_abd_system()
{
    m_abd_sim_data->upload();
    m_abd_system->init_system(*m_abd_sim_data);
}

void GIPC::create_LinearSystem(device_TetraData& tet)
{
    std::cout << "    - create ABD Linear Subsystem ..." << std::endl;
    auto& abd = m_global_linear_system->create<gipc::ABDLinearSubsystem>(
        *this, *m_abd_system, *m_abd_sim_data);
    std::cout << "    - create FEM Linear Subsystem ..." << std::endl;
    auto& fem = m_global_linear_system->create<gipc::FEMLinearSubsystem>(*this, tet);


    std::cout << "- create PCG Solver" << std::endl;
    gipc::PCGSolverConfig cfg;
    cfg.global_tol_rate = pcg_threshold;
    auto& pcg           = m_global_linear_system->create<gipc::PCGSolver>(cfg);

    std::cout << "- create Preconditioner" << std::endl;
    
    m_global_linear_system->create<gipc::ABDPreconditioner>(abd, *m_abd_system, *m_abd_sim_data);

    if(pcg_data.P_type == 1)
    {

        m_global_linear_system->create<gipc::MAS_Preconditioner>(
            fem, pcg_data.MP, tet.masses, h_cpNum);
    }
    else if(pcg_data.P_type == 2)
    {
        m_global_linear_system->create<gipc::TraditionalMAS32_Preconditioner>(
            fem, pcg_data.traditional_mas32, h_cpNum);
    }
    else
    {
        m_global_linear_system->create<gipc::DiagPreconditioner>();
    }
}
