cmake_minimum_required(VERSION 3.18)
project(MeshProcess)

#add_library(MeshProcess INTERFACE)
#add_library(MeshProcess::mp ALIAS MeshProcess)

add_subdirectory(External)
add_subdirectory(metis_partition)

#add_dependencies(MeshProcess metis_partition)
#add_dependencies(MeshProcess External)

#set_target_properties(External PROPERTIES FOLDER "MeshProcess")
#set_target_properties(metis_partition PROPERTIES FOLDER "MeshProcess")