cmake_minimum_required(VERSION 3.18)
#add_library(External INTERFACE)
#add_library(External::et ALIAS External)

add_subdirectory(GKlib)
add_subdirectory(METIS)

#add_dependencies(External GKlib)
#add_dependencies(External METIS)

#set_target_properties(GKlib PROPERTIES FOLDER "External")
#set_target_properties(METIS PROPERTIES FOLDER "External")



