cmake_minimum_required (VERSION 3.5)

# project name
project (my_DE)

# Sources
set (SRCS src/differential_evolution.c)

# Include directories
find_package(MPI REQUIRED)
# set (PATH_TO_MPI_INC ${HOME}/.openmpi/include)
include_directories(include ${MPI_INCLUDE_PATH})

# Compiler options
add_compile_options(-Wall -O3)

# add the library
set (LIB_NAME ${PROJECT_NAME})
set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../lib)
add_library (${LIB_NAME} ${SRCS})

# math.h library (must be after the target!)
# set (PATH_TO_MPI_LIB ${HOME}/.openmpi)
target_link_libraries(${LIB_NAME} ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
  set_target_properties(${LIB_NAME} PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(${LIB_NAME} PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

