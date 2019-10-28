cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(compressor LANGUAGES CXX)
find_library(ADIOS2_LIBRARY NAMES adios2)
find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Kokkos REQUIRED)
find_path(ADIOS2_INCLUDE_DIR NAMES adios2.h)

add_executable(gaussian_filter_kokkos gaussian_filter_kokkos.cpp)
target_link_libraries(gaussian_filter_kokkos MPI::MPI_CXX ${ADIOS2_LIBRARY} Kokkos::kokkos)
target_include_directories(gaussian_filter_kokkos PUBLIC ${ADIOS2_INCLUDE_DIR})
set_property(TARGET gaussian_filter_kokkos PROPERTY CXX_STANDARD 11)

install(TARGETS gaussian_filter_kokkos)
