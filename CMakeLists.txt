cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(EntropyVolumeRenderer LANGUAGES CXX CUDA)

file(GLOB SOURCES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/src/entropy/*.cpp
    ${PROJECT_SOURCE_DIR}/src/cuda/*.cu
    )

file(GLOB HEADERS
    ${PROJECT_SOURCE_DIR}/src/*.h
    ${PROJECT_SOURCE_DIR}/src/entropy/*.h
    ${PROJECT_SOURCE_DIR}/src/cuda/*.h
    ${PROJECT_SOURCE_DIR}/src/util/*.h
    )

add_library(MIVolumeRender STATIC
    ${SOURCES}
    ${HEADERS}
    )

target_compile_features(MIVolumeRender PUBLIC cxx_std_11)

set_target_properties(  MIVolumeRender
                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                        )

add_executable(MIVolumeRenderer src/main.cpp)

find_package(Eigen3 3.3 REQUIRED)
find_package(OpenGL REQUIRED COMPONENTS OpenGL)
find_package(GLUT REQUIRED)
find_package(GLEW 2.0 REQUIRED)

add_dependencies(MIVolumeRenderer OpenGL::OpenGL)
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
    src/entropy
    src/cuda
    src/util
    )
target_link_libraries(MIVolumeRenderer PRIVATE MIVolumeRender
    ${GLUT_LIBRARY}
    Eigen3::Eigen
    OpenGL::OpenGL
    GLEW::GLEW
    )
