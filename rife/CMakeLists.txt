cmake_minimum_required(VERSION 3.24)
project(interpolini_rife LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON CACHE BOOL "")
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreaded)

include(FetchContent)

set(NCNN_VULKAN ON CACHE BOOL "")
set(NCNN_SHARED_LIB OFF CACHE BOOL "")
set(NCNN_OPENMP OFF CACHE BOOL "")
set(NCNN_INSTALL_SDK OFF CACHE BOOL "")
set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "")
set(NCNN_BUILD_TESTS OFF CACHE BOOL "")
set(NCNN_BUILD_TOOLS OFF CACHE BOOL "")
set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "")
set(NCNN_DISABLE_RTTI ON CACHE BOOL "")
set(NCNN_DISABLE_EXCEPTION OFF CACHE BOOL "")
set(NCNN_INT8 OFF CACHE BOOL "")
set(NCNN_PIXEL_ROTATE OFF CACHE BOOL "")
set(NCNN_PIXEL_AFFINE OFF CACHE BOOL "")
set(NCNN_PIXEL_DRAWING OFF CACHE BOOL "")

FetchContent_Declare(ncnn
    GIT_REPOSITORY https://github.com/Tencent/ncnn.git
    GIT_TAG e54f7b1f88434e1d844ea0551b880a1cfb079ce1
    GIT_SUBMODULES glslang
    PATCH_COMMAND ${CMAKE_COMMAND} -Dfile=src/platform.h.in -P ${CMAKE_CURRENT_SOURCE_DIR}/log.cmake)
FetchContent_MakeAvailable(ncnn)

# the models come from this fork of rife-ncnn-vulkan
set(fork https://raw.githubusercontent.com/TNTwise/rife-ncnn-vulkan/13338e38debe2e400b3eeecf6792312d01a692f9)
set(source ${CMAKE_CURRENT_BINARY_DIR}/source)
if(NOT EXISTS ${source}/LICENSE)
    file(DOWNLOAD ${fork}/LICENSE ${source}/LICENSE)
endif()
foreach(file flownet.param flownet.bin)
    if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/models/rife-v4.6/${file})
        file(DOWNLOAD ${fork}/models/rife-v4.6/${file} ${CMAKE_CURRENT_BINARY_DIR}/models/rife-v4.6/${file})
    endif()
endforeach()

foreach(name yuv_in yuv_out map interp warp shuffle)
    set(header ${CMAKE_CURRENT_BINARY_DIR}/${name}.comp.hex.h)
    add_custom_command(
        OUTPUT ${header}
        COMMAND ${CMAKE_COMMAND} -Dinput=${CMAKE_CURRENT_SOURCE_DIR}/${name}.comp -Doutput=${header} -P ${CMAKE_CURRENT_SOURCE_DIR}/shader.cmake
        DEPENDS ${name}.comp shader.cmake
        VERBATIM)
    list(APPEND headers ${header})
endforeach()

add_library(interpolini_rife SHARED rife.cpp graph.cpp ${headers})
target_include_directories(interpolini_rife PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ncnn_SOURCE_DIR}/src/layer)
target_link_libraries(interpolini_rife PRIVATE ncnn)

# rife on tensorrt: only the headers are needed to build it, the app downloads the libraries
if(NOT APPLE)
    FetchContent_Declare(tensorrt
        URL https://github.com/NVIDIA/TensorRT/archive/refs/tags/v10.9.0.tar.gz
        URL_HASH SHA256=bbeefc189a3387ac55fdce56545d9462f019c98406a0151c58f915349b3af249
        SOURCE_SUBDIR none)
    FetchContent_MakeAvailable(tensorrt)
    file(DOWNLOAD
        https://raw.githubusercontent.com/onnx/onnx-tensorrt/d5dce67db7c2e64b07e055571f5ec06f7f254de2/NvOnnxParser.h
        ${CMAKE_CURRENT_BINARY_DIR}/NvOnnxParser.h
        EXPECTED_HASH SHA256=2adb4493575008781e5e51a6ca9c8104b337e6f1ee196a94b01df27043a8fd9e)
    add_library(interpolini_rife_trt SHARED trt.cpp)
    target_include_directories(interpolini_rife_trt PRIVATE cuda ${tensorrt_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(interpolini_rife_trt PRIVATE ${CMAKE_DL_LIBS})
endif()
