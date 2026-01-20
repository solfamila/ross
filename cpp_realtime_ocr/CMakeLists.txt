# CMakeLists.txt for Trading Screen Monitor
# Requires: CUDA 12.4+, TensorRT 10.x, Ninja + MSVC toolchain

cmake_minimum_required(VERSION 3.28)

# =============================================================================
# CUDA Configuration (13.1 with VS 2026) - MUST be before project()
# =============================================================================
# Force CUDA 13.1 path
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1")
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe")

# Allow unsupported compiler (VS 2026 not officially supported by CUDA 13.1 yet)
# This MUST be set before project() for compiler detection to work
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")

# Project configuration
project(trading_monitor
    VERSION 1.0.0
    LANGUAGES CXX CUDA
    DESCRIPTION "High-performance trading screen capture with OCR"
)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Windows SDK version (Windows 11)
set(CMAKE_SYSTEM_VERSION 10.0.22621.0)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Ensure nvcc can always locate the MSVC host compiler (cl.exe), even when PATH
# isn't fully configured (common with Ninja builds).
if(MSVC AND CMAKE_CXX_COMPILER)
    if(NOT CMAKE_CUDA_FLAGS MATCHES "-ccbin")
        get_filename_component(_msvc_ccbin_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
        file(TO_CMAKE_PATH "${_msvc_ccbin_dir}" _msvc_ccbin_dir)

        # Convert to a short (8.3) path to avoid spaces/quoting pitfalls.
        # This prevents nvcc from mis-parsing -ccbin on Windows Ninja builds.
        execute_process(
            COMMAND cmd /c for %I in ("${_msvc_ccbin_dir}") do @echo %~sI
            OUTPUT_VARIABLE _msvc_ccbin_dir_short
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )

        if(_msvc_ccbin_dir_short)
            file(TO_CMAKE_PATH "${_msvc_ccbin_dir_short}" _msvc_ccbin_dir_short)
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=${_msvc_ccbin_dir_short}")
        else()
            # Fallback: keep the original dir (may require a properly initialized dev environment).
            set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=\"${_msvc_ccbin_dir}\"")
        endif()
    endif()
endif()

find_package(CUDAToolkit 13.0 REQUIRED)

# Optional OpenCV (used for fast template matching / tracking in entry-trigger mode)
find_package(OpenCV QUIET COMPONENTS core imgproc)
if (OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    add_compile_definitions(TM_USE_OPENCV=1)
else()
    message(STATUS "OpenCV not found; falling back to slow NCC matcher. (Install OpenCV via vcpkg or system package for speed.)")
endif()

message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Toolkit Include: ${CUDAToolkit_INCLUDE_DIRS}")

# CUDA architecture (adjust for your GPU)
# RTX 30 series: 86, RTX 40 series: 89, RTX 50 series: 120
set(CMAKE_CUDA_ARCHITECTURES 86 89 120)

# =============================================================================
# TensorRT 10.14 Configuration
# =============================================================================
set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT installation directory")

if(NOT TENSORRT_ROOT)
    # Default paths to check (including user Downloads folder)
    set(TENSORRT_SEARCH_PATHS
        "$ENV{USERPROFILE}/Downloads/TensorRT-10.14.1.48"
        "C:/TensorRT-10.14.1.48"
        "C:/TensorRT-10.14.0.16"
        "C:/TensorRT-10.14"
        "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT"
    )
    foreach(path ${TENSORRT_SEARCH_PATHS})
        if(EXISTS ${path})
            set(TENSORRT_ROOT ${path})
            break()
        endif()
    endforeach()
endif()

if(NOT TENSORRT_ROOT OR NOT EXISTS ${TENSORRT_ROOT})
    message(FATAL_ERROR
        "TensorRT not found. Set TENSORRT_ROOT environment variable or cmake -DTENSORRT_ROOT=<path>")
endif()

message(STATUS "TensorRT Root: ${TENSORRT_ROOT}")

# Find TensorRT libraries (TensorRT 10.x uses versioned names like nvinfer_10.lib)
find_library(TENSORRT_NVINFER NAMES nvinfer_10 nvinfer HINTS ${TENSORRT_ROOT}/lib REQUIRED)
find_library(TENSORRT_NVONNXPARSER NAMES nvonnxparser_10 nvonnxparser HINTS ${TENSORRT_ROOT}/lib REQUIRED)
find_library(TENSORRT_NVINFER_PLUGIN NAMES nvinfer_plugin_10 nvinfer_plugin HINTS ${TENSORRT_ROOT}/lib)

message(STATUS "TensorRT nvinfer: ${TENSORRT_NVINFER}")
message(STATUS "TensorRT nvonnxparser: ${TENSORRT_NVONNXPARSER}")

# =============================================================================
# Dependencies via vcpkg or FetchContent
# =============================================================================
include(FetchContent)

# nlohmann/json for config parsing
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(json)

# =============================================================================
# Source Files
# =============================================================================
set(CPP_SOURCES
    src/main.cpp
    src/capture/d3d11_capture.cpp
    src/capture/cuda_interop.cpp
    src/video/mf_source_reader.cpp
    src/tracking/panel_tracker.cpp
    src/tracking/edge_expand.cpp
    src/processing/roi_extractor.cpp
    src/processing/row_detector.cpp
    src/processing/template_matcher.cpp
    src/ocr/tensorrt_engine.cpp
    src/ocr/svtr_inference.cpp
    src/ocr/ctc_decoder.cpp
    src/detection/panel_finder.cpp
    src/detection/symbol_matcher.cpp
    src/detection/entry_trigger.cpp
    src/detection/glyph_ocr.cpp
    src/detection/trigger_roi_builder.cpp
    src/detection/text_parser.cpp
    src/detection/change_detector.cpp
    src/detection/yolo_detector.cpp
    src/utils/config_loader.cpp
    src/utils/roi_selector.cpp
    src/utils/roi_overlay.cpp
    src/utils/timer.cpp
    src/utils/profiler.cpp
    src/utils/logger.cpp
    src/utils/image_loader.cpp
)

set(CUDA_SOURCES
    src/processing/cuda_kernels.cu
    src/cuda/template_matching.cu
)

set(HEADER_FILES
    include/capture/d3d11_capture.h
    include/capture/cuda_interop.h
    include/video/mf_source_reader.h
    include/tracking/panel_tracker.h
    include/tracking/edge_expand.h
    include/processing/roi_extractor.h
    include/processing/row_detector.h
    include/processing/template_matcher.h
    include/processing/cuda_kernels.h
    include/ocr/tensorrt_engine.h
    include/ocr/svtr_inference.h
    include/ocr/ctc_decoder.h
    include/detection/panel_finder.h
    include/detection/symbol_matcher.h
    include/detection/entry_trigger.h
    include/detection/glyph_ocr.h
    include/detection/trigger_roi_builder.h
    include/detection/text_parser.h
    include/detection/change_detector.h
    include/detection/yolo_detector.h
    include/utils/config_loader.h
    include/utils/roi_selector.h
    include/utils/roi_overlay.h
    include/utils/timer.h
    include/utils/profiler.h
    include/utils/logger.h
    include/types.h
)

# =============================================================================
# Executable
# =============================================================================
add_executable(${PROJECT_NAME} ${CPP_SOURCES} ${CUDA_SOURCES} ${HEADER_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TENSORRT_ROOT}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# =============================================================================
# Linking
# =============================================================================
target_link_libraries(${PROJECT_NAME} PRIVATE
    # CUDA libraries
    CUDA::cudart
    CUDA::cuda_driver
    
    # TensorRT libraries
    ${TENSORRT_NVINFER}
    ${TENSORRT_NVONNXPARSER}
    
    # Windows libraries
    d3d11.lib
    dxgi.lib
    windowsapp.lib      # For Windows.Graphics.Capture (C++/WinRT)

    # Media Foundation (offline MP4 decode)
    mfplat.lib
    mfreadwrite.lib
    mfuuid.lib
    
    # JSON library
    nlohmann_json::nlohmann_json
)

if (OpenCV_FOUND)
    target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
endif()

# Optional: TensorRT plugin library
if(TENSORRT_NVINFER_PLUGIN)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TENSORRT_NVINFER_PLUGIN})
endif()

# =============================================================================
# C++/WinRT Configuration (for Windows.Graphics.Capture)
# =============================================================================
# C++ compile options (only for CXX, not CUDA)
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/EHsc>           # Exception handling
    $<$<COMPILE_LANGUAGE:CXX>:/W4>             # Warning level 4
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/GL>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/Od>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/Zi>
)

# Note: /await is deprecated in VS 2026. C++20 coroutines are enabled by default
# with /std:c++20 which is set via CMAKE_CXX_STANDARD 20

# CUDA compile options (separate to avoid flag conflicts)
set_source_files_properties(${CUDA_SOURCES} PROPERTIES
    COMPILE_OPTIONS "--use_fast_math;-lineinfo;--allow-unsupported-compiler"
)

# =============================================================================
# Copy TensorRT DLLs to output (for runtime)
# TensorRT 10.x uses versioned DLL names like nvinfer_10.dll
# =============================================================================
set(TENSORRT_DLLS "")
foreach(_dll_name IN ITEMS nvinfer_10 nvinfer_plugin_10 nvonnxparser_10
                           nvinfer_builder_resource_sm120_10
                           nvinfer_builder_resource_sm89_10
                           nvinfer_builder_resource_sm86_10)
    if(EXISTS "${TENSORRT_ROOT}/bin/${_dll_name}.dll")
        list(APPEND TENSORRT_DLLS "${TENSORRT_ROOT}/bin/${_dll_name}.dll")
    elseif(EXISTS "${TENSORRT_ROOT}/lib/${_dll_name}.dll")
        list(APPEND TENSORRT_DLLS "${TENSORRT_ROOT}/lib/${_dll_name}.dll")
    endif()
endforeach()

if(TENSORRT_DLLS)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TENSORRT_DLLS}
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
        COMMENT "Copying TensorRT DLLs..."
    )
else()
    message(WARNING "TensorRT DLLs not found under ${TENSORRT_ROOT}/bin or ${TENSORRT_ROOT}/lib; skipping post-build copy.")
endif()

# =============================================================================
# Install
# =============================================================================
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/models/
    DESTINATION bin/models
    FILES_MATCHING PATTERN "*.engine" PATTERN "*.onnx"
)

# =============================================================================
# Test Applications
# =============================================================================

# D3D11-CUDA Interop Test (Step 2 & 3 verification)
add_executable(test_interop
    src/test_interop.cpp
    src/capture/d3d11_capture.cpp
    src/capture/cuda_interop.cpp
    src/processing/cuda_kernels.cu
    src/utils/timer.cpp
)

target_include_directories(test_interop PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(test_interop PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    d3d11.lib
    dxgi.lib
    windowsapp.lib
)

# CUDA compile options for test_interop
set_source_files_properties(src/processing/cuda_kernels.cu PROPERTIES
    COMPILE_OPTIONS "--use_fast_math;-lineinfo;--allow-unsupported-compiler"
)

# Copy runtime DLLs to test_interop output for easy execution from build dir
set(CUDA_RUNTIME_DLLS "")
set(_cuda_bin "")
if(DEFINED CUDAToolkit_BIN_DIR)
    set(_cuda_bin "${CUDAToolkit_BIN_DIR}")
elseif(DEFINED CUDAToolkit_ROOT)
    set(_cuda_bin "${CUDAToolkit_ROOT}/bin")
elseif(DEFINED ENV{CUDA_PATH})
    set(_cuda_bin "$ENV{CUDA_PATH}/bin")
endif()

if(_cuda_bin AND EXISTS "${_cuda_bin}")
    file(GLOB CUDA_RUNTIME_DLLS "${_cuda_bin}/cudart64_*.dll")
endif()

if(CUDA_RUNTIME_DLLS)
    add_custom_command(TARGET test_interop POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CUDA_RUNTIME_DLLS}
            $<TARGET_FILE_DIR:test_interop>
        COMMENT "Copying CUDA runtime DLLs for test_interop..."
    )
endif()

if(TENSORRT_DLLS)
    add_custom_command(TARGET test_interop POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TENSORRT_DLLS}
            $<TARGET_FILE_DIR:test_interop>
        COMMENT "Copying TensorRT DLLs for test_interop..."
    )
endif()

# Note: /await is deprecated in VS 2026. C++20 coroutines are enabled by default
target_compile_options(test_interop PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/Od>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/Zi>
)

# TensorRT OCR Test (Step 4 verification)
add_executable(test_tensorrt
    src/test_tensorrt.cpp
    src/ocr/tensorrt_engine.cpp
    src/ocr/svtr_inference.cpp
    src/ocr/ctc_decoder.cpp
    src/utils/timer.cpp
)

target_include_directories(test_tensorrt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TENSORRT_ROOT}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(test_tensorrt PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
    ${TENSORRT_NVINFER}
    ${TENSORRT_NVONNXPARSER}
)

target_compile_options(test_tensorrt PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/Od>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:/Zi>
)

if(TENSORRT_DLLS)
    add_custom_command(TARGET test_tensorrt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TENSORRT_DLLS}
            $<TARGET_FILE_DIR:test_tensorrt>
        COMMENT "Copying TensorRT DLLs for test_tensorrt..."
    )
endif()

# =============================================================================
# Unit Testing (optional)
# =============================================================================
option(BUILD_TESTS "Build unit tests" OFF)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

