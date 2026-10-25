# Called only when neither CMAKE_CUDA_ARCHITECTURES nor CUDAARCHS was supplied.
set(MOLLM_CUDA_ARCHITECTURE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}")
function(mollm_detect_cuda_architectures result)
    set(fallback 75 80 86)
    if(CMAKE_CROSSCOMPILING)
        message(STATUS "CUDA architectures: ${fallback} (cross-compiling; override for deployment)")
        set(${result} "${fallback}" PARENT_SCOPE)
        return()
    endif()

    # Query cudart instead of nvidia-smi so CUDA_VISIBLE_DEVICES is respected
    # and no external management utility is required. Recheck on reconfigure.
    unset(MOLLM_CUDA_ARCH_RUN CACHE)
    unset(MOLLM_CUDA_ARCH_COMPILED CACHE)
    try_run(MOLLM_CUDA_ARCH_RUN MOLLM_CUDA_ARCH_COMPILED
        "${CMAKE_BINARY_DIR}/CMakeFiles/mollm-cuda-detect"
        "${MOLLM_CUDA_ARCHITECTURE_SOURCE_DIR}/DetectCudaArchitectures.cpp"
        LINK_LIBRARIES CUDA::cudart
        RUN_OUTPUT_VARIABLE detected)
    string(STRIP "${detected}" detected)
    if(NOT MOLLM_CUDA_ARCH_COMPILED OR NOT "${MOLLM_CUDA_ARCH_RUN}" STREQUAL "0"
       OR NOT detected MATCHES "^[0-9]+(;[0-9]+)*$")
        message(STATUS "CUDA architectures: ${fallback} (no usable GPU probe; override for deployment)")
        set(${result} "${fallback}" PARENT_SCOPE)
        return()
    endif()

    set(selected)
    list(REMOVE_DUPLICATES detected)
    foreach(architecture IN LISTS detected)
        # Native NVFP4 MMA requires the architecture-specific feature image;
        # generic PTX compiled for an older GPU cannot recover that code.
        if(architecture STREQUAL "120" OR architecture STREQUAL "121")
            set(architecture "${architecture}a")
        endif()
        unset(MOLLM_CUDA_ARCH_SUPPORTED CACHE)
        # try_compile propagates this variable from the calling scope too.
        set(CMAKE_CUDA_ARCHITECTURES "${architecture}")
        set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
        try_compile(MOLLM_CUDA_ARCH_SUPPORTED
            "${CMAKE_BINARY_DIR}/CMakeFiles/mollm-cuda-target"
            "${MOLLM_CUDA_ARCHITECTURE_SOURCE_DIR}/CudaArchitectureProbe.cu"
            CMAKE_FLAGS "-DCMAKE_CUDA_ARCHITECTURES:STRING=${architecture}"
            OUTPUT_VARIABLE compiler_output)
        if(NOT MOLLM_CUDA_ARCH_SUPPORTED)
            message(FATAL_ERROR
                "Detected GPU requires CUDA target ${architecture}, but the selected compiler cannot build it. "
                "Use a supporting CUDA toolkit, or deliberately override CMAKE_CUDA_ARCHITECTURES.\n"
                "${compiler_output}")
        endif()
        list(APPEND selected "${architecture}")
    endforeach()
    message(STATUS "CUDA architectures (detected GPUs): ${selected}")
    set(${result} "${selected}" PARENT_SCOPE)
endfunction()
