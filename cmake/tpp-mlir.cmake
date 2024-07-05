# If TPP-MLIR is in library path, add it to the dependencies
# This should be the build directory, not the source or the 'lib'
# FIXME: Make this an actual CMake discovery
if (TPP_MLIR_DIR)
    message(STATUS "TPP-MLIR at ${TPP_MLIR_DIR}")
    add_compile_definitions(TPP_MLIR)
    set(TPP_MLIR_LIBS
            TPPPipeline
            tpp_xsmm_runner_utils
        )
    function(enable_tpp_mlir target)
        target_include_directories(${target} PRIVATE ${TPP_MLIR_DIR}/../include ${TPP_MLIR_DIR}/include)
        target_link_directories(${target} PRIVATE ${TPP_MLIR_DIR}/lib)
        target_link_libraries(${target} PRIVATE ${TPP_MLIR_LIBS})
    endfunction()
else()
    function(enable_tpp_mlir target)
        message(DEBUG "TPP-MLIR not enabled, skipping ${target}")
    endfunction()
endif()
