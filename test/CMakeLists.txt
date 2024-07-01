
find_package(Catch2 REQUIRED)
include (CTest)
include(Catch)

function(add_test_executable)

    set(EXEC_NAME ${ARGV0})
    cmake_parse_arguments(ARGS 
        ""
        ""
        "SRCS;LIBS"
        ${ARGN})
    
    add_executable(${EXEC_NAME} ${ARGS_SRCS})
    target_link_libraries(${EXEC_NAME} PRIVATE Catch2::Catch2WithMain)
    if(NOT ${ARGS_LIBS} EQUAL "")
        target_link_libraries(${EXEC_NAME} PRIVATE ${ARGS_LIBS})
    endif()
    
    catch_discover_tests(${EXEC_NAME})
    message(STATUS "Adding test executable: ${EXEC_NAME}")
endfunction()

# CPU tests
add_test_executable(Vectors_cpu             SRCS cpu/Vectors.cpp            LIBS cuvoxmap_cpu)
add_test_executable(ArrayIndexing_cpu       SRCS cpu/ArrayIndexing.cpp      LIBS cuvoxmap_cpu)
add_test_executable(ArrayAllocator_cpu      SRCS cpu/ArrayAllocator.cpp     LIBS cuvoxmap_cpu)
add_test_executable(MapAllocator_cpu        SRCS cpu/MapAllocator.cpp       LIBS cuvoxmap_cpu)
add_test_executable(MapAccessorHost_cpu     SRCS cpu/MapAccessorHost.cpp    LIBS cuvoxmap_cpu)
add_test_executable(RayCaster_cpu           SRCS cpu/RayCaster.cpp          LIBS cuvoxmap_cpu)
add_test_executable(Cuvoxmap_cpu            SRCS cpu/Cuvoxmap.cpp           LIBS cuvoxmap_cpu)
add_test_executable(GlobLocalCvt_cpu        SRCS cpu/GlobLocalCvt.cpp       LIBS cuvoxmap_cpu)

# GPU tests
add_test_executable(Vectors_gpu             SRCS gpu/Vectors.cu             LIBS cuvoxmap_gpu)
add_test_executable(ArrayIndexing_gpu       SRCS gpu/ArrayIndexing.cu       LIBS cuvoxmap_gpu)
add_test_executable(ArrayAllocator_gpu      SRCS gpu/ArrayAllocator.cu      LIBS cuvoxmap_gpu)
add_test_executable(MapAllocator_gpu        SRCS gpu/MapAllocator.cu        LIBS cuvoxmap_gpu)
add_test_executable(MapAccessorHost_gpu     SRCS gpu/MapAccessorHost.cu     LIBS cuvoxmap_gpu)
add_test_executable(MapAccessorDevice_gpu   SRCS gpu/MapAccessorDevice.cu   LIBS cuvoxmap_gpu)
add_test_executable(RayCaster_gpu           SRCS gpu/RayCaster.cu           LIBS cuvoxmap_gpu)



