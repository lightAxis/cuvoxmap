
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

add_test_executable(Vectors_cpu         SRCS Vectors.cpp        LIBS cuvoxmap_cpu)
add_test_executable(ArrayIndexing_cpu   SRCS ArrayIndexing.cpp  LIBS cuvoxmap_cpu)
add_test_executable(ArrayAllocator_cpu  SRCS ArrayAllocator.cpp LIBS cuvoxmap_cpu)
add_test_executable(MapAllocator_cpu    SRCS MapAllocator.cpp   LIBS cuvoxmap_cpu)
add_test_executable(MapAccessorHost_cpu SRCS MapAccessorHost.cpp LIBS cuvoxmap_cpu)
add_test_executable(RayCaster_cpu       SRCS RayCaster.cpp      LIBS cuvoxmap_cpu)
add_test_executable(Cuvoxmap_cpu        SRCS Cuvoxmap.cpp       LIBS cuvoxmap_cpu)
add_test_executable(GlobLocalCvt_cpu    SRCS GlobLocalCvt.cpp   LIBS cuvoxmap_cpu)

add_test_executable(Vectors_gpu         SRCS Vectors.cu         LIBS cuvoxmap_gpu)
add_test_executable(ArrayIndexing_gpu   SRCS ArrayIndexing.cu   LIBS cuvoxmap_gpu)
add_test_executable(ArrayAllocator_gpu  SRCS ArrayAllocator.cu  LIBS cuvoxmap_gpu)
add_test_executable(MapAllocator_gpu    SRCS MapAllocator.cu    LIBS cuvoxmap_gpu)
add_test_executable(MapAccessorHost_gpu SRCS MapAccessorHost.cu LIBS cuvoxmap_gpu)
add_test_executable(MapAccessorDevice_gpu SRCS MapAccessorDevice.cu LIBS cuvoxmap_gpu)



