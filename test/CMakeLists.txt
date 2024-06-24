
find_package(Catch2 3 REQUIRED)
include(Catch)

set(TEST_TARGETS "")

function(add_test_executable)

    set(EXEC_NAME ${ARGV0})
    cmake_parse_arguments(ARGS 
        ""
        ""
        "SRCS;LIBS"
        ${ARGN})
    
    add_executable(${EXEC_NAME} ${ARGS_SRCS})
    target_link_libraries(${EXEC_NAME} PRIVATE Catch2::Catch2)
    if(NOT ${ARGS_LIBS} EQUAL "")
        target_link_libraries(${EXEC_NAME} PRIVATE ${ARGS_LIBS})
    endif()

    set(TEST_TARGETS ${TEST_TARGETS} ${EXEC_NAME} PARENT_SCOPE)
endfunction()

add_test_executable(main_test SRCS main_test.cpp LIBS cuvoxmap)

foreach(target ${TEST_TARGETS})
    message(STATUS "Adding test : ${target}")
    catch_discover_tests(${target})
endforeach()