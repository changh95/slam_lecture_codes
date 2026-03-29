link_directories(
  ${PROJECT_BINARY_DIR}/test
)


# Build gtest
add_library(gtest STATIC gtest/src/gtest-all.cc)
target_include_directories(gtest
  PUBLIC
  ${PROJECT_SOURCE_DIR}/test/gtest/include
  ${PROJECT_SOURCE_DIR}/test/gtest
  ${PROJECT_SOURCE_DIR}/test
)

add_library(gtest_main STATIC gtest/src/gtest_main.cc)
target_link_libraries(gtest_main gtest)

execute_process(COMMAND cmake -E remove_directory ${CMAKE_BINARY_DIR}/test_results)
execute_process(COMMAND cmake -E make_directory ${CMAKE_BINARY_DIR}/test_results)

set(tests
     console_TEST.cc)

#################################################
# Build all the tests
foreach(GTEST_SOURCE_file ${tests})
  string(REGEX REPLACE ".cc" "" BINARY_NAME ${GTEST_SOURCE_file})
  add_executable(${BINARY_NAME} ${GTEST_SOURCE_file})

  add_dependencies(${BINARY_NAME}
      console_bridge
      gtest
      gtest_main)

  target_link_libraries(${BINARY_NAME}
     gtest_main
     gtest
     console_bridge)
  if (UNIX AND NOT ANDROID AND NOT QNX)
    target_link_libraries(${BINARY_NAME} pthread)
  endif()

   add_test(NAME    ${BINARY_NAME}
            COMMAND ${BINARY_NAME}
                    --gtest_output=xml:${CMAKE_BINARY_DIR}/test_results/${BINARY_NAME}.xml)

   set_tests_properties(${BINARY_NAME} PROPERTIES TIMEOUT 240)
endforeach()

find_package(PythonInterp 3 REQUIRED)

if(NOT PYTHONINTERP_FOUND)
  message(WARNING "No PythonInterp found. Linters will not be executed")
  return()
endif()

find_program(WGET_EXE wget)
if(WGET_EXE)
  message(STATUS "Found WGet: ${WGET_EXE}")
  add_custom_target(wget_cppcheck
                    COMMAND ${WGET_EXE} -q -O cppcheck.py https://raw.githubusercontent.com/ament/ament_lint/master/ament_cppcheck/ament_cppcheck/main.py
  )

  add_test(NAME console_bridge_cppcheck
           COMMAND "${PYTHON_EXECUTABLE}"
                   "cppcheck.py"
                   "--language=c++" "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/include"
  )

  add_custom_target(wget_cpplint
                    COMMAND wget -q -O cpplint.py https://raw.githubusercontent.com/ament/ament_lint/master/ament_cpplint/ament_cpplint/cpplint.py
  )

  add_test(NAME console_bridge_cpplint
           COMMAND "${PYTHON_EXECUTABLE}"
                   "cpplint.py"
                   "--counting=detailed"
                   "--extensions=cpp,h"
                   "--linelength=100"
                   "--repository=${PROJECT_SOURCE_DIR}"
                   "--filter=-build/c++11,-runtime/references,-whitespace/braces,-whitespace/indent,-whitespace/parens,-whitespace/semicolon"
           "${PROJECT_SOURCE_DIR}/include/console_bridge/console.h"
           "${PROJECT_SOURCE_DIR}/src/console.cpp"
  )

  add_dependencies(console_bridge wget_cppcheck wget_cpplint)

else()
  message(WARNING "wget not found. Linters will not be executed")
endif()
