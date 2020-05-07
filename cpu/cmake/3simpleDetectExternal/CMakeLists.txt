cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(findPython LANGUAGES C)

# 查找python解释器
find_package(PythonInterp REQUIRED)

execute_process(COMMAND
               ${PYTHON_EXECUTABLE} "-c" "print('Hello, World')"
               RESULT_VARIABLE _status
               OUTPUT_VARIABLE _hello_world
               ERROR_QUIET
               OUTPUT_STRIP_TRAILING_WHITESPACE
               )

message(STATUS "RESULT_VARIABLE is: ${_status}")
message(STATUS "OUTPUT_VARIABLE is: ${_hello_world}")

include(CMakePrintHelpers)
cmake_print_variables(_status _hello_world)

#查找pythnon库
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_C_STANDARD_REQUIRED ON)

#查找python头文件和库的模块，称为FindPythonLibs.cmake
find_package(PythonLibs 
             ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} 
             EXACT REQUIRED)

add_executable(hello-embedded-python hello-embedded-python.c)
target_include_directories(hello-embedded-python
                           PRIVATE
                           ${PYTHON_INCLUDE_DIRS})
target_link_libraries(hello-embedded-python
                      PRIVATE
                      ${PYTHON_LIBRARIES})
