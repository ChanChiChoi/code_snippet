cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(simpleControllCompilerFlags LANGUAGES CXX)

message(STATUS "C++ compiler flags: ${CMAKE_CXX_FLAGS}")

list(APPEND flags "-fPIC" "-Wall")
list(APPEND flags "-Wextra" "-Wpedantic")

# 虽然显示flags:-fPIC-Wall-Wextra-Wpedantic
# 变量的值放在双引号内和外 有区别
message(STATUS "flags(outside):" ${flags})
message(STATUS "flags(inside): ${flags}")

add_library(display
          STATIC
          display.h
          display.cpp)

target_compile_options(display
                PRIVATE
               ${flags})

add_executable(simpleControllCompilerFlags main.cpp)
target_compile_options(simpleControllCompilerFlags
                PRIVATE
               "-fPIC")

target_link_libraries(simpleControllCompilerFlags display)
