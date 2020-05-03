cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(hello)

set(CMAKE_CXX_STANDARD 11)
add_executable(hello helloWorld.cpp)

add_library(message
            STATIC
             Message.cpp
             Message.hpp)

target_link_libraries(hello message)
