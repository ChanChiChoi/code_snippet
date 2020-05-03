cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(hello LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)


set(USE_LIBRARY OFF)
message(STATUS "Compile sources into a library? ${USE_LIBRARY}")

# 关闭生成动态库 则add_library默认不生成
set(BUILD_SHARED_LIBS OFF)

list(APPEND _sources Message.hpp Message.cpp)


if(USE_LIBRARY)
  # add_library生成静态库
  add_library(message ${_sources})
  add_executable(hello helloWorld.cpp)
  target_link_libraries(hello message)
else()
  add_executable(hello helloWorld.cpp ${_sources})
endif()

