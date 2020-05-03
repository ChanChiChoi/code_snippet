cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(hello LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

set(USE_LIBRARY OFF)

# 用户可输入不同选项
option(USE_LIBRARY "Compile sources into a library" OFF)
message(STATUS "Compile sources into a library? ${USE_LIBRARY}")

# 关闭生成动态库 则add_library默认不生成
set(BUILD_SHARED_LIBS OFF)

list(APPEND _sources Message.hpp Message.cpp)

#有时选项之间会有依赖的情况。
#如果没有将USE_LIBRARY逻辑设置为ON，则此选项没有任何意义。
#CMake提供cmake_dependent_option()命令用来定义依赖于其他选项的选项
include(CMakeDependentOption)

cmake_dependent_option(
   MAKE_STATIC_LIBRARY "Compile sources into a static library" OFF
   "USE_LIBRARY" ON
  )
cmake_dependent_option(
   MAKE_SHARED_LIBRARY "Compile sources into a static library" ON
   "USE_LIBRARY" ON
  )

if(USE_LIBRARY)
  # add_library生成静态库
  add_library(message ${_sources})
  add_executable(hello helloWorld.cpp)
  target_link_libraries(hello message)
else()
  add_executable(hello helloWorld.cpp ${_sources})
endif()

