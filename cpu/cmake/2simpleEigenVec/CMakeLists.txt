cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(eigen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#使用Eigen库，我们需要在系统上找到它的头文件
find_package(Eigen3 3.3 REQUIRED CONFIG)
#CheckCXXCompilerFlag.cmake标准模块文件
include(CheckCXXCompilerFlag)

#检查-march=native编译器标志是否工作
check_cxx_compiler_flag("-march=native" _march_native_works)
#另一个选项-xHost编译器标志也开启
check_cxx_compiler_flag("-xHost" _xhost_works)

# 设置了一个空变量_CXX_FLAGS，来保存刚才检查的两个编译器中找到的编译器标志。
# 如果看到_march_native_works，我们将_CXX_FLAGS设置为-march=native。
# 如果看到_xhost_works，我们将_CXX_FLAGS设置为-xHost。
# 如果它们都不起作用，_CXX_FLAGS将为空，并禁用矢量化

set(_CXX_FLAGS)
if(_march_native_works)
  message(STATUS "Using processor's vector instructions \
                 (-march=native compiler flag set)")
  set(_CXX_FLAGS "-march=native")
elseif(_xhost_works)
  message(STATUS "Using processor's vector instructions \
                 (-xHost compiler flag set)")
  set(_CXX_FLAGS "-xHost")
else()
  message(STATUS "No suitable compiler flag found for vectorization")
endif()

#为了便于比较，我们还为未优化的版本定义了一个可执行目标，不使用优化标志
add_executable(linear-algebra-unoptimized linear-algebra.cpp)
target_link_libraries(linear-algebra-unoptimized
                       PRIVATE Eigen3::Eigen)

#定义一个优化版本
add_executable(linear-algebra linear-algebra.cpp)
target_compile_options(linear-algebra 
                       PRIVATE ${_CXX_FLAGS})
target_link_libraries(linear-algebra 
                       PRIVATE Eigen3::Eigen)

