cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(controlFlow LANGUAGES CXX)

add_library(display
           STATIC
           display1.hpp display1.cpp
           display2.hpp display2.cpp
          )
target_compile_options(display
                      PRIVATE
                       -O3)

list(APPEND sources_lower_optim
          display1.cpp display2.cpp)
message(STATUS "for echo:${sources_lower_optim}")

message(STATUS "Setting source properties using In LISTS syntax:")
foreach(_source IN LISTS sources_lower_optim)
  set_source_files_properties(${_source}
                             PROPERTIES
                            COMPILE_FLAGS -O2)
  message(STATUS "Appending -O2 flag for ${_source}")
endforeach()

message(STATUS "Querying sources properties using plain syntax:")
#foreach(_source "display1.cpp;display2.cpp") # 不能简单理解成是字符串
foreach(_source ${sources_lower_optim})
  get_source_file_property(_flags ${_source} COMPILE_FLAGS)
  message(STATUS "Source ${_source} has the following extra COMPILE_FLAGS: ${_flags}")
endforeach()

add_executable(main main.cpp)
target_link_libraries(main display)

