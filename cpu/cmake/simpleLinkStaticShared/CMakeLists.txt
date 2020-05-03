cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(hello)

set(CMAKE_CXX_STANDARD 11)
add_executable(hello helloWorld.cpp)

#生成目标文件
add_library(message-obj
           OBJECT
           Message.cpp
           Message.hpp)
#生成位置无关 -fPIC
set_target_properties(message-obj
                     PROPERTIES
                    POSITION_INDEPENDENT_CODE 1)

#引用目标文件生成静态库
add_library(message-static
            STATIC
            $<TARGET_OBJECTS:message-obj>
             )
#静态库在cmake内部需要唯一名称,但是文件名可以输出修改
set_target_properties(message-static
               PROPERTIES
               OUTPUT_NAME "message")

add_library(message-shared
            SHARED
            $<TARGET_OBJECTS:message-obj>
             )
set_target_properties(message-shared
               PROPERTIES
               OUTPUT_NAME "message")

target_link_libraries(hello message-static)
