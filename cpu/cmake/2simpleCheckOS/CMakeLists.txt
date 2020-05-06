cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(checkOS LANGUAGES CXX)

add_executable(helloWorld hello-world.cpp)

#------------------------------
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "Configuring on/for Linux")
  target_compile_definitions(helloWorld PUBLIC "IS_LINUX")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  message(STATUS "Configuring on/for macOS")
  target_compile_definitions(helloWorld PUBLIC "IS_MACOS")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  message(STATUS "Configuring on/for Windows")
  target_compile_definitions(helloWorld PUBLIC "IS_WINDOWS")
elseif(CMAKE_SYSTEM_NAME STREQUAL "AIX")
  message(STATUS "Configuring on/for IBM AIX")
  target_compile_definitions(helloWorld PUBLIC "IS_IBM_AIX")
else()
  message(STATUS "Configuring on/for ${CMAKE_SYSTEM_NAME}")
endif()

#-------------------------------
target_compile_definitions(helloWorld PUBLIC "COMPILER_NAME=${CMAKE_CXX_COMPILER_ID}")

if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
  target_compile_definitions(helloWorld PUBLIC "IS_INTEL_CXX_COMPILER")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
  target_compile_definitions(helloWorld PUBLIC "IS_GNU_CXX_COMPILER")
endif()

#--------------------------------
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  target_compile_definitions(helloWorld PUBLIC "IS_64_BIT_ARCH")
  message(STATUS "Target is 64 bits")
else()
  target_compile_definitions(helloWorld PUBLIC "IS_32_BIT_ARCH")
  message(STATUS "Target is 32 bits")
endif()

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "i386")
  message(STATUS "i386 architecture detected")
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "i686")
  message(STATUS "i686 architecture detected")
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
  message(STATUS "x86_64 architecture detected")
else()
  message(STATUS "host processor architecture is unkonw")
endif()
  target_compile_definitions(helloWorld PUBLIC "ARCHITECTURE=${CMAKE_HOST_SYSTEM_PROCESSOR}")
