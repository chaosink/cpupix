cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
project(cpupix LANGUAGES CXX)

# C++ flags
set(CMAKE_CXX_STANDARD 11)
if(MSVC)
	add_definitions(-DCONSTEXPR=const -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS)
else()
	add_definitions(-DCONSTEXPR=constexpr)
	set(CXX_WARNINGS "-Wall -Wextra -Wpointer-arith -Wcast-align -fstrict-aliasing -Wno-unused-local-typedefs -Wno-misleading-indentation -Wno-maybe-uninitialized -Wno-int-in-bool-context -Wno-implicit-fallthrough")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_WARNINGS} -fvisibility-inlines-hidden -O3")
endif()

find_package(OpenMP)
if(OPENMP_FOUND)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(
	src/core
	src/utility
	src/thirdparty
	src/thirdparty/glew-2.1.0/include
	src/thirdparty/glfw-3.2.1/include
	src/thirdparty/glm-0.9.8.5
)

add_library(core
	src/core/CPUPix.cpp
	src/core/CPUPixKernel.cpp
)

add_executable(cpupix
	src/cpupix/cpupix.cpp
	src/utility/Camera.cpp
	src/utility/Model.cpp
	src/utility/FPS.cpp
	src/utility/Texture.cpp
	src/utility/Video.cpp
	src/utility/Toggle.cpp
)
target_link_libraries(cpupix
	core
	glew_s
	glfw
)

add_executable(cpupix_gl
	src/cpupix_gl/cpupix_gl.cpp
	src/cpupix_gl/OGL.cpp
	src/utility/Camera.cpp
	src/utility/Model.cpp
	src/utility/FPS.cpp
	src/utility/Texture.cpp
	src/utility/Video.cpp
	src/utility/Toggle.cpp
)
target_link_libraries(cpupix_gl
	glew_s
	glfw
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/utility/raw2mp4.sh ${CMAKE_CURRENT_BINARY_DIR}/raw2mp4.sh COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/font/bitmap_font.data ${CMAKE_CURRENT_BINARY_DIR}/font/bitmap_font.data COPYONLY)
file(GLOB textures RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/texture/*)
foreach(tex ${textures})
	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${tex} ${CMAKE_CURRENT_BINARY_DIR}/${tex} COPYONLY)
endforeach()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/cpupix_gl/shader/vertex.glsl ${CMAKE_CURRENT_BINARY_DIR}/shader/vertex.glsl COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/cpupix_gl/shader/fragment.glsl ${CMAKE_CURRENT_BINARY_DIR}/shader/fragment.glsl COPYONLY)



option(BUILD_UTILS "utilities" OFF)
add_subdirectory(src/thirdparty/glew-2.1.0/build/cmake)

option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
add_subdirectory(src/thirdparty/glfw-3.2.1)
