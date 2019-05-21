cmake_minimum_required(VERSION 2.8)
project( fractal )
find_package( OpenCV REQUIRED)
add_executable( IFS IFS.cpp)
target_link_libraries( IFS ${OpenCV_LIBS} )

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -fopenmp")
