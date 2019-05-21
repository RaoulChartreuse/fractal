cmake_minimum_required(VERSION 2.8)
project( fractal )
find_package( OpenCV REQUIRED)
find_package( Boost COMPONENTS system filesystem REQUIRED )
add_executable( IFS IFS.cpp)
target_link_libraries( IFS ${OpenCV_LIBS} ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -fopenmp")