cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(pose_camera_cpp)

add_definitions("-std=c++14")

find_package(OpenCV REQUIRED PATHS "d:/wondertree/opencv/build")
find_package(Torch REQUIRED PATHS "d:/wondertree/libtorch")

include_directories(include/)
file(GLOB_RECURSE SOURCES
        include/*.h
        src/*.cc
        )

add_executable(pose_camera_cpp ${SOURCES} main.cc)
target_link_libraries(pose_camera_cpp 
        ${OpenCV_LIBS} 
        ${TORCH_LIBRARIES}
        )

set_property(TARGET pose_camera_cpp PROPERTY CXX_STANDARD 11)

