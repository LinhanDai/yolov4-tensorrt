cmake_minimum_required(VERSION 3.2)
project(yolo_trt)
set(CMAKE_CXX_COMPILIER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,$ORIGIN")
set(CMAKE_BUILD_TYPE Debug)

FIND_PACKAGE(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

file(GLOB LIB_SOURCES  src/*.cu src/*.cpp include/*.h)

find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# cuda
find_package(CUDA)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

CUDA_ADD_LIBRARY(yoloLib SHARED ${LIB_SOURCES})
target_link_libraries(yoloLib nvinfer nvonnxparser cudart ${OpenCV_LIBS})
target_include_directories(yoloLib PRIVATE include/ ${OpenCV_INCLUDE_DIRS} ${CUDA_TOOLKIT_ROOT_DIR}/include)
add_executable(yolo_trt ${PROJECT_SOURCE_DIR}/yolov4_demo.cpp)
target_link_libraries(yolo_trt yoloLib)