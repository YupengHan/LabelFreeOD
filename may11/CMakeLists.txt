cmake_minimum_required(VERSION 2.8.3)
project(may11)


find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

find_package(OpenCV REQUIRED)
set(OpenCV_DIR /usr/local/share/OpenCV)

find_package(CUDA REQUIRED)


catkin_package(
  INCLUDE_DIRS
  CATKIN_DEPENDS  roscpp
                  sensor_msgs
                  std_msgs
                  cv_bridge
                  image_transport
)


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)


set(CUDA_NVCC_FLAGS "-arch=compute_30" CACHE STRING "nvcc flags" FORCE)
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE) 
set(LIB_TYPE STATIC)
cuda_add_library(testCUDALib ${LIB_TYPE} src/template.cu)

add_executable(subImg src/subImg.cpp)
add_dependencies(subImg testCUDALib)
target_link_libraries(subImg ${catkin_LIBRARIES} ${OpenCV_LIBS} testCUDALib)

add_executable(clrmsk src/color_mask_loc.cpp)
add_dependencies(clrmsk testCUDALib)
target_link_libraries(clrmsk ${catkin_LIBRARIES} ${OpenCV_LIBS} testCUDALib)



add_executable(saveImg src/saveimg.cpp)
add_dependencies(saveImg testCUDALib)
target_link_libraries(saveImg ${catkin_LIBRARIES} ${OpenCV_LIBS} testCUDALib)

add_executable(backgroundSub src/bcsub.cpp)
add_dependencies(backgroundSub testCUDALib)
target_link_libraries(backgroundSub ${catkin_LIBRARIES} ${OpenCV_LIBS} testCUDALib)

add_executable(depthBG src/bcdep.cpp)
add_dependencies(depthBG testCUDALib)
target_link_libraries(depthBG ${catkin_LIBRARIES} ${OpenCV_LIBS} testCUDALib)