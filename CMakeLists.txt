cmake_minimum_required(VERSION 3.12)
project(svm_trainer)

set(CMAKE_CXX_STANDARD 14)

set(LIBFOLDER "libs")
set(VISIONHEAD_FOLDER "visionhead")
set(VISIONDOWN_FOLDER "visiondown")

## Add Lib Dlib
set(USE_AVX_INSTRUCTIONS ON CACHE BOOL "Use AVX instructions")
set(DLIB_JPEG_SUPPORT ON CACHE BOOL "Use JPEG Support")
set(DLIB_USE_LAPACK ON CACHE BOOL "Use Lapack library")
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/${LIBFOLDER}/dlib-19.14/dlib" dlib_build)

# Add matio
#link_directories("${CMAKE_CURRENT_LIST_DIR}/${LIBFOLDER}/matio/.libs")

# Add Lib OpenCV
#find_package(OpenCV REQUIRED)

add_executable(svm_trainer main.cpp)

target_link_libraries(svm_trainer ${OpenCV_LIBS} dlib::dlib)
