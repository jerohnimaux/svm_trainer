cmake_minimum_required(VERSION 3.12)
project(svm_trainer)

set(CMAKE_CXX_STANDARD 14)

set(LIBFOLDER "libs")
set(VISIONHEAD_FOLDER "visionhead")
set(VISIONDOWN_FOLDER "visiondown")

#Matlab libs
set(MATLIB
        "${CMAKE_CURRENT_LIST_DIR}/${LIBFOLDER}/matlab/libMatlabDataArray.so"
        "${CMAKE_CURRENT_LIST_DIR}/${LIBFOLDER}/matlab/libMatlabEngine.so")

link_directories("/home/larnal/Programmes/Matlab/bin/glnxa64"
        "/home/larnal/Programmes/Matlab/extern/bin/glnxa64")

include_directories("${CMAKE_CURRENT_LIST_DIR}/${LIBFOLDER}/matlab/include")


add_executable(svm_trainer matdgns.c)

target_link_libraries(svm_trainer libMatlabEngine.so libMatlabDataArray.so libmat.so libmx.so)
message(STATUS "${MATLIB}")