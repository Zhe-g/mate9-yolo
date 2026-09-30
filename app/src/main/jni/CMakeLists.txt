cmake_minimum_required(VERSION 3.10)
project(yolov8ncnn)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.12.0-android/OpenCV-mobile-4.12.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

if(NOT ANDROID_ABI AND CMAKE_ANDROID_ARCH_ABI)
    set(ANDROID_ABI ${CMAKE_ANDROID_ARCH_ABI})
endif()
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20250916-android-vulkan/ncnn-20250916-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(yolov8ncnn SHARED yolov8ncnn.cpp yolo.cpp ndkcamera.cpp)

target_link_libraries(yolov8ncnn ncnn ${OpenCV_LIBS} camera2ndk mediandk)
