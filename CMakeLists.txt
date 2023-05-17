cmake_minimum_required(VERSION 3.16)
project(untitled2)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

add_executable(untitled2 resources/MapReduceFramework.h resources/MapReduceClient.h resources/MapReduceFramework.cpp
        resources/Sample_Client/SampleClient.cpp resources/Barrier/Barrier.cpp resources/Barrier/Barrier.h)
