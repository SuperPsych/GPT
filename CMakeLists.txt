cmake_minimum_required(VERSION 3.30)
project(NeuralNet)

set(CMAKE_CXX_STANDARD 20)

add_executable(NeuralNet main.cpp
        data_gen.cpp
)
