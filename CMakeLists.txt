cmake_minimum_required(VERSION 2.8.12)
project(robo-game-sim)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(Eigen3 REQUIRED)
add_subdirectory(lib/pybind11)

include_directories(
  include
  lib/pybind11/include
  ${EIGEN3_INCLUDE_DIRS}
)

 pybind11_add_module(robo_game_py src/RoboGamePy.cpp src/GameSim.cpp)
