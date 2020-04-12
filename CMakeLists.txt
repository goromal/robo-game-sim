cmake_minimum_required(VERSION 2.8.12)
project(robo-game-sim)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(Eigen3 REQUIRED)
add_subdirectory(lib/pybind11)
add_subdirectory(lib/math-utils-lib)
add_subdirectory(lib/logging-utils-lib)
add_subdirectory(lib/progress_bar)

include_directories(
  include
  lib/pybind11/include
  lib/math-utils-lib/include
  lib/logging-utils-lib/include
  lib/progress_bar/include
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(internal-game
  src/InternalGame.cpp
  src/ClassicalTeam.cpp
  src/GameSim.cpp
)
target_link_libraries(internal-game
  math-utils-lib
  logging-utils-lib
  progress_bar
  yaml-cpp
  stdc++fs
)

# pybind11_add_module(robo_game_py src/RoboGamePy.cpp)
