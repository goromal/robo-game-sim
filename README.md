# Robo Game Sim

![](mindless-example.gif)

A minigame resembling multi-player air hockey for testing and developing optimal control and reinforcement learning algorithms.

## Dependencies

  * (CMake, C++ build tools, etc.)
  * libeigen3-dev
  * libyaml-cpp-dev
  * libboost-all-dev

## Installation

To build the repository, execute the following

```bash
git clone --recurse-submodules https://github.com/goromal/robo-game-sim.git
cd robo-game-sim/py
cmake ..
make
```

## Usage

To run the sim and view the results, run the matlab script *matlab/visualize\_game.m*. Simulation parameters can be modified by changing the values in *param/ss_game_to_3.yaml* or by making a new YAML file and pointing to it in the matlab script.

**More functionality coming soon!**
