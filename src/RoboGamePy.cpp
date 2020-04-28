#include "GameSim.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
//#include <pybind11/operators.h>
//#include <sstream>

using namespace Eigen;
namespace py = pybind11;

PYBIND11_MODULE(robo_game_py, m)
{
  m.doc() = "Python binding module for the robo game sim class.";

  py::class_<GameSim>(m, "GameSim")
          .def(py::init())
          .def("reset", &GameSim::reset)
          .def("undecided", &GameSim::undecided)
          .def("run", &GameSim::run)
          .def("__repr__",
               [](const GameSim &gs) {
                return "GameSim Object";
                }
  );
}

