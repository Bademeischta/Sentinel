#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mcts.h"
#include "position.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_mcts, m) {
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<const std::string&, int, int>(),
             py::arg("model_path"), py::arg("num_simulations") = 100, py::arg("threads") = 1)
        .def("run", [](MCTS &self, const std::string &fen) {
            Position pos(fen);
            return self.run(pos);
        });
}
