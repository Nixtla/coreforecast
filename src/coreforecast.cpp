#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_scalers(py::module_ &);
void init_ga(py::module_ &);

PYBIND11_MODULE(_lib, m)
{
    init_scalers(m);
    init_ga(m);
}