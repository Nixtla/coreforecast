#include "bindings.h"

#include "seasonal.h"

template <typename T> int Period(const py::array_t<T> data, size_t max_lag) {
  T out;
  const auto x = AsContiguous(data);
  seasonal::GreatestAutocovariance(View(x), std::span<T>{&out, 1},
                                   static_cast<index_t>(max_lag));
  return static_cast<int>(out);
}

void init_seas(py::module_ &m) {
  py::module_ seas = m.def_submodule("seasonal");
  seas.def("period", &Period<double>);
  seas.def("period", &Period<float>);
}
