#include "nb.h"

void init_diff(nb::module_ &);
void init_exp_weigh(nb::module_ &);
void init_expanding(nb::module_ &);
void init_ga(nb::module_ &);
void init_rolling(nb::module_ &);
void init_scalers(nb::module_ &);

NB_MODULE(_coreforecast, m) {
  init_diff(m);
  init_exp_weigh(m);
  init_expanding(m);
  init_ga(m);
  init_rolling(m);
  init_scalers(m);
}
