
# This script investigates the reported discrepancies between R and coreforecast
# when computing the Guerrero Box-Cox lambda. 

library(forecast)
library(tsibbledata)

forecast::BoxCox.lambda(fma::condmilk)

ts_gas = ts(as.numeric(tsibbledata::aus_production$Gas), start=c(1956,1), frequency=4)
forecast::BoxCox.lambda(ts_gas)

