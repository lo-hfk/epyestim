# epyestim

## Introduction

epyestim estimates the effective reproduction number
from time series of reported case numbers of epidemics. It is a
Python reimplementation of the method outlined by
Huisman et al. [1], making use of the method by Cori et al. [2]
to estimate the reproduction number R from infection data, 
available in the R package EpiEstim [3].

The main steps for estimation of the effective reproduction number are:

  * Bootstrapping the series of reported case numbers
  * Smoothing using a LOWESS filter
  * MLE of the infection incidence time series
    using an adaptation of the Richardson-Lucy deconvolution algorithm.
  * Bayesian estimation of the effective reproduction number using the
    method of Cori et al. [2]
    
Aggregate estimates for the reproduction number are obtained by bootstrap
aggregation (bagging).

The user can choose to output either time-varying estimates or piecewise
constant estimates on fixed arbitrary time intervals.

## How to install epyestim

```
pip install epyestim
```

## How to use epyestim

Basic usage of the epyestim package applied to COVID-19 data is explained
in the [Jupyter tutorial notebook](https://github.com/lo-hfk/epyestim/blob/main/notebooks/covid_tutorial.ipynb).

The core functions relevant for users are:

* `epyestim.bagging_r` for the complete estimation process
  outlined above
* `epyestim.covid19.r_covid` for the same function with default
  parameters for COVID-19
* `epyestim.estimate_r.estimate_r` for the R estimation from
  infection numbers, based on the EpiEstim package

## Authors

* [Lorenz Hilfiker](mailto:lorenz.hilfiker@gmail.com)
* [Johannes Josi](mailto:johannes@josi.info)

## How to contribute

Error reports and suggestions for improvements via GitHub issues
are very welcome.

## References

[1] Jana S. Huisman, Jeremie Scire, Daniel Angst, Richard Neher,
Sebastian Bonhoeffer, Tanja Stadler: A method to monitor the effective
reproductive number of SARS-CoV-2
https://ibz-shiny.ethz.ch/covid-19-re/methods.pdf

[2] Anne Cori, Neil M. Ferguson, Christophe Fraser, Simon Cauchemez:
A New Framework and Software to Estimate Time-Varying Reproduction
Numbers During Epidemics, American Journal of Epidemiology, Volume 178,
Issue 9, 1 November 2013, Pages 1505–1512,
https://doi.org/10.1093/aje/kwt133

[3] EpiEstim CRAN package:
https://cran.r-project.org/web/packages/EpiEstim/index.html
