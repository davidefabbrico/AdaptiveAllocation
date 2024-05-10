<!-- README.md is generated from README.Rmd. Please edit that file -->

# AdaptiveAllocation

<!-- badges: start -->

[![CRAN
status](https://img.shields.io/cran/v/invent)](https://CRAN.R-project.org/package=invent)
[![Lifecycle:
stable](https://img.shields.io/badge/lifecycle-stable-brightgreen.svg)](https://lifecycle.r-lib.org/articles/stages.html#stable)
<!-- badges: end -->

An R package for Adaptive Allocation in Finite Gaussian and Categorical Mixture Model

The implementation has been done in C++ through the use of Rcpp and
RcppArmadillo.

Authors: Davide Fabbrico, Filippo Pagani, Alice Corbella, Sebastiano Grazzi, Paul Kirk and Gareth Roberts.

Maintainer: Davide Fabbrico.

## Installation

You can install the development version of AdaptiveAllocation from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("davidefabbrico/AdaptiveAllocation")
```

## Usage

In this section, we will demonstrate a basic example to show how the
functions in the R package `AdaptiveAllocation` work.

The R package contains various functions that are designed to perform
specific tasks. To show the functionality, we will go through a
simple example that illustrates the use of one of these functions.

``` r
rm(list = ls())
# load the library
library(AdaptiveAllocation)
# create the data
n <- 1000 # observation
d <- 30  # dimension
K <- 10 # cluster
data <- AdaptiveAllocation::genGaussianGM(n = n, d = d, K = K)
# visualize the scatter plot (only in 2d)
AdaptiveAllocation::scattPlot2d(data = data[[1]][,1:d])
# MCMC settings
iter <- 1000
burnin <- 0
thin <- 1
# run the Adaptive MCMC
res <- DiversityGibbs(X = data[[1]][, 1:d], K = K, iteration = iter, burnin = burnin, 
                      thin = thin, DiversityIndex = "Laplace", q = 5, method = "EB")
