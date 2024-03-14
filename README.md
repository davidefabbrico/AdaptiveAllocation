<!-- README.md is generated from README.Rmd. Please edit that file -->

# AdaptiveAllocation

<!-- badges: start -->

[![CRAN
status](https://img.shields.io/cran/v/invent)](https://CRAN.R-project.org/package=invent)
[![Lifecycle:
stable](https://img.shields.io/badge/lifecycle-stable-brightgreen.svg)](https://lifecycle.r-lib.org/articles/stages.html#stable)
<!-- badges: end -->

An R package for Adaptive Allocation in Finite Gaussian Mixture Model

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
specific tasks. To showcase the functionality, we will go through a
simple example that illustrates the use of one of these functions. This
is what `AdaptiveAllocation::contPlot2d()`, `AdaptiveAllocation::ssg()` and  `AdaptiveAllocation::ssg()` do.

``` r
# Load the R package
library(AdaptiveAllocation)

# Generate synthetic data
data <- AdaptiveAllocation::genGM(n = 100, K = 3, d = 2)

# If d = 2 you can plot the generated mixture model
plotCluster <- AdaptiveAllocation::contPlot2d(data = data[[1]])
print(plotCluster)

# Run the Adaptive Allocation 
res <- AdaptiveAllocation::ssgd(as.matrix(data[[1]]), K = 3, hyper = c(1, 1, 0, 0.1, 1, 1))



# For 1-dimensional gaussian data there is also a specific function
# Generate synthetic data
rmix <- function(n, pi, mu, s){
  z <- sample(1:length(pi), prob = pi, size = n, replace=TRUE)
  x <- rnorm(n, mu[z], s[z])
  return(x)
}
x <- rmix(n = 1000, pi = c(0.5, 0.5), mu = c(-2, 2), s = c(1, 1))
# hist(x)

# Run the Adaptive Allocation 
res <- AdaptiveAllocation::ssg(x, hyper = c(1, 1, 0, 0.1, 1, 1), K = 2,
                                iteration = 10000, thin = 20, burnin = 500)
