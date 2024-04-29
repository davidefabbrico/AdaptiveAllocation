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
library(AdaptiveAllocation)
library(MASS)
library(ellipse)
library(ggplot2)

# number of dimension
d <- 2
# number of observation
n <- 1600
set.seed(2)
# set the values for the mean
mu1 <- c(0, -2)
mu2 <- c(0, 2)
# set the values for the precision
sigma1 <- matrix(c(1, 0, 0, 1), nrow = 2) 
sigma2 <- matrix(c(1, 0, 0, 1), nrow = 2) 
# create the two cluster
cluster1 <- MASS::mvrnorm(floor(n/2), mu1, sigma1)
cluster2 <- MASS::mvrnorm(floor(n/2), mu2, sigma2)
result <- rbind(cluster1, cluster2)
# add the true allocation
cluster_id <- rep(0:1, each = floor(n/2))
result <- cbind(result, cluster_id)
data <- result
resTest <- result[,1:2]
# print the scatter plot
plotClusterScatter <- AdaptiveAllocation::scattPlot2d(data = resTest)
print(plotClusterScatter)
# iteration
iter <- 5000
burnin <- 0
thin <- 1 
# how many obs update each iteration?
m <- 150
replica <- 1
saveChainRandom <- saveChainAdaptive <- saveChainSys <- matrix(0, ncol = replica, nrow = iter)
exTimeS <- exTimeR <- exTimeA <- c()
for (i in 1:replica) {
  set.seed(floor(runif(1, 1, 100))+replica)
  # Systematic Scan
  resS <- AdaptiveAllocation::ssgd(as.matrix(data[,1:d]), K = 2, hyper = c(1, 1, 0, 1, 1, 1),
                                   method = "EB", iteration = iter, trueAllocation = data[,d+1], thin = thin, burnin = burnin)
  # Random Scan
  resR <- rssg(as.matrix(data[,1:d]), K = 2, m = m, hyper = c(1, 1, 0, 1, 1, 1),
               method = "EB", iteration = iter, trueAllocation = data[,d+1], thin = thin, burnin = burnin)
  # Adaptive Scan
  resA <- AdaptiveAllocation::DiversityGibbs(as.matrix(data[,1:d]), K = 2, m = m, hyper = c(1, 1, 0, 1, 1, 1), gamma = 1,
                                             method = "EB", iteration = iter, diversity = "Gini-Simpson", trueAllocation = data[,d+1], thin = thin, burnin = burnin)
  
  saveChainSys[,i] <- resS$Loss
  saveChainRandom[,i] <- resR$Loss
  saveChainAdaptive[,i] <- resA$Loss
  exTimeS[i] <- resS$Execution_Time
  exTimeR[i] <- resR$Execution_Time
  exTimeA[i] <- resA$Execution_Time
  # entropy plot
  print(AdaptiveAllocation::scattPlot2d(res = resA, data = data, diversity = T))
}

# animated plot with centroid and covariance matrix

# Adaptive Scanner animation
hm <- seq(from = 1, to = iter, by = 100)
for (i in hm) {
  plot(data[,1], data[,2], xlab = "V1", ylab = "V2", main = "Adaptive")
  centroide1 <- resA$Mu[[i]][1,]
  covarianza1 <- resA$Precision[[i]][1,]
  covarianza1 <- diag(1/covarianza1)
  centroide2 <- resA$Mu[[i]][2,]
  covarianza2 <- resA$Precision[[i]][2,]
  covarianza2 <- diag(1/covarianza2)
  ellisse1 <- ellipse(covarianza1, centre=centroide1, level=0.95)
  lines(ellisse1, col="red", lwd = 3) 
  ellisse2 <- ellipse(covarianza2, centre=centroide2, level=0.95)
  lines(ellisse2, col="blue", lwd = 3) 
  Sys.sleep(0.1)
}

# Random Scanner animation
for (i in hm) {
  plot(data[,1], data[,2], xlab = "V1", ylab = "V2", main = "Random")
  centroide1 <- resR$Mu[[i]][1,]
  covarianza1 <- resR$Precision[[i]][1,]
  covarianza1 <- diag(1/covarianza1)
  centroide2 <- resR$Mu[[i]][2,]
  covarianza2 <- resR$Precision[[i]][2,]
  covarianza2 <- diag(1/covarianza2)
  ellisse1 <- ellipse(covarianza1, centre=centroide1, level=0.95)
  lines(ellisse1, col="red", lwd = 3) 
  ellisse2 <- ellipse(covarianza2, centre=centroide2, level=0.95)
  lines(ellisse2, col="blue", lwd = 3) 
  Sys.sleep(0.1)
}

# entropy plot
print(AdaptiveAllocation::scattPlot2d(res = resA, data = data, diversity = T))

# trace plot adjusted rand index
plot(1:iter, saveChainSys[,1], type = "l", ylab = "ARI", ylim = c(0, 1))
lines(1:iter, saveChainRandom[,1], col = "red")
lines(1:iter, saveChainAdaptive[,1], col = "green")
