#' ssg
#' 
#' @export

##### ------------------------------------------------------------------ ######
ssg <- function(X, hyper = c(1, 1, 0, 1, 1, 1), K = 3, iteration = 1000, burnin = 50, thin = 5, method = "", trueAllocation = numeric(0)) {
  # Hyperparameters description:
  # 1 concPar Dirichlet
  # 2 categorical
  # 3 mean mu prior
  # 4 precision mu prior
  # 5 a0 gamma
  # 6 b0 gamma
  # d-dimensional gaussian data
  res <- SSG(as.matrix(X), as.vector(hyper), as.integer(K),
             as.integer(iteration), as.integer(burnin), as.integer(thin), as.character(method),
             as.vector(trueAllocation))
  return(res)
}

#' rssg
#' 
#' @export

##### ------------------------------------------------------------------ ######
rssg <- function(X, hyper = c(1, 1, 0, 1, 1, 1), K = 3, m = 10, iteration = 1000, burnin = 50, thin = 5, method = "", trueAllocation = numeric(0)) {
  res <- RSSG(as.matrix(X), as.vector(hyper), as.integer(K), as.integer(m),
              as.integer(iteration), as.integer(burnin), as.integer(thin), as.character(method),
              as.vector(trueAllocation))
  return(res)
}


#' Diversity-Guided Gibbs Sampler
#' 
#' @export

##### ------------------------------------------------------------------ ######
DiversityGibbs <- function(X, hyper = c(1, 1, 0, 1, 1, 1), K = 3, m = 10, iteration = 1000, burnin = 50, thin = 5, method = "", gamma = 0.5, trueAllocation = numeric(0), adaptiveGamma = FALSE, q = 1) {
  res <- DiversityGibbsSamp(as.matrix(X), as.vector(hyper), as.integer(K), as.integer(m),
              as.integer(iteration), as.integer(burnin), as.integer(thin), as.character(method),
              as.double(gamma), as.vector(trueAllocation), as.logical(adaptiveGamma),
              as.double(q))
  return(res)
}



#' Categorical Systematic Scan Gibbs
#' 
#' @export

##### ------------------------------------------------------------------ ######
cssg <- function(X, hyper = c(1, 1, 1), K = 3, R = 3, iteration = 1000, burnin = 50, thin = 5, trueAllocation = numeric(0)) {
  res <- CSSG(as.matrix(X), as.vector(hyper), as.integer(K), as.integer(R),
                            as.integer(iteration), as.integer(burnin), as.integer(thin), as.vector(trueAllocation))
  return(res)
}


#' Categorical Random Scan Gibbs
#' 
#' @export

##### ------------------------------------------------------------------ ######
crsg <- function(X, hyper = c(1, 1, 1), K = 3, R = 3, m = 10, iteration = 1000, burnin = 50, thin = 5, trueAllocation = numeric(0)) {
  res <- CRSG(as.matrix(X), as.vector(hyper), as.integer(K), as.integer(R), as.integer(m),
              as.integer(iteration), as.integer(burnin), as.integer(thin), as.vector(trueAllocation))
  return(res)
}


#' Categorical Diversity-Guided Scan Gibbs
#' 
#' @export

##### ------------------------------------------------------------------ ######
cdsg <- function(X, hyper = c(1, 1, 1), K = 3, R = 3, m = 10, iteration = 1000, burnin = 50, thin = 5, gamma = 0.5, trueAllocation = numeric(0), adaptiveGamma = FALSE, q = 1) {
  res <- CDSG(as.matrix(X), as.vector(hyper), as.integer(K), as.integer(R), as.integer(m),
              as.integer(iteration), as.integer(burnin), as.integer(thin), as.double(gamma),
              as.vector(trueAllocation), as.logical(adaptiveGamma), as.integer(q))
  return(res)
}
