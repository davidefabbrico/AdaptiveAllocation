#' ssg
#' 
#' @export

##### ------------------------------------------------------------------ ######
ssg <- function(X, hyper = c(1, 1, 0, 1, 1, 1), K = 3, iteration = 1000, burnin = 50, thin = 5, method = "", trueAll = FALSE) {
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
             as.logical(trueAll))
  return(res)
}

#' rssg
#' 
#' @export

##### ------------------------------------------------------------------ ######
rssg <- function(X, hyper = c(1, 1, 0, 1, 1, 1), K = 3, m = 10, iteration = 1000, burnin = 50, thin = 5, method = "") {
  res <- RSSG(as.matrix(X), as.vector(hyper), as.integer(K), as.integer(m),
              as.integer(iteration), as.integer(burnin), as.integer(thin), as.character(method))
  return(res)
}


#' Diversity-Guided Gibbs Sampler
#' 
#' @export

##### ------------------------------------------------------------------ ######
AdaptRSG <- function(X, hyper = c(1, 1, 0, 1, 1, 1), K = 3, m = 10, 
                           iteration = 1000, burnin = 50, iterTuning = 50, thin = 5, 
                           updateProbAllocation = 1, method = "", gamma = 0.5, q = 1, 
                           lambda = 1, kWeibull = 1, alphaPareto = 1, xmPareto = 0.5,
                           DiversityIndex = "Exponential", adaptive = FALSE, nSD = 1, lambda0 = 40,
                     zeta = 0.996, a = 100, w_fun = "hyperbolic") {
  res <- DiversityGibbsSamp(as.matrix(X), as.vector(hyper), as.integer(K), as.integer(m),
                            as.integer(iteration), as.integer(burnin), as.integer(iterTuning), as.integer(thin), 
                            as.integer(updateProbAllocation), as.character(method),
                            as.double(gamma), as.double(q), as.double(lambda), as.double(kWeibull),
                            as.double(alphaPareto), as.double(xmPareto),
                            as.character(DiversityIndex), as.logical(adaptive), as.double(nSD), as.double(lambda0),
                            as.double(zeta), as.double(a), as.character(w_fun))
  return(res)
}



#' Categorical Systematic Scan Gibbs
#' 
#' @export

##### ------------------------------------------------------------------ ######
cssg <- function(X, hyper = c(1, 1, 1), K = 3, R = 3, iteration = 1000, burnin = 50, thin = 5) {
  res <- CSSG(as.matrix(X), as.vector(hyper), as.integer(K), as.integer(R),
                            as.integer(iteration), as.integer(burnin), as.integer(thin))
  return(res)
}


#' Categorical Random Scan Gibbs
#' 
#' @export

##### ------------------------------------------------------------------ ######
crsg <- function(X, hyper = c(1, 1, 1), K = 3, R = 3, m = 10, iteration = 1000, burnin = 50, thin = 5) {
  res <- CRSG(as.matrix(X), as.vector(hyper), as.integer(K), as.integer(R), as.integer(m),
              as.integer(iteration), as.integer(burnin), as.integer(thin))
  return(res)
}


#' Categorical Diversity-Guided Scan Gibbs
#' 
#' @export

##### ------------------------------------------------------------------ ######
AdaptCRSG <- function(X, hyper = c(1, 1, 1), K = 3, R = 3, m = 10, iteration = 1000, burnin = 50, iterTuning = 50, thin = 5, 
                      updateProbAllocation = 1, method = "", gamma = 0.5, q = 1, 
                      lambda = 1, kWeibull = 1, alphaPareto = 1, xmPareto = 0.5,
                      DiversityIndex = "Exponential", adaptive = FALSE, nSD = 1, lambda0 = 40,
                      zeta = 0.996, a = 100) {
  res <- CDSG(as.matrix(X), as.vector(hyper), as.integer(K), as.integer(R), as.integer(m),
              as.integer(iteration), as.integer(burnin), as.integer(iterTuning), as.integer(thin), 
              as.integer(updateProbAlloc), as.character(method), as.double(gamma), as.integer(q),
              as.double(lambda), as.double(kWeibull), as.double(alphaPareto), as.double(xmPareto),
              as.character(DiversityIndex), as.logical(adaptive), as.double(nSD), as.double(lambda0),
              as.double(zeta), as.double(a))
  return(res)
}
