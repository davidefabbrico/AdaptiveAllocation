#' ssg
#' 
#' @export

##### ------------------------------------------------------------------ ######
ssg <- function(x, hyper = c(1, 1, 0, 20, 1, 1), K = 3, iteration = 1000, burnin = 50, thin = 5) {
  # unidimensional gaussian data
  res <- SSG(as.vector(x), as.vector(hyper), as.integer(K),
             as.integer(iteration), as.integer(burnin), as.integer(thin))
  return(res)
}

#' dssg
#' 
#' @export

##### ------------------------------------------------------------------ ######
ssgd <- function(X, hyper = c(1, 1, 0, 20, 1, 1), K = 3, iteration = 1000, burnin = 50, thin = 5, method = "EB") {
  # Hyperparameters description:
  # 1 concPar Dirichlet
  # 2 categorical
  # 3 mean mu prior
  # 4 precision mu prior
  # 5 a0 gamma
  # 6 b0 gamma
  # d-dimensional gaussian data
  res <- DSSG(as.matrix(X), as.vector(hyper), as.integer(K),
             as.integer(iteration), as.integer(burnin), as.integer(thin), as.character(method))
  return(res)
}

#' rssg
#' 
#' @export

##### ------------------------------------------------------------------ ######
rssg <- function(X, hyper = c(1, 1, 0, 20, 1, 1), K = 3, m, iteration = 1000, burnin = 50, thin = 5, method = "EB") {
  res <- RSSG(as.matrix(X), as.vector(hyper), as.integer(K), as.integer(m),
              as.integer(iteration), as.integer(burnin), as.integer(thin), as.character(method))
  return(res)
}


#' Entropy-Guided Gibbs Sampler
#' 
#' @export

##### ------------------------------------------------------------------ ######
EntropyGibbs <- function(X, hyper = c(1, 1, 0, 20, 1, 1), K = 3, m, iteration = 1000, burnin = 50, thin = 5, method = "EB", gamma = 0.5) {
  res <- EntropyGibbsSamp(as.matrix(X), as.vector(hyper), as.integer(K), as.integer(m),
              as.integer(iteration), as.integer(burnin), as.integer(thin), as.character(method),
              as.double(gamma))
  return(res)
}