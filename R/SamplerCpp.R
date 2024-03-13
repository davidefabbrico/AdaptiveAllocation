#' ssg
#' 
#' @export

##### ------------------------------------------------------------------ ######
ssg <- function(x, hyper = c(1, 1, 0, 0.1, 1, 1), K = 3, iteration = 1000, burnin = 50, thin = 5) {
  # unidimensional gaussian data
  res <- SSG(as.vector(x), as.vector(hyper), as.integer(K),
             as.integer(iteration), as.integer(burnin), as.integer(thin))
  return(res)
}

#' dssg
#' 
#' @export

##### ------------------------------------------------------------------ ######
ssgd <- function(x, hyper = c(1, 1, 0, 1, 1, 1), K = 3, iteration = 1000, burnin = 50, thin = 5) {
  # d-dimensional gaussian data
  res <- DSSG(as.matrix(X), as.vector(hyper), as.integer(K),
             as.integer(iteration), as.integer(burnin), as.integer(thin))
  return(res)
}