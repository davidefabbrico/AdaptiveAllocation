#' ssg
#' 
#' @export

##### ------------------------------------------------------------------ ######
ssg <- function(x, hyper = c(0.5, 1, 0, 1, 2, 2), K = 3, iteration = 1000, burnin = 50, thin = 5) {
  res <- SSG(as.vector(x), as.vector(hyper), as.integer(K),
             as.integer(iteration), as.integer(burnin), as.integer(thin))
  return(res)
}