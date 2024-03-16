#' Generating Mechanism
#'
#' @export
genGM <- function(n, K, d, startM = 1, endM = 4, startP = 1, endP = 2) {
  matMean = matrix(0, nrow = K, ncol = d)
  listPrec = list()
  data <- data.frame()
  for (k in 1:K) {
    matMean[k, ] <- sample(startM:endM, d, replace = TRUE)
    precMat <- diag(d)
    diag(precMat) <- 1/sample(startP:endP, d, replace = TRUE)
    listPrec[[k]] <- precMat
    cluster <- mvrnorm(round(n/K), matMean[k,], listPrec[[k]])
    data <- rbind(data, cluster)
  }
  return(list(data, matMean, precMat))
}

#' 2d Contour Plot
#'
#' @export
contPlot2d <- function(data) {
  if (dim(data)[2] != 2) {
    return("Look at the Dimensions of your Data!")
  } else {
    df <- data.frame(x = data[, 1], y = data[, 2])
    plotGen <- ggplot(df, aes(x, y)) +
      geom_density_2d() +
      labs(title = "Contour Plot of Probability Densities")
    return(plotGen)
  }
}


#' Scatter Plot + entropy
#'
#' @export
scattPlot2d <- function(res, data, entropy) {
  if (entropy == TRUE) {
    ent <- apply(res$Entropy, 2, mean)
    plotEnt <- ggplot(data = as.data.frame(data), aes(x = V1, y = V2, color = ent)) +
      geom_point(size = 2) +
      scale_color_gradient(low = "blue", high = "red") +
      labs(title = "Two-dimensional Gaussian Finite Mixture", x = "V1", y = "V2")
  } else {
    plotEnt <- ggplot(data = as.data.frame(data), aes(x = V1, y = V2)) +
      geom_point(color = "blue", size = 2) +
      labs(title = "Two-dimensional Gaussian Finite Mixture", x = "V1", y = "V2")
  }
  return(plotEnt)
}

