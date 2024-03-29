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
scattPlot2d <- function(res = NULL, data, diversity = FALSE) {
  if (diversity == TRUE) {
    div <- apply(res$Diversity, 2, mean)
    dfDiversity <- data.frame(x = data[,1], y = data[,2], diversity = div)
    plotEnt <- ggplot(data = dfDiversity, aes(x = x, y = y, color = diversity)) +
      geom_point(size = 2) +
      scale_color_gradient(low = "#001F3F", high = "#009E73") +
      labs(title = "Two-dimensional Gaussian Finite Mixture", x = "V1", y = "V2")
  } else {
    dfNDiversity <- data.frame(x = data[,1], y = data[,2])
    plotEnt <- ggplot(data = dfNDiversity, aes(x = x, y = y)) +
      geom_point(color = "#001F3F", size = 2) +
      labs(title = "Two-dimensional Gaussian Finite Mixture", x = "V1", y = "V2")
  }
  return(plotEnt)
}


#' Posterior Summary - Bayesian nonparametric mixture inconsistency for the number of components: 
#' How worried should we be in practice? - by Paul Kirk page 7.
#' 
#' @export

psm <- function(z) {
  sumPost <- summary_Posterior(as.matrix(z))
  hm <- heatmap(sumPost, 
          col = colorRampPalette(c("#001F3F", "#009E73"))(100),  # Define color gradient
          # main = "Posterior Similarity Matrix",
          scale = "none",  # Turn off scaling for values
          symm = TRUE)     # Ensure the matrix is symmetric
  return(hm)
}

