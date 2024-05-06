#' Gaussian Generating Mechanism
#'
#' @export
genGaussianGM <- function(n, K, d, startM = 1, endM = 4, startP = 1, endP = 2) {
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
  cluster_id <- rep(0:(K-1), each = round(n/K))
  data <- cbind(data, cluster_id)
  return(list(data, matMean, precMat))
}

#' Categorical Generating Mechanism
#'
#' @export
genCategoricalGM <- function(n, K, d, categories = 3, concPar = 0.1) {
  data <- data.frame()
  for (k in 1:K) {
    data_component <- matrix(sample(0:(categories-1), floor(n/K) * d, replace = TRUE, prob = MCMCpack::rdirichlet(1, rep(concPar, categories))), ncol = d)
    data <- rbind(data, data_component)
  }
  labels <- rep(0:(K-1), each = floor(n/K))
  data <- data.frame(Data = data, Component = labels)
  return(data)
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


#' Posterior Summary - Categorical Data
#' 
#' @export

heatMapCat <- function(dataMatrix, zCurr = NULL, zTrue = NULL, categories) {
  if (categories == 2) {
    if (!is.null(zCurr) && !is.null(zTrue)) {
      currentAnnotationRow <- data.frame(
        Cluster = factor(zCurr),
        trueClusters = factor(zTrue)
      )
      rownames(currentAnnotationRow) <- rownames(dataMatrix)
      phmap <- pheatmap(dataMatrix[sort(zCurr, index.return = T)$ix,], 
                        cluster_rows = F, show_rownames = F, show_colnames = F, 
                        color = colorRampPalette(colors = c("white", "black"))(2),
                        annotation_row = currentAnnotationRow)
    } else {
      phmap <- pheatmap(dataMatrix, 
                        cluster_rows = F, show_rownames = F, show_colnames = F, 
                        color = colorRampPalette(colors = c("white", "black"))(2), 
                        format = "d")
    }
  } else {
    if (!is.null(zCurr) && !is.null(zTrue)) {
      currentAnnotationRow <- data.frame(
        Cluster = factor(zCurr),
        trueClusters = factor(zTrue)
      )
      rownames(currentAnnotationRow) <- rownames(dataMatrix)
      phmap <- pheatmap(dataMatrix[sort(zCurr, index.return = T)$ix,], 
                        cluster_rows = F, show_rownames = F, show_colnames = F, 
                        color = brewer.pal(n = categories, name = "Spectral"),
                        annotation_row = currentAnnotationRow)
    } else {
      phmap <- pheatmap(dataMatrix, 
                        cluster_rows = F, show_rownames = F, show_colnames = F, 
                        color = brewer.pal(n = categories, name = "Spectral"), 
                        format = "d")
    }
  }
  
  return(phmap)
}
