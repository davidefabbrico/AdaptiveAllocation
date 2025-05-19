#' Gaussian Generating Mechanism
#'
#' @export
genGaussianGM <- function(n, K, d, startM = 1, endM = 4, startP = 1, endP = 2) {
  matMean = matrix(0, nrow = K, ncol = d)
  listPrec = list()
  data <- data.frame()
  for (k in 1:K) {
    matMean[k, ] <- sample(seq(startM, endM, length.out = 10), d, replace = TRUE)
    precMat <- diag(d)
    diag(precMat) <- 1/sample(startP:endP, d, replace = TRUE)
    listPrec[[k]] <- precMat
    cluster <- mvrnorm(round(n/K), matMean[k,]/sqrt(d), listPrec[[k]])
    data <- rbind(data, cluster)
  }
  cluster_id <- rep(0:(K-1), each = round(n/K))
  for (j in 1:dim(data)[2]) {
    data[, j] <- (data[, j]-mean(data[,j]))/sd(data[,j])
  }
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
      geom_point(size = 4) + theme_minimal() +
      scale_color_gradient(low = "#001F3F", high = "#ADD8E6") +
      labs(title = "", x = expression(x), y = expression(y))
  } else {
    dfNDiversity <- data.frame(x = data[,1], y = data[,2])
    plotEnt <- ggplot(data = dfNDiversity, aes(x = x, y = y)) +
      geom_point(color = "#001F3F", size = 4) +
      labs(title = "", x = expression(x), y = expression(y)) + theme_minimal()
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


#' @export
generate_plots <- function(
    saveMyChain, 
    saveSysChain, 
    saveRandomChain,
    sysTime, 
    myTime, 
    randomTime,
    replica, 
    iter, 
    nout
) {
  # Mean -----------------------------------
  meanCMyLoss <- rowMeans(saveMyChain)
  meanASysLoss <- rowMeans(saveSysChain)
  meanBRandomLoss <- rowMeans(saveRandomChain)
  
  # Quantiles
  quantiles <- list(
    DIG = list(
      lb = apply(saveMyChain, 1, quantile, 0.05),
      ub = apply(saveMyChain, 1, quantile, 0.95)
    ),
    RSG = list(
      lb = apply(saveRandomChain, 1, quantile, 0.05),
      ub = apply(saveRandomChain, 1, quantile, 0.95)
    ),
    SSG = list(
      lb = apply(saveSysChain, 1, quantile, 0.05),
      ub = apply(saveSysChain, 1, quantile, 0.95)
    )
  )
  
  # Dati for ARI Plot -----------------------------------------
  df <- data.frame(
    iteration = 1:nout,
    DIG_mean = meanCMyLoss,
    RSG_mean = meanBRandomLoss,
    SSG_mean = meanASysLoss,
    DIG_lb = quantiles$DIG$lb,
    DIG_ub = quantiles$DIG$ub,
    RSG_lb = quantiles$RSG$lb,
    RSG_ub = quantiles$RSG$ub,
    SSG_lb = quantiles$SSG$lb,
    SSG_ub = quantiles$SSG$ub
  )
  
  df_long <- df %>%
    pivot_longer(
      cols = -iteration,
      names_to = c("Method", ".value"),
      names_sep = "_"
    ) %>%
    mutate(Method = factor(Method, levels = c("SSG", "RSG", "DIG")))
  
  # ARI Plot ----------------------------------------------------
  ari_plot <- ggplot(df_long, aes(x = iteration, y = mean, color = Method)) +
    geom_ribbon(
      aes(ymin = lb, ymax = ub, fill = Method),
      alpha = 0.15,
      colour = NA 
    ) +
    geom_line(linewidth = 1) +
    scale_color_manual(
      values = c(
        "SSG" = "#0072B2",
        "RSG" = "#009E73",
        "DIG" = "#D55E00"
      )
    ) +
    scale_fill_manual(
      values = c(
        "SSG" = "#0072B2",
        "RSG" = "#009E73",
        "DIG" = "#D55E00"
      )
    ) +
    labs(x = "Iteration", y = "ARI") +
    ylim(-0.05, 1) +
    theme_classic() +
    theme(
      legend.position = "top",
      legend.title = element_blank(),
      # axis.title = element_text(size = 12),
      # axis.text = element_text(size = 10)
    )
  
  # Time To Convergence ------------------------------------------
  meanMSysLoss <- mean(meanASysLoss[(iter/2):iter]) # mean(tail(meanASysLoss))
  window_size <- 100
  
  nIterMyLoss <- nIterRandLoss <- nIterSysLoss <- rep(iter, replica)
  boolSSG <- boolRSG <- boolARSG <- F
  for (repl in 1:replica) {
    for (i in 1:length(saveMyChain[,repl])) {
      if ((i + window_size) >= length(saveMyChain[,repl])) {
        break
      }
      window_ARSG <- saveMyChain[i:(i+window_size), repl]
      window_RSG <- saveRandomChain[i:(i+window_size), repl]
      window_SSG <- saveSysChain[i:(i+window_size), repl]
      mean_ARSG <- mean(window_ARSG)
      mean_RSG <- mean(window_RSG)
      mean_SSG <- mean(window_SSG)
      if ((mean_SSG <= meanMSysLoss + 0.005 & mean_SSG >= meanMSysLoss - 0.005) & !boolSSG) {
        nIterSysLoss[repl] <- i
        boolSSG <- T
      }
      var_ARSG <- var(window_ARSG)
      var_RSG <- var(window_RSG)
      last_SSG <- tail(window_SSG, round(length(window_SSG) * 0.1))
      var_SSG <- var(last_SSG)
      staT_AS <- (mean_ARSG - mean(last_SSG))/sqrt(var_ARSG/length(window_ARSG) + var_SSG/length(last_SSG))
      staT_RS <- (mean_RSG - mean(last_SSG))/sqrt(var_RSG/length(window_RSG) + var_SSG/length(last_SSG))
      if ((abs(staT_AS) <= 1.96) & !boolARSG) {
        nIterMyLoss[repl] <- i
        boolARSG <- T
      }
      if ((abs(staT_RS) <= 1.96) & !boolRSG) {
        nIterRandLoss[repl] <- i
        boolRSG <- T
      }
    }
    boolSSG <- boolRSG <- boolARSG <- F
  }
  
  MyTimeRepl <- RandomTimeRepl <- SysTimeRepl <- c()
  for (repl in 1:replica) {
    SysTimeRepl[repl] <- sysTime[nIterSysLoss[repl], repl]
    MyTimeRepl[repl] <- myTime[nIterMyLoss[repl], repl]
    RandomTimeRepl[repl] <- randomTime[nIterRandLoss[repl], repl]
  }
  
  # Time Data -----------------------------------------------
  time_data <- data.frame(
    SSG = SysTimeRepl / 1e6,
    RSG = RandomTimeRepl / 1e6,
    DIG = MyTimeRepl / 1e6
  ) %>%
    pivot_longer(
      everything(),
      names_to = "Model",
      values_to = "Time"
    ) %>%
    mutate(Model = factor(Model, levels = c("SSG", "RSG", "DIG")))
  
  # Time Plot ---------------------------------------------------
  time_plot <- ggplot(time_data, aes(x = Model, y = Time, fill = Model)) +
    geom_boxplot(width = 0.6, outlier.shape = NA) +
    scale_fill_manual(values = c("#0072B2", "#009E73", "#D55E00")) +
    labs(y = "Time to Convergence (sec.)", x = "") +
    theme_minimal() +
    theme(
      legend.position = "none",
      legend.title = element_blank(),
      # axis.text.x = element_text(angle = 45, hjust = 1),
      panel.grid.major.x = element_blank()
    )
  
  # 7. Numerical Results ----------------------------------------------------
  print_convergence_results <- function(meanASysLoss, meanBRandomLoss, meanCMyLoss,
                                        SysTimeRepl, RandomTimeRepl, MyTimeRepl) {
    
    results <- data.frame(
      Method = c("SSG", "RSG", "DIG"),
      Time_Mean = c(mean(SysTimeRepl / 1e6), mean(RandomTimeRepl / 1e6), mean(MyTimeRepl / 1e6)),
      Time_SD = c(sd(SysTimeRepl / 1e6), sd(RandomTimeRepl / 1e6), sd(MyTimeRepl / 1e6)),
      ARI_Mean = c(mean(tail(meanASysLoss, 1000)), mean(tail(meanBRandomLoss, 1000)), mean(tail(meanCMyLoss, 1000))),
      ARI_SD = c(sd(tail(meanASysLoss, 1000)), sd(tail(meanBRandomLoss, 1000)), sd(tail(meanCMyLoss, 1000))),
      stringsAsFactors = FALSE
    )
    
    # Codici ANSI per i colori
    red <- "\033[31m"
    green <- "\033[32m"
    yellow <- "\033[33m"
    blue <- "\033[34m"
    silver <- "\033[37m"
    bold <- "\033[1m"
    reset <- "\033[0m"
    
    # Column widths
    col_widths <- c(
      Method = max(nchar(results$Method)) + 1,
      Time_Mean = max(nchar(sprintf("%.3f", results$Time_Mean))) + 2,
      Time_SD = max(nchar(sprintf("%.3f", results$Time_SD))) + 2,
      ARI_Mean = max(nchar(sprintf("%.3f", results$ARI_Mean))) + 2,
      ARI_SD = max(nchar(sprintf("%.3f", results$ARI_SD))) + 2
    )
    
    # Header
    cat("\n")
    cat(paste0(blue, bold, "===================================================\n",
               "               CONVERGENCE RESULTS               \n",
               "===================================================\n", reset))
    
    cat(sprintf("%-7s %10s %10s %10s %10s\n",
                "Method", "Time_Mean", "Time_SD", "ARI_Mean", "ARI_SD"))
    cat(paste0(rep("-", 51), collapse = ""), "\n")
    
    for(i in 1:nrow(results)) {
      method <- results$Method[i]
      
      time_mean <- if(results$Time_Mean[i] == min(results$Time_Mean)) {
        paste0(green, bold, sprintf("%10.4f", results$Time_Mean[i]), reset)
      } else if(results$Time_Mean[i] == max(results$Time_Mean)) {
        paste0(red, bold, sprintf("%10.4f", results$Time_Mean[i]), reset)
      } else {
        paste0(yellow, sprintf("%10.4f", results$Time_Mean[i]), reset)
      }
      
      ari_mean <- if(results$ARI_Mean[i] == max(results$ARI_Mean)) {
        paste0(green, bold, sprintf("%10.4f", results$ARI_Mean[i]), reset)
      } else if(results$ARI_Mean[i] == min(results$ARI_Mean)) {
        paste0(red, bold, sprintf("%10.4f", results$ARI_Mean[i]), reset)
      } else {
        paste0(yellow, sprintf("%10.4f", results$ARI_Mean[i]), reset)
      }
      
      cat(sprintf("%-6s %s %10.4f %s %10.4f\n",
                  method, time_mean, results$Time_SD[i], ari_mean, results$ARI_SD[i]))
    }
    
    cat(paste0(rep("-", 51), collapse = ""), "\n")
  
    
    # Check convergenza
    ssg_loss <- results$ARI_Mean[1]
    thresholds <- abs(results$ARI_Mean - ssg_loss)
    
    cat("\n")
    if(any(thresholds > 0.01)) {
      cat(paste0(red, bold, "⚠️  CONVERGENCE WARNING ⚠️\n", reset))
      cat(paste0(red, "• Some methods didn't converge (threshold > 0.01)\n", reset))
      cat(paste0(red, "• Max difference: ", round(max(thresholds), 4), "\n", reset))
    } else {
      cat(paste0(green, bold, "✓ ALL METHODS CONVERGED ✓\n", reset))
      cat(paste0(green, "• All methods are within convergence threshold\n", reset))
    }
  }
  
  
  print_convergence_results(
    meanASysLoss = meanASysLoss,
    meanBRandomLoss = meanBRandomLoss,
    meanCMyLoss = meanCMyLoss,
    SysTimeRepl = SysTimeRepl,
    RandomTimeRepl = RandomTimeRepl,
    MyTimeRepl = MyTimeRepl
  )
  
  # Output ---------------------------------------------------------
  return(list(
    ari_plot = ari_plot,
    time_plot = time_plot
  ))
}
