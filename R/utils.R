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
  return(list(data, matMean, listPrec))
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

# #' @export
# model_comparison <- function(data, K, iteration, burnin, thin, replica, m, trueClusters) {
#   n <- dim(data)[1]
#   nout <- (iteration - burnin)/thin
#   saveRandomChain <- saveMyChain <- saveSysChain <- matrix(0, ncol = replica, nrow = nout)
#   myTime <- randomTime <- sysTime <- matrix(0, nrow = nout, ncol = replica)
#   parLambda <- matrix(0, nrow = nout, ncol = replica)
#   meanProb <- matrix(0, nrow = nout, ncol = replica)
#   Convergence <- Check <- divMean <- matrix(0, nrow = dim(allData)[1], ncol = replica)
#   saveAllocAdapt <- saveAllocSys <- saveAllocRandom <- list()
#   
#   # TRUE SSG
#   trueSSG_Chain <- ssg(data, K = K, iteration = iter, method = "EB", hyper = c(1, 1, 0, 1, 1, 1), burnin = 0, thin = thin,
#                        seed = 1, trueAll = trueClusters, trueParameters = T)
#   SysARI_TRUE <- rep(0, nout)
#   for (i in 1:nout) {
#     SysARI_TRUE[i] <- mclust::adjustedRandIndex(trueSSG_Chain$Allocation[i,], trueClusters)
#   }
#   
#   for (repl in 1:replica) {
#     set.seed(repl)
#     SysRes <- ssg(data, K = K, iteration = iter, method = "EB", hyper = c(1, 1, 0, 1, 1, 1), burnin = 0, thin = thin,
#                   trueParameters = F, seed = repl)
#     RandomRes <- rssg(data, K = K, m = m, iteration = iter,
#                       burnin = 0, thin = thin, method = "EB",
#                       hyper = c(1, 1, 0, 1, 1, 1), seed = repl)
#     myRes <- AdaptRSG(data, K = K, m = m, iteration = iter,
#                       burnin = burnin, thin = thin, q = 1,
#                       method = "EB", hyper = c(1, 1, 0, 1, 1, 1), adaptive = T,
#                       DiversityIndex = "Exponential", lambda = 2, updateProbAllocation = 0,
#                       lambda0 = 0, L = 0, c = 1, max_lambda = 100,, a = 1, nSD = 1.96, w_fun = "hyperbolic", 
#                       sp = ceiling((n/m)*(K-1) + nSD*sqrt((n/m)*K*(K-1))), seed = repl)
#     # compute the ARIs
#     myARI <- rep(0, nout)
#     SysARI <- rep(0, nout)
#     RandomARI <- rep(0, nout)
#     for (i in 1:nout) {
#       myARI[i] <- mclust::adjustedRandIndex(myRes$Allocation[i,], trueClusters)
#       SysARI[i] <- mclust::adjustedRandIndex(SysRes$Allocation[i,], trueClusters)
#       RandomARI[i] <- mclust::adjustedRandIndex(RandomRes$Allocation[i,], trueClusters)
#     }
#     saveMyChain[, repl] <- myARI
#     saveSysChain[, repl] <- SysARI
#     saveRandomChain[, repl] <- RandomARI
#     
#     # TIME
#     myTime[,repl] <- myRes$Execution_Time
#     randomTime[,repl] <- RandomRes$Execution_Time
#     sysTime[,repl] <- SysRes$Execution_Time
#     
#     # ALLOCATION 
#     saveAllocAdapt[[repl]] <- myRes$Allocation
#     saveAllocSys[[repl]] <- SysRes$Allocation
#     saveAllocRandom[[repl]] <- RandomRes$Allocation
#   }
#   
#   # RETURN OBJECTS
#   listOne <- list(
#     saveSysChain = saveSysChain,
#     saveRandomChain = saveRandomChain,
#     saveMyChain = saveMyChain,
#     sysTime = sysTime,
#     randomTime = randomTime,
#     myTime = myTime,
#     saveAllocSys = saveAllocSys,
#     saveAllocRandom = saveAllocRandom,
#     saveAllocAdapt = saveAllocAdapt,
#     trueSSG = tail(SysARI_TRUE)
#   )
#   return(listOne)
# }


#' @export
generate_plots <- function(
    saveSysChain, 
    saveRandomChain,
    saveMyChain, 
    sysTime, 
    randomTime,
    lastSSG, # mean of the last 500 iterations of SSG across the replicas
    myTime, 
    replica, 
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
  
  # meanMSysLoss <- mean(meanASysLoss[(nout/2):nout])
  
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
  
  lastSSG_mean <- mean(lastSSG)
  # ARI Plot ----------------------------------------------------
  ari_plot <- ggplot(df_long, aes(x = iteration, y = mean, color = Method)) +
    geom_hline(yintercept = lastSSG_mean, color = "#001F3F", linewidth = 1, linetype = "dashed") +
    geom_vline(xintercept = ceiling((n/m)*(K-1) + nSD*sqrt((n/m)*K*(K-1))), 
               color = "#CC79A7", linetype = "dotted", linewidth = 0.8) +
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
    theme_minimal() +
    theme(
      legend.position = "top",
      legend.title = element_blank(),
      # axis.title = element_text(size = 12),
      # axis.text = element_text(size = 10)
    )
  
  # Time To Convergence ------------------------------------------
  window_size <- 500
  
  nIterMyLoss <- nIterRandLoss <- nIterSysLoss <- rep(nout, replica)
  boolSSG <- boolRSG <- boolARSG <- F
  for (repl in 1:replica) {
    for (i in 1:nout) {
      if ((i + window_size) >= nout) {
        break
      }
      window_ARSG <- saveMyChain[i:(i+window_size), repl]
      window_RSG <- saveRandomChain[i:(i+window_size), repl]
      window_SSG <- saveSysChain[i:(i+window_size), repl]
      mean_ARSG <- mean(window_ARSG)
      mean_RSG <- mean(window_RSG)
      mean_SSG <- mean(window_SSG)
      var_ARSG <- var(window_ARSG)
      var_RSG <- var(window_RSG)
      # last_SSG <- tail(window_SSG, round(length(window_SSG) * 0.1))
      last_SSG <- lastSSG
      var_SSG <- var(last_SSG)
      staT_SS <- (mean_SSG - mean(last_SSG))/sqrt(var_SSG/length(window_SSG) + var_SSG/length(last_SSG))
      staT_AS <- (mean_ARSG - mean(last_SSG))/sqrt(var_ARSG/length(window_ARSG) + var_SSG/length(last_SSG))
      staT_RS <- (mean_RSG - mean(last_SSG))/sqrt(var_RSG/length(window_RSG) + var_SSG/length(last_SSG))
      if ((abs(staT_SS) <= 1.96) & !boolSSG) {
        nIterSysLoss[repl] <- i
        boolSSG <- T
      }
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
    geom_boxplot(width = 0.6) +
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


#' @export
proportion_plot <- function(saveAlloc, K, replica, nreplica, nout) {
  listPlot <- list()
  for (repl in 1:nreplica) {
    matPlot <- matrix(0, nrow = nout, ncol = K)
    for (i in 1:nout) {
      occ <- as.data.frame(table(saveAlloc[[repl]][i,]))
      keys <- occ$Var1
      values <- occ$Freq
      complete_keys <- seq(0, K-1)
      complete_values <- rep(0, length(complete_keys))
      complete_values[match(keys, complete_keys)] <- values
      prop <- complete_values / length(saveAlloc[[repl]][i,])
      matPlot[i, ] <- as.vector(prop)
    }
    listPlot[[repl]] <- matPlot
  }
  array_of_matrices <- simplify2array(listPlot)
  mean_Prop <- apply(array_of_matrices, c(1, 2), mean)
  quantile_5_Prop <- apply(array_of_matrices, c(1, 2), quantile, probs = 0.05)
  quantile_95_Prop <- apply(array_of_matrices, c(1, 2), quantile, probs = 0.95)
  maxone <- max(quantile_95_Prop)
  matPlotOrder <- t(apply(listPlot[[replica]], 1, function(x) sort(x, decreasing = TRUE)))
  df <- data.frame(Iterations = 1:nout, matPlotOrder)
  df_long <- pivot_longer(df, cols = -Iterations, names_to = "Cluster", values_to = "Proportion")
  plott <- ggplot(df_long, aes(x = Iterations, y = Proportion, color = Cluster, group = Cluster)) +
    geom_line(linewidth = 1.3) +
    scale_y_continuous(limits = c(0, maxone)) + 
    labs(x = "Iteration", y = "Proportion", title = "") +
    theme_minimal() + theme(legend.position = "none")
  return(plott)
}


#' @export
log_likelihood_observed <- function(data, pi, mu_list, sigma_list) {
  n <- nrow(data)
  K <- length(pi)
  loglik <- 0
  for (i in 1:n) {
    mix_sum <- 0
    for (k in 1:K) {
      mix_sum <- mix_sum + pi[k] * mvtnorm::dmvnorm(data[i, ], mean = mu_list[k,], sigma = diag(sigma_list[k,]))
    }
    loglik <- loglik + log(mix_sum + .Machine$double.eps)
  }
  return(loglik)
}

#' @export
log_likelihood_complete <- function(data, z, pi, mu_list, sigma_list) {
  n <- nrow(data)
  loglik <- 0
  for (i in 1:n) {
    k <- z[i] + 1 # z is 0-indexed, so we add 1 to match the list indexing
    dens <- mvtnorm::dmvnorm(data[i, ], mean = mu_list[k,], sigma = diag(sigma_list[k,]))
    loglik <- loglik + log(pi[k] + .Machine$double.eps) + log(dens + .Machine$double.eps)
  }
  
  return(loglik)
}

#' @export
single_chain_likelihood_plot <- function(chain) {
  # Crea un data frame long per ggplot
  df <- data.frame(
    Iteration = seq_along(chain$Complete_Likelihood),
    Complete = chain$Complete_Likelihood,
    Observed = chain$Observed_Likelihood
  ) %>%
    pivot_longer(cols = c("Complete", "Observed"), names_to = "Type", values_to = "Likelihood")
  
  # Plot con ggplot
  ggplot(df, aes(x = Iteration, y = Likelihood, color = Type)) +
    geom_line(linewidth = 1) +
    labs(title = "Likelihood Trace Plot",
         x = "Iteration",
         y = "Log-Likelihood",
         color = "Type") +
    scale_color_manual(values = c("Complete" = "blue", "Observed" = "red")) +
    theme_minimal(base_size = 14)
}


likelihood_plot <- function(SSG_mat, RSG_mat, DIG_mat, breaks_x = 500,
                            y_title) {
  
  # mean and quantile functions
  summarize_method <- function(mat) {
    df <- data.frame(
      mean = rowMeans(mat, na.rm = TRUE),
      lower = apply(mat, 1, quantile, probs = 0.025, na.rm = TRUE),
      upper = apply(mat, 1, quantile, probs = 0.975, na.rm = TRUE)
    )
    return(df)
  }
  
  # compute the summary statistics for each method
  df_ssg <- summarize_method(SSG_mat)
  df_rsg <- summarize_method(RSG_mat)
  df_dig <- summarize_method(DIG_mat)
  
  df_all <- data.frame(
    Iteration = 1:nrow(df_ssg),
    SSG_mean = df_ssg$mean,
    SSG_lower = df_ssg$lower,
    SSG_upper = df_ssg$upper,
    RSG_mean = df_rsg$mean,
    RSG_lower = df_rsg$lower,
    RSG_upper = df_rsg$upper,
    DIG_mean = df_dig$mean,
    DIG_lower = df_dig$lower,
    DIG_upper = df_dig$upper
  )
  
  df_long <- df_all %>%
    dplyr::select(Iteration,
           SSG_mean, SSG_lower, SSG_upper,
           RSG_mean, RSG_lower, RSG_upper,
           DIG_mean, DIG_lower, DIG_upper) %>%
    pivot_longer(
      cols = -Iteration,
      names_to = c("Method", "Stat"),
      names_pattern = "(SSG|RSG|DIG)_(mean|lower|upper)"
    ) %>%
    pivot_wider(
      names_from = Stat,
      values_from = value
    )
  
  p <- ggplot(df_long, aes(x = Iteration, y = mean, color = Method)) +
    geom_line(linewidth = 1) +
    geom_ribbon(aes(ymin = lower, ymax = upper, fill = Method),
                alpha = 0.2,
                colour = NA,
                linetype = 0,
                show.legend = FALSE) + 
    labs(
      title = "",
      x = "Iteration",
      y = y_title,
      color = "Method"
    ) +
    theme_minimal() +
    theme(
      legend.position = "top",
      legend.title = element_blank(),
    )
  
  return(p)
}

