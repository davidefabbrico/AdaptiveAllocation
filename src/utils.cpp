#define ARMA_64BIT_WORD
#include <RcppArmadillo.h>
#include <math.h>
#include <R.h>
#include <Rmath.h>
#include <random>
#include <stdlib.h> 
#include <limits.h>
#include <unordered_set>
#include <chrono>
#include <algorithm>
#include <RcppArmadilloExtensions/sample.h>


#define ANSI_RESET       "\033[0m"
#define ANSI_BOLD        "\033[1m"
#define ANSI_WHITE       "\033[97m"
#define ANSI_RED         "\033[31m"
#define ANSI_GREEN       "\033[32m"
#define ANSI_MAGENTA     "\033[35m"
#define ANSI_YELLOW      "\033[33m"
#define ANSI_BG_GREEN    "\033[42m"
#define ANSI_BG_YELLOW   "\033[43m"
#define ANSI_BG_CYAN     "\033[46m"
#define ANSI_BG_BLUE    "\033[44m"
#define ANSI_BRIGHT_RED  "\033[91m"
#define ANSI_BRIGHT_GREEN "\033[92m"

using namespace Rcpp;
using namespace arma;
using namespace std;

// Modello:
// xi | zi, mu, sigma^2 è una normale , da cui R::rnorm(0, 1)
// zi | pi1, ..., piK è una Multinomiale, da cui non sappiamo ancora campionare
// pi1, ..., piK segue una distribuzione di Dirichlet, da cui non sappiamo ancora campionare
// mu segue una Normale con iperparametri mu0 e sigma0^2
// sigma^2 segue una inverse gamma 1/callrgamma

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::mat customMatrix(arma::irowvec z) {
  int n = z.n_elem;
  arma::mat A = arma::zeros<arma::mat>(n, n);
  for (int i = 0; i<n; i++) {
    for (int j = 0; j<n; j++) {
      if (z(i) == z(j)) {
        A(i, j) = 1;
      }
    }
  }
  return(A);
}

// [[Rcpp::export]]
double tanup(int t, double s, double a) {
  double result = 0.5 * (1 + tanh((t - s) / a));
  return result;
}

// [[Rcpp::export]]
double tanlo(int t, double s, double a) {
  double result = 0.5 * (1 - tanh((t - s) / a));
  return result;
}

// [[Rcpp::export]]
double expup(int t, double s) {
  double result = exp(t)/(s+exp(t));
  return result;
}

// [[Rcpp::export]]
double explo(int t, double s) {
  double result = s/(s+exp(t));
  return result;
}


// come campionare da una distribuzione Gamma
// [[Rcpp::export]]
Rcpp::NumericVector callrgamma(int n, double shape, double scale) { 
  // n sono i campioni che voglio
  // shape e scale sono i parametri della gamma
  return(rgamma(n, shape, scale)); 
}

// [[Rcpp::export]]
NumericVector csample_num( NumericVector x,
                           int size,
                           bool replace,
                           NumericVector prob) {
  NumericVector ret = Rcpp::sample(x, size, replace, prob);
  return ret;
}

// [[Rcpp::export]]
NumericVector csample_num_new(NumericVector x,
                                 int size,
                                 Nullable<NumericVector> prob = R_NilValue) {
  int n = x.size();
  
  if (size > n) {
    stop("Cannot sample more elements than available without replacement.");
  }
  
  NumericVector result(size);
  
  // Generatore random per std::shuffle e R::runif
  std::random_device rd;
  std::mt19937 gen(rd());
  
  // Caso: pesi uniformi
  if (prob.isNull()) {
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    for (int i = 0; i < size; ++i) {
      result[i] = x[indices[i]];
    }
    return result;
  }
  
  // Caso: pesi non uniformi (sampling senza rimpiazzo con metodo di Efraimidis–Spirakis)
  NumericVector prob_val(prob);
  if (prob_val.size() != n) {
    stop("Length of 'x' and 'prob' must match.");
  }
  
  std::vector<std::pair<double, int>> heap;
  
  for (int i = 0; i < n; ++i) {
    if (prob_val[i] <= 0.0) continue;
    double u = R::runif(0.0, 1.0); // usa R per coerenza con Rcpp
    double key = std::pow(u, 1.0 / prob_val[i]);
    heap.emplace_back(key, i);
  }
  
  if (static_cast<int>(heap.size()) < size) {
    stop("Not enough positive weights to sample requested number of elements.");
  }
  
  std::partial_sort(heap.begin(), heap.begin() + size, heap.end(),
                    [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                      return a.first > b.first;
                    });
  
  for (int i = 0; i < size; ++i) {
    result[i] = x[heap[i].second];
  }
  
  return result;
}


// [[Rcpp::export]]
int diracF(int a, int b){
  if (a == b) {
    return 1;
  } else {
    return 0;
  }
}

// [[Rcpp::export]]
arma::mat summary_Posterior(arma::imat z) {
  int N = z.n_cols;
  int M = z.n_rows;
  arma::mat sumPost = arma::zeros<arma::mat>(N, N);
  for (int i = 0; i<N; i++) {
    for (int j = 0; j<N; j++) {
      for (int m = 0; m<M; m++) {
        sumPost(i, j) = sumPost(i, j) + diracF(z(m, i), z(m, j));
      }
    }
  }
  sumPost = sumPost/M;
  return(sumPost);
}

// [[Rcpp::export]]
arma::mat rdirichlet_cpp(int num_samples, arma::vec alpha_m) {
  int distribution_size = alpha_m.n_elem;
  // each row will be a draw from a Dirichlet
  arma::mat distribution = arma::zeros(num_samples, distribution_size);
  
  for (int i = 0; i < num_samples; ++i) {
    double sum_term = 0;
    // loop through the distribution and draw Gamma variables
    for (int j = 0; j < distribution_size; ++j) {
      double cur = callrgamma(1, alpha_m[j], 1.0)(0);
      distribution(i,j) = cur;
      sum_term = sum_term + cur;
    }
    // now normalize
    for (int j = 0; j < distribution_size; ++j) {
      distribution(i,j) = distribution(i,j)/sum_term;
    }
  }
  return(distribution);
}

////////////////////////////////////////////////////
////////////////// Loss Function ///////////////////
///////////////////////////////////////////////////

// [[Rcpp::export]]
double BinderLoss(arma::irowvec eAlloc, arma::irowvec tAlloc) {
  int n = eAlloc.n_elem;
  double BL;
  for (int i = 0; i<n; i++) {
    for (int j = (i+1); j<n; j++) {
      BL = BL + diracF(tAlloc(i), tAlloc(j))*(1-diracF(eAlloc(i), eAlloc(j)))+ (1-diracF(tAlloc(i), tAlloc(j)))*diracF(eAlloc(i), eAlloc(j));
    }
  }
  return(BL);
} 

int coefBinom(int n, int k) {
  if (k > n)
    return 0;
  if (k == 0 || k == n)
    return 1;
  return coefBinom(n - 1, k - 1) + coefBinom(n - 1, k);
}

// [[Rcpp::export]]
double ari(arma::irowvec eAlloc, arma::irowvec tAlloc) {
  int n = tAlloc.n_elem; 
  arma::irowvec nx = arma::unique(tAlloc);
  int nE = nx.n_elem;
  arma::mat cT(nE, nE);
  for (int i = 0; i<n; i++) {
    cT(tAlloc(i), eAlloc(i)) += 1;
  }
  arma::vec a = sum(cT, 1); // Riga
  arma::rowvec b = sum(cT, 0); // Colonna
  // NUMERATOR
  double numPart1 = 0;
  for (int i = 0; i<nE; i++) {
    for (int j = 0; j<nE; j++) {
      numPart1 += coefBinom(cT(i, j), 2);
    }
  }
  double numP2 = 0;
  for (int i = 0; i<nE; i++) {
    numP2 += coefBinom(a(i), 2);
  }
  // cout << "numP2 " << numP2 << "\n";
  double numP3 = 0;
  for (int i = 0; i<nE; i++) {
    numP3 += coefBinom(b(i), 2);
    // cout << coefBinom(b(i), 2) << "\n";
  }
  // cout << "numP3 " << numP3 << "\n";
  double denden = coefBinom(n, 2);
  double numPart2 = (numP2*numP3)/denden;
  double num = numPart1 - numPart2;
  // Denominator
  double denPart1 = (numP2 + numP3)/2;
  double denPart2 = (numP2*numP3)/denden;
  double den = denPart1 - denPart2;
  return(num/den);
}

// [[Rcpp::export]]
double myProduct(arma::vec a) {
  double prod = 1;
  int n = a.n_elem;
  for (int i = 0; i<n; i++) {
    prod = prod*a(i);
  }
  return prod;
}

// [[Rcpp::export]]
double mySum(arma::vec a) {
  double somma = 0;
  int n = a.n_elem;
  for (int i = 0; i<n; i++) {
    somma = somma + a(i);
  }
  return somma;
}



// Log-density multivariata con precisione diagonale
double log_mvnorm_diag_precision(const arma::rowvec& x, const arma::rowvec& mu, const arma::rowvec& diag_prec) {
  arma::rowvec diff = x - mu;
  arma::rowvec quad = arma::square(diff) % diag_prec;
  double quad_form = arma::accu(quad);
  double log_det = arma::sum(arma::log(diag_prec + std::numeric_limits<double>::epsilon()));
  double d = x.n_elem;
  double log_density = 0.5 * (log_det - d * std::log(2 * M_PI) - quad_form);
  return log_density;
}

// [[Rcpp::export]]
double log_likelihood_observed(const arma::mat& data,
                               const arma::rowvec& pi,
                               const arma::mat& mu_list,
                               const arma::mat& diag_precision_list) {
  int n = data.n_rows;
  int K = pi.n_elem;
  double loglik = 0.0;
   
  for (int i = 0; i < n; ++i) {
    double mix_sum = 0.0;
    for (int k = 0; k < K; ++k) {
      double log_dens = log_mvnorm_diag_precision(data.row(i), mu_list.row(k), diag_precision_list.row(k));
      mix_sum += pi[k] * std::exp(log_dens);
    } 
    loglik += std::log(mix_sum + std::numeric_limits<double>::epsilon());
  }
   
  return loglik;
} 

// [[Rcpp::export]]
double log_likelihood_complete(const arma::mat& data,
                               const arma::uvec& z,
                               const arma::rowvec& pi,
                               const arma::mat& mu_list,
                               const arma::mat& diag_precision_list) {
  int n = data.n_rows;
  double loglik = 0.0;
  
  for (int i = 0; i < n; ++i) {
    int k = z[i];  // 0-based index
    double log_dens = log_mvnorm_diag_precision(data.row(i), mu_list.row(k), diag_precision_list.row(k));
    loglik += std::log(pi[k] + std::numeric_limits<double>::epsilon()) + log_dens;
  } 
  
  return loglik;
} 

////////////////////////////////////////////////////
//////////////// D-Dimensional Data ////////////////
///////////////////////////////////////////////////

// [[Rcpp::export]]
List SSG(arma::mat X, arma::vec hyper, int K, int iteration, int burnin, 
         int thin, String method, bool trueParameters, arma::irowvec trueAll, 
         arma::mat trueMean, arma::mat truePrec, arma::rowvec truePerc, 
         int seed, bool pb, bool likelihood, bool onlyComp) {
  // All the seed
  arma::arma_rng::set_seed(seed);
  Rcpp::Environment base_env = Rcpp::Environment::namespace_env("base");
  Rcpp::Function set_seed_r = base_env["set.seed"];
  set_seed_r(seed);
  std::srand(seed);
  // precision and not variance!!
  ////////////////////////////////////////////////////
  ////////////////// Initial settings ///////////////
  ///////////////////////////////////////////////////
  int nout = (iteration-burnin)/thin; // number of sample from the posterior
  int n = X.n_rows; // number of rows
  int d = X.n_cols; // number of columns
  int idx = 0;
  // Index for the cluster
  NumericVector indC(K);
  for (int k = 0; k<K; k++) {
    indC(k) = k;
  }
  // Z
  arma::imat Z(nout, n);
  // Probability allocation
  List PROB(nout);
  // TIME
  NumericVector TIME(nout);
  // PI
  arma::mat PI(nout, K);
  // MU
  List MU(nout);
  // SIGMA
  List PREC(nout);
  // OBSERVED LIKELIHOOD
  NumericVector OBS_LIK(nout);
  // COMPLETE LIKELIHOOD
  NumericVector COMP_LIK(nout);
  double obs_likelihood = 0.0;
  double comp_likelihood = 0.0;
  ////////////////////////////////////////////////////
  ////////// Emprical bayes prior settings ///////////
  ///////////////////////////////////////////////////
  arma::rowvec hyper_mu_mean(d);
  double hyper_prec_b;
  double hyper_prec_a;
  arma::mat hyper_mu_prec(K, d);
  if (method == "EB") {
    hyper_mu_mean = mean(X, 0);
    hyper_prec_a = d + 2;
    hyper_prec_b = sum(var(X, 0)/d)/(pow(K, 2/d));
  } else {
    cout << "Remember to specify the hyperparameter values!" << "\n";
    for (int j = 0; j<d; j++) {
      hyper_mu_mean(j) = hyper(2);
    } 
    hyper_prec_b = hyper(5);
    for (int k = 0; k<K; k++) {
      hyper_mu_prec.row(k) = hyper(3)*arma::ones<arma::rowvec>(d);
    }
    hyper_prec_a = hyper(4);
  }
  ////////////////////////////////////////////////////
  ////////////////// Initial value //////////////////
  ///////////////////////////////////////////////////
  // Inizialize the vector allocation
  arma::irowvec z(n);
  // Iniziallize the mean matrix
  arma::mat mu(K, d);
  // Inizialize the precision matrix
  arma::mat prec(K, d);
  // Inizialize the percentage vector
  arma::rowvec pi(K);
  arma::vec concPar = (hyper(1)/K) * arma::ones<arma::vec>(K);
  if (!trueParameters) {
    // ALLOCATION 
    NumericVector probC(K);
    for (int k = 0; k<K; k++) {
      probC(k) = hyper(0)/K;
    }
    for (int i = 0; i<n; i++) {
      // z(i) = csample_num(indC, 1, true, probC)(0);
      z(i) = csample_num_new(indC, 1, probC)(0);
    }
    // PI
    pi = rdirichlet_cpp(1, concPar);
    // PRECISION
    for (int j = 0; j<d; j++) {
      for (int k = 0; k<K; k++) {
        prec(k,j) = callrgamma(1, hyper_prec_a, 1.0/hyper_prec_b)(0);
      }
    }
    // MU
    if (method == "EB") {
      double kP = 0.01;
      for (int j = 0; j<d; j++) {
        for (int k = 0; k<K; k++) {
          hyper_mu_prec(k,j) = prec(k,j);
          mu(k,j) = R::rnorm(hyper_mu_mean(j), 1.0/sqrt(kP*hyper_mu_prec(k,j)));
        } 
      } 
    } else {
      for (int j = 0; j<d; j++) {
        for (int k = 0; k<K; k++) {
          mu(k,j) = R::rnorm(hyper_mu_mean(j), 1.0/sqrt(hyper_mu_prec(k,j)));
        }
      } 
    }
  } else {
    cout << "True parameters are used!" << "\n";
    // true allocation
    z = trueAll;
    // true mean
    mu = trueMean;
    // true precision
    prec = truePrec;
    // true percentage
    pi = truePerc;
  }
  // Probability Matrix
  NumericMatrix probAllocation(n, K);
  ////////////////////////////////////////////////////
  /////////////////// Main Part /////////////////////
  ///////////////////////////////////////////////////
  double durationOld;
  // Progress bar setup
  int barWidth = 30;
  double progress = 0.0;
  for (int t = 0; t<iteration; t++) {
    // start time
    auto start = std::chrono::high_resolution_clock::now();
    // update probability
    // STEP 1. Compute the probability
    for (int i = 0; i<n; i++) {
      for (int k = 0; k<K; k++) {
        arma::vec vecTmp(d);
        for (int j = 0; j<d; j++) {
          vecTmp(j) = R::dnorm(X(i,j), mu(k,j), sqrt(1.0/prec(k,j)), true);
        }
        probAllocation(i,k) = exp(log(pi(k)) + mySum(vecTmp)) + std::numeric_limits<double>::denorm_min();;
      } 
    }
    // Normalize the rows
    arma::vec rSum = rowSums(probAllocation);
    for (int i = 0; i<n; i++) {
      probAllocation(i, _) = probAllocation(i, _) / rSum(i);
    }
    // update z
    for (int i = 0; i<n; i++) {
      NumericVector prob_r = as<NumericVector>(wrap(probAllocation(i, _)));
      z(i) = csample_num_new(indC, 1, prob_r)(0);
      // z(i) = csample_num(indC, 1, false, probAllocation(i, _))(0);
    }
    // compute N
    arma::irowvec N(K);
    for (int i = 0; i < n; i++) {
      N[z[i]] = N[z[i]]+1;
    }
    // update pi
    arma::rowvec parDirich = concPar.t() + N;
    arma::vec parDirichR = parDirich.t();
    pi = rdirichlet_cpp(1, parDirichR);
    // update params
    // mu
    double muPost;
    double precPost;
    arma::mat sampMean(K, d);
    // Prior Setting (tutti i cluster partono con la stessa hyperpriori per la media)
    for (int k = 0; k<K; k++) {
      for (int j = 0; j<d; j++) {
        for (int i = 0; i<n; i++) {
          sampMean(k,j) = sampMean(k,j) + diracF(z(i), k)*X(i, j);
        }
      } 
    } 
    
    for (int j = 0; j<d; j++) {
      for (int k = 0; k<K; k++) {
        muPost = (prec(k,j)*sampMean(k,j)+hyper(2)*hyper(3))/(prec(k,j)*N(k)+hyper(3));
        precPost = prec(k,j)*N(k)+hyper(3);
        mu(k,j) = R::rnorm(muPost, sqrt(1.0/precPost));
      } 
    } 
    // precision
    arma::mat sumNum(K, d);
    for (int k = 0; k<K; k++) {
      for (int j = 0; j<d; j++) {
        for (int i = 0; i<n; i++) {
          sumNum(k,j) = sumNum(k,j) + diracF(z(i), k)*pow(X(i,j)-mu(k,j), 2);
        }
      }
    }
    for (int j = 0; j<d; j++) {
      for (int k = 0; k<K; k++) {
        double shape = hyper(4)+(N(k)/2.0);
        double scale = hyper(5)+(sumNum(k,j)/2.0);
        prec(k,j) = callrgamma(1, shape, 1.0/scale)(0);
      }
    }
    
    if (pb) {
      if (t % 100 == 0 || t == iteration - 1) {
        // Calculate progress and time estimates
        progress = static_cast<double>(t + 1) / iteration;
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
        double remaining = elapsed / progress * (1 - progress);
        
        // Progress bar
        Rcpp::Rcout << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
          if (i < pos) Rcpp::Rcout << "=";
          else if (i == pos) Rcpp::Rcout << ">";
          else Rcpp::Rcout << " ";
        }
        
        // Display information
        Rcpp::Rcout << ANSI_BOLD << ANSI_BG_BLUE << ANSI_WHITE << " SSG RUNNING " << ANSI_RESET << " ";
        
        if (t == iteration - 1) {
          Rcpp::Rcout << ANSI_BOLD << ANSI_BRIGHT_GREEN << "✓ " 
                      << ANSI_BG_GREEN << ANSI_WHITE << "COMPLETED" << ANSI_RESET << " "
                      << ANSI_BOLD << ANSI_BRIGHT_GREEN << "100%" << ANSI_RESET << "  "
                      << "Iter: " << ANSI_BOLD << ANSI_BRIGHT_RED << iteration << ANSI_RESET << "  "
                      << "Time: " << ANSI_BOLD << ANSI_BRIGHT_RED << elapsed << "s" << ANSI_RESET
                      << ANSI_BRIGHT_GREEN << " ✔" << ANSI_RESET << "\n";
        } else { 
          Rcpp::Rcout << ANSI_BOLD << "[" << ANSI_BRIGHT_GREEN 
                      << std::setw(3) << int(progress * 100.0) << "%" << ANSI_RESET << "] "
                      << ANSI_BOLD << ANSI_BRIGHT_RED << std::setw(5) << t + 1 << ANSI_RESET 
                      << "/" << iteration << " "
                      << ANSI_BRIGHT_RED << "⏱ " << elapsed << "s" 
                      << "<" << remaining << "s" << ANSI_RESET
                      << "   \r";
        }
      } 
    }
    ///////////////////////////////////////////////////
    /////////////////// Likelihood ////////////////////
    ///////////////////////////////////////////////////
    if (likelihood) {
      // Observed likelihood
      obs_likelihood = log_likelihood_observed(X, pi, mu, prec);
      // Complete likelihood
      arma::uvec z_uvec = arma::conv_to<arma::uvec>::from(z);
      comp_likelihood = log_likelihood_complete(X, z_uvec, pi, mu, prec);
    }
    ///////////////////////////////////////////////////
    ///////////////// Store Results ///////////////////
    ///////////////////////////////////////////////////
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = durationOld + std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    durationOld = duration;
    
    if (onlyComp) {
      if(t%thin == 0 && t > burnin-1) {
        TIME[idx] = duration;
        if (likelihood) {
          COMP_LIK[idx] = comp_likelihood;
        }
        idx = idx + 1;
      }
    } else {
      if(t%thin == 0 && t > burnin-1) {
        Z.row(idx) = z;
        PI.row(idx) = pi;
        TIME[idx] = duration;
        PROB[idx] = probAllocation;
        MU[idx] = mu;
        PREC[idx] = prec;
        if (likelihood) {
          OBS_LIK[idx] = obs_likelihood;
          COMP_LIK[idx] = comp_likelihood;
        }
        idx = idx + 1;
      }
    }
  }
  if (onlyComp) {
    return List::create(Named("Complete_Likelihood") = COMP_LIK,
                        Named("Execution_Time") = TIME);
  } else {
    return List::create(Named("Allocation") = Z,
                        Named("Probability") = PROB,
                        Named("Proportion_Parameters") = PI,
                        Named("Mu") = MU,
                        Named("Precision") = PREC,
                        Named("Observed_Likelihood") = OBS_LIK,
                        Named("Complete_Likelihood") = COMP_LIK,
                        Named("Execution_Time") = TIME);
  }
}



////////////////////////////////////////////////////
////////////// Random Gibbs Sampler ///////////////
///////////////////////////////////////////////////

// Random Gibbs sampler d-dimensional
// [[Rcpp::export]]
List RSSG(arma::mat X, arma::vec hyper, int K, int m, int iteration, 
          int burnin, int thin, String method, bool trueParameters,
          arma::irowvec trueAll, arma::mat trueMean, arma::mat truePrec,
          arma::rowvec truePerc, int seed, bool pb, bool likelihood,
          bool onlyComp) {
  arma::arma_rng::set_seed(seed);
  Rcpp::Environment base_env = Rcpp::Environment::namespace_env("base");
  Rcpp::Function set_seed_r = base_env["set.seed"];
  set_seed_r(seed);
  std::srand(seed);
  // precision and not variance!!
  // m: how many observation I want to update
  ////////////////////////////////////////////////////
  ////////////////// Initial settings ///////////////
  ///////////////////////////////////////////////////
  int nout = (iteration-burnin)/thin; // number of sample from the posterior
  int n = X.n_rows; // number of rows
  int d = X.n_cols; // number of columns
  NumericVector indC(K); // index for the cluster
  for (int k = 0; k<K; k++) {
    indC(k) = k;
  }
  NumericVector indI(n); // index for the observation
  for (int i = 0; i<n; i++) {
    indI(i) = i;
  }
  // uniform weight vector
  NumericVector constVal(n);
  for (int i = 0; i<n; i++) {
    constVal(i) = 1.0/n;
  }
  int idx = 0;
  // Z
  arma::imat Z(nout, n);
  // ALPHA
  NumericMatrix ALPHA(nout, n);
  // PI
  arma::mat PI(nout, K);
  // Time 
  NumericVector TIME(nout);
  // Entropy
  NumericMatrix D(nout, n);
  // MU
  List MU(nout);
  // SIGMA
  List PREC(nout);
  // OBSERVED LIKELIHOOD
  NumericVector OBS_LIK(nout);
  // COMPLETE LIKELIHOOD
  NumericVector COMP_LIK(nout);
  double obs_likelihood = 0.0;
  double comp_likelihood = 0.0;
  ////////////////////////////////////////////////////
  ////////// Emprical bayes prior settings ///////////
  ///////////////////////////////////////////////////
  arma::rowvec hyper_mu_mean(d);
  double hyper_prec_b;
  double hyper_prec_a;
  arma::mat hyper_mu_prec(K, d);
  if (method == "EB") {
    hyper_mu_mean = mean(X, 0);
    hyper_prec_a = d + 2;
    hyper_prec_b = sum(var(X, 0)/d)/(pow(K, 2.0/d));
  } else {
    cout << "Remember to specify the hyperparameter values!" << "\n";
    for (int j = 0; j<d; j++) {
      hyper_mu_mean(j) = hyper(2);
    } 
    hyper_prec_b = hyper(5);
    for (int k = 0; k<K; k++) {
      hyper_mu_prec.row(k) = hyper(3)*arma::ones<arma::rowvec>(d);
    } 
    hyper_prec_a = hyper(4);
  } 
  ////////////////////////////////////////////////////
  ////////////////// Initial value //////////////////
  ///////////////////////////////////////////////////
  // Inizialize the vector allocation
  arma::irowvec z(n);
  // Iniziallize the mean matrix
  arma::mat mu(K, d);
  // Inizialize the precision matrix
  arma::mat prec(K, d);
  // Inizialize the percentage vector
  arma::rowvec pi(K);
  arma::vec concPar = (hyper(1)/K) * arma::ones<arma::vec>(K);
  if (!trueParameters) {
    // ALLOCATION 
    NumericVector probC(K);
    for (int k = 0; k<K; k++) {
      probC(k) = hyper(0)/K;
    }
    for (int i = 0; i<n; i++) {
      // z(i) = csample_num(indC, 1, true, probC)(0);
      z(i) = csample_num_new(indC, 1, probC)(0);
    }
    // PI
    pi = rdirichlet_cpp(1, concPar);
    // PRECISION
    for (int j = 0; j<d; j++) {
      for (int k = 0; k<K; k++) {
        prec(k,j) = callrgamma(1, hyper_prec_a, 1.0/hyper_prec_b)(0);
      }
    }
    // MU
    if (method == "EB") {
      double kP = 0.01;
      for (int j = 0; j<d; j++) {
        for (int k = 0; k<K; k++) {
          hyper_mu_prec(k,j) = prec(k,j);
          mu(k,j) = R::rnorm(hyper_mu_mean(j), 1.0/sqrt(kP*hyper_mu_prec(k,j)));
        } 
      } 
    } else {
      for (int j = 0; j<d; j++) {
        for (int k = 0; k<K; k++) {
          mu(k,j) = R::rnorm(hyper_mu_mean(j), 1.0/sqrt(hyper_mu_prec(k,j)));
        }
      } 
    }
  } else {
    cout << "True parameters are used!" << "\n";
    // true allocation
    z = trueAll;
    // true mean
    mu = trueMean;
    // true precision
    prec = truePrec;
    // true percentage
    pi = truePerc;
  }
  NumericVector rI(m);
  // Probability Matrix
  NumericMatrix probAllocation(n, K);
  // alpha uniform weight 
  NumericVector alpha = constVal;
  // Time 
  double durationOld;
  // Progress bar setup
  int barWidth = 30;
  double progress = 0.0;
  ////////////////////////////////////////////////////
  /////////////////// Main Part /////////////////////
  ///////////////////////////////////////////////////
  for (int t = 0; t<iteration; t++) {
    // start time
    auto start = std::chrono::high_resolution_clock::now();
    // sample according to alpha (uniform)
    // rI = csample_num(indI, m, false, alpha);
    rI = csample_num_new(indI, m, alpha);
    // update probability
    // STEP 1. Compute the probability
    for (int i = 0; i<m; i++) {
      for (int k = 0; k<K; k++) {
        arma::vec vecTmp(d);
        for (int j = 0; j<d; j++) {
          vecTmp(j) = R::dnorm(X(rI[i],j), mu(k,j), sqrt(1.0/prec(k,j)), true);
        }
        probAllocation(rI[i],k) = exp(log(pi(k)) + mySum(vecTmp)) + std::numeric_limits<double>::denorm_min();
      } 
    }
    // Normalize the rows
    arma::vec rSum = rowSums(probAllocation);
    for (int i = 0; i<m; i++) {
      probAllocation(rI[i], _) = probAllocation(rI[i], _) / rSum(rI[i]);
    }
    // update z
    for (int i = 0; i<m; i++) {
      NumericVector prob_r = as<NumericVector>(wrap(probAllocation(rI[i], _)));
      z(rI[i]) = csample_num_new(indC, 1, prob_r)(0);
      // z(rI[i]) = csample_num(indC, 1, false, probAllocation(rI[i], _))(0);
    }
    // compute N
    arma::irowvec N(K);
    for (int i = 0; i < n; i++) {
      N[z[i]] = N[z[i]]+1;
    }
    // update pi
    arma::rowvec parDirich = concPar.t() + N;
    arma::vec parDirichR = parDirich.t();
    pi = rdirichlet_cpp(1, parDirichR);
    // update params
    // mu
    double muPost;
    double precPost;
    arma::mat sampMean(K, d);
    // Prior Setting (tutti i cluster partono con la stessa hyperpriori per la media)
    for (int k = 0; k<K; k++) {
      for (int j = 0; j<d; j++) {
        for (int i = 0; i<n; i++) {
          sampMean(k,j) = sampMean(k,j) + diracF(z(i), k)*X(i, j);
        }
      } 
    } 
    
    for (int j = 0; j<d; j++) {
      for (int k = 0; k<K; k++) {
        muPost = (prec(k,j)*sampMean(k,j)+hyper(2)*hyper(3))/(prec(k,j)*N(k)+hyper(3));
        precPost = prec(k,j)*N(k)+hyper(3);
        mu(k,j) = R::rnorm(muPost, sqrt(1.0/precPost));
      } 
    } 
    // precision
    arma::mat sumNum(K, d);
    for (int k = 0; k<K; k++) {
      for (int j = 0; j<d; j++) {
        for (int i = 0; i<n; i++) {
          sumNum(k,j) = sumNum(k,j) + diracF(z(i), k)*pow(X(i,j)-mu(k,j), 2);
        }
      }
    }
    for (int j = 0; j<d; j++) {
      for (int k = 0; k<K; k++) {
        double shape = hyper(4)+(N(k)/2.0);
        double scale = hyper(5)+(sumNum(k,j)/2.0);
        prec(k,j) = callrgamma(1, shape, 1.0/scale)(0);
      }
    }
    
    if (pb) {
      if (t % 100 == 0 || t == iteration - 1) {
        // Calculate progress and time estimates
        progress = static_cast<double>(t + 1) / iteration;
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
        double remaining = elapsed / progress * (1 - progress);
        
        // Progress bar
        Rcpp::Rcout << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
          if (i < pos) Rcpp::Rcout << "=";
          else if (i == pos) Rcpp::Rcout << ">";
          else Rcpp::Rcout << " ";
        }
        
        // Display information
        Rcpp::Rcout << ANSI_BOLD << ANSI_BG_BLUE << ANSI_WHITE << " RSG RUNNING " << ANSI_RESET << " ";
        
        if (t == iteration - 1) {
          Rcpp::Rcout << ANSI_BOLD << ANSI_BRIGHT_GREEN << "✓ " 
                      << ANSI_BG_GREEN << ANSI_WHITE << "COMPLETED" << ANSI_RESET << " "
                      << ANSI_BOLD << ANSI_BRIGHT_GREEN << "100%" << ANSI_RESET << "  "
                      << "Iter: " << ANSI_BOLD << ANSI_BRIGHT_RED << iteration << ANSI_RESET << "  "
                      << "Time: " << ANSI_BOLD << ANSI_BRIGHT_RED << elapsed << "s" << ANSI_RESET
                      << ANSI_BRIGHT_GREEN << " ✔" << ANSI_RESET << "\n";
        } else {
          Rcpp::Rcout << ANSI_BOLD << "[" << ANSI_BRIGHT_GREEN 
                      << std::setw(3) << int(progress * 100.0) << "%" << ANSI_RESET << "] "
                      << ANSI_BOLD << ANSI_BRIGHT_RED << std::setw(5) << t + 1 << ANSI_RESET 
                      << "/" << iteration << " "
                      << ANSI_BRIGHT_RED << "⏱ " << elapsed << "s" 
                      << "<" << remaining << "s" << ANSI_RESET
                      << "   \r";
        }
      }
    }
    ///////////////////////////////////////////////////
    /////////////////// Likelihood ////////////////////
    ///////////////////////////////////////////////////
    if (likelihood) {
      // Observed likelihood
      obs_likelihood = log_likelihood_observed(X, pi, mu, prec);
      // Complete likelihood
      arma::uvec z_uvec = arma::conv_to<arma::uvec>::from(z);
      comp_likelihood = log_likelihood_complete(X, z_uvec, pi, mu, prec);
    }
    ////////////////////////////////////////////////////
    ///////////////// Store Results ///////////////////
    ///////////////////////////////////////////////////
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = durationOld + std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();//1000000;
    durationOld = duration;
    
    if (onlyComp) {
      if(t%thin == 0 && t > burnin-1) {
        TIME[idx] = duration;
        if (likelihood) {
          COMP_LIK[idx] = comp_likelihood;
        }
        idx = idx + 1;
      }
    } else {
      if(t%thin == 0 && t > burnin-1) {
        Z.row(idx) = z;
        PI.row(idx) = pi;
        TIME[idx] = duration;
        MU[idx] = mu;
        PREC[idx] = prec;
        if (likelihood) {
          OBS_LIK[idx] = obs_likelihood;
          COMP_LIK[idx] = comp_likelihood;
        }
        idx = idx + 1;
      }
    }

  }
  if (onlyComp) {
    return List::create(Named("Complete_Likelihood") = COMP_LIK,
                        Named("Execution_Time") = TIME);
  } else {
    return List::create(Named("Allocation") = Z,
                        Named("Proportion_Parameters") = PI,
                        Named("Mu") = MU,
                        Named("Precision") = PREC,
                        Named("Observed_Likelihood") = OBS_LIK,
                        Named("Complete_Likelihood") = COMP_LIK,
                        Named("Execution_Time") = TIME);
  }
}

////////////////////////////////////////////////////
////////// Entropy-Guided Gibbs Sampler ///////////
//////////////////////////////////////////////////


double myRound(double x) {
  double out = round(x*1000.0) / 1000.0;
  return(out);
}

double find_lambda(const arma::vec& pi, double m_target, double max_lambda, double lambda_init = 1.0, double tol = 1e-2, int max_iter = 10) {
  double lambda = std::clamp(lambda_init, 1.0, max_lambda);
  double ESS = 0.0;
  
  for(int i = 0; i < max_iter; ++i) {
    arma::vec a = -lambda * pi;
    double m1 = arma::max(a);
    arma::vec exp_a = arma::exp(a - m1);
    double log_S1 = m1 + std::log(arma::sum(exp_a));
     
    arma::vec b = -2.0 * lambda * pi;
    double m2 = arma::max(b);
    arma::vec exp_b = arma::exp(b - m2);
    double log_S2 = m2 + std::log(arma::sum(exp_b));
     
    ESS = std::exp(2 * log_S1 - log_S2);
     
    double sum_pi_exp_a = arma::dot(pi, exp_a) * std::exp(m1);
    double sum_pi_exp_b = arma::dot(pi, exp_b) * std::exp(m2);
     
    double dS1_dlambda = -sum_pi_exp_a;
    double dS2_dlambda = -2.0 * sum_pi_exp_b;
     
    double numerator = 2.0 * std::exp(2 * log_S1 - log_S2) * dS1_dlambda - ESS * dS2_dlambda;
    double denominator = std::exp(2 * log_S2);
    double dESS_dlambda = numerator / denominator;
     
    double h = ESS - m_target;
     
    if (std::abs(dESS_dlambda) < 1e-10 || !std::isfinite(dESS_dlambda)) break;
     
    double delta = h / dESS_dlambda;
    double lambda_new = std::clamp(lambda - delta, 1.0, max_lambda);
     
    if (std::abs(lambda_new - lambda) < tol) {
      lambda = lambda_new;
      break;
    }
    
    lambda = lambda_new;
  }
  
  return lambda;
} 

double sma(
    const arma::vec& x, 
    int L,              
    double lambda_hat   
) {

  if (L <= 0) {
    return std::max(lambda_hat, 1.0);
  }
  
  arma::uvec non_zero_indices = arma::find(x != 0.0);
  int count_non_zero = non_zero_indices.n_elem;
   
  int take = std::min(L - 1, count_non_zero);
   
  double sum_x = 0.0;
  if (take > 0) {
    int start_idx = std::max(0, count_non_zero - take);
    arma::uvec selected_indices = non_zero_indices.subvec(start_idx, count_non_zero - 1);
    sum_x = arma::sum(x.elem(selected_indices));
  }
   
  int denominator = take + 1;
   
  double average = (sum_x + lambda_hat) / denominator;
  return std::max(average, 1.0);
}


// [[Rcpp::export]]
List DiversityGibbsSamp(arma::mat X, arma::vec hyper, int K, 
                        double m, int iteration, int burnin, 
                        int thin, int updateProbAlloc, String method, 
                        double q, double lambda,
                        double kWeibull, double alphaPareto, 
                        double xmPareto, String DiversityIndex, 
                        bool adaptive, double nSD, double lambda0,
                        int L, double max_lambda, double c, double a, String w_fun, int sp,
                        int seed, bool pb, bool likelihood, bool onlyComp) {
  arma::arma_rng::set_seed(seed);
  Rcpp::Environment base_env = Rcpp::Environment::namespace_env("base");
  Rcpp::Function set_seed_r = base_env["set.seed"];
  set_seed_r(seed);
  std::srand(seed);
  // precision and not variance!!
  // m: how many observation I want to update
  ////////////////////////////////////////////////////
  ////////////////// Initial settings ///////////////
  ///////////////////////////////////////////////////
  int nout = (iteration-burnin)/thin; // number of sample from the posterior
  int n = X.n_rows; // number of rows
  int d = X.n_cols; // number of columns
  NumericVector indC(K); // index for the cluster
  for (int k = 0; k<K; k++) {
    indC(k) = k;
  }
  NumericVector indI(n); // index for the observation
  for (int i = 0; i<n; i++) {
    indI(i) = i;
  }
  // uniform weight vector
  NumericVector constVal(n);
  for (int i = 0; i<n; i++) {
    constVal(i) = 1.0/n;
  }
  // ones vector
  NumericVector OnesVal(n);
  for (int i = 0; i<n; i++) {
    OnesVal(i) = 1.0;
  }
  int idx = 0;
  // TIME 
  NumericVector TIME(nout);
  // Z
  arma::imat Z(nout, n);
  // Parameter
  NumericVector PAR(nout);
  // Probability allocation
  List PROB(nout);
  // ALPHA
  NumericMatrix ALPHA(nout, n);
  // Probability Distribution
  NumericMatrix PDIST(nout, n);
  // PI
  arma::mat PI(nout, K);
  // Entropy
  NumericMatrix D(nout, n);
  // MU
  List MU(nout);
  // SIGMA
  List PREC(nout);
  // OBSERVED LIKELIHOOD
  NumericVector OBS_LIK(nout);
  // COMPLETE LIKELIHOOD
  NumericVector COMP_LIK(nout);
  double obs_likelihood = 0.0;
  double comp_likelihood = 0.0;
  ////////////////////////////////////////////////////
  ////////// Emprical bayes prior settings ///////////
  ///////////////////////////////////////////////////
  arma::rowvec hyper_mu_mean(d);
  double hyper_prec_b;
  double hyper_prec_a;
  arma::mat hyper_mu_prec(K, d);
  if (method == "EB") {
    hyper_mu_mean = mean(X, 0);
    hyper_prec_a = d + 2;
    hyper_prec_b = sum(var(X, 0)/d)/(pow(K, 2.0/d));
  } else {
    cout << "Remember to specify the hyperparameter values!" << "\n";
    for (int j = 0; j<d; j++) {
      hyper_mu_mean(j) = hyper(2);
    } 
    hyper_prec_b = hyper(5);
    for (int k = 0; k<K; k++) {
      hyper_mu_prec.row(k) = hyper(3)*arma::ones<arma::rowvec>(d);
    } 
    hyper_prec_a = hyper(4);
  } 
  if (q < 0) {
    cout << "q should be greater or equal to 0!" << "\n";
  }
  // Progress bar setup
  int barWidth = 30;
  double progress = 0.0;
  ////////////////////////////////////////////////////
  ////////////////// Initial value //////////////////
  ///////////////////////////////////////////////////
  // Z
  arma::irowvec z(n);
  NumericVector probC(K);
  for (int k = 0; k<K; k++) {
    probC(k) = hyper(0)/K;
  } 
  for (int i = 0; i<n; i++) {
    // z(i) = csample_num(indC, 1, true, probC)(0);
    z(i) = csample_num_new(indC, 1, probC)(0);
  } 
  // PI
  arma::rowvec pi(K);
  arma::vec concPar = (hyper(1)/K) * arma::ones<arma::vec>(K);
  pi = rdirichlet_cpp(1, concPar);
  // PRECISION
  arma::mat prec(K, d);
  for (int j = 0; j<d; j++) {
    for (int k = 0; k<K; k++) {
      prec(k,j) = callrgamma(1, hyper_prec_a, 1.0/hyper_prec_b)(0);
    }
  }
  // MU
  arma::mat mu(K, d);
  if (method == "EB") {
    double kP = 0.01;
    for (int j = 0; j<d; j++) {
      for (int k = 0; k<K; k++) {
        hyper_mu_prec(k,j) = prec(k,j);
        mu(k,j) = R::rnorm(hyper_mu_mean(j), 1.0/sqrt(kP*hyper_mu_prec(k,j)));
      } 
    } 
  } else {
    for (int j = 0; j<d; j++) {
      for (int k = 0; k<K; k++) {
        mu(k,j) = R::rnorm(hyper_mu_mean(j), 1.0/sqrt(hyper_mu_prec(k,j)));
      }
    } 
  }
  NumericVector rI(m);
  // Probability Matrix
  NumericMatrix probAllocation(n, K);
  // alpha weight 
  NumericVector alpha(n);
  NumericVector alpha_norm(n);
  NumericVector alpha_prec(n);
  NumericVector alpha_custom(n);
  // adaptive diversity function
  int updateLambda = 1;
  NumericVector probDiv(n);
  double lambda_est;
  double sds = ceil(sqrt((n/m)*K*(K-1)));
  double s = ceil((n/m)*(K-1) + nSD*sds);
  // Time 
  double durationOld;
  ////////////////////////////////////////////////////
  /////////////////// Main Part /////////////////////
  ///////////////////////////////////////////////////
  // To update the allocation matrix, the chain is divided into 3 segments:  
  // 1. First, split the chain into two equal halves (50% each)  
  // 2. Split the first half into two sub-segments:  
  //    - First sub-segment: Update matrix every 3 iterations  
  //    - Second sub-segment: Update matrix every 6 iterations  
  // 3. In the second half: Update matrix every 10 iterations  
  for (int t = 0; t<iteration; t++) {
    // start time
    auto start = std::chrono::high_resolution_clock::now();
    // schedule for updating the probability allocation matrix
    if (updateProbAlloc == 0) {
      // 1-5-10, 3-6-10, 5-8-10, 7-10-15, 10-15-20
      if (t <= ceil(iteration/4)) {
        updateProbAlloc = 3;
        if (lambda0 == 0) {updateLambda = 3;}
      } else if (t > ceil(iteration/4) && t <= ceil(iteration/2)) {
        updateProbAlloc = 6; 
        if (lambda0 == 0) {updateLambda = 6;}
      } else {
        updateProbAlloc = 10;
        if (lambda0 == 0) {updateLambda = 10;}
      }
    }
    // if (lambda0 == 0) {updateLambda = 100;}
    // update probability
    if (t%updateProbAlloc == 0) {
      for (int i = 0; i<n; i++) {
        for (int k = 0; k<K; k++) {
          arma::vec vecTmp(d);
          for (int j = 0; j<d; j++) {
            vecTmp(j) = R::dnorm(X(i,j), mu(k,j), sqrt(1.0/prec(k,j)), true);
          }
          probAllocation(i,k) = exp(log(pi(k)) + mySum(vecTmp)) + std::numeric_limits<double>::denorm_min();
        } 
      }
      // Normalize the rows
      arma::vec rSum = rowSums(probAllocation);
      for (int i = 0; i<n; i++) {
        probAllocation(i, _) = probAllocation(i, _) / rSum(i);
      }
    }
    if (adaptive) {
      DiversityIndex = "Exponential";
      if (t > sp) {
        lambda = 1;
      } else {
        if (lambda0 == 0) {
          if (t%updateLambda == 0) {
            for (int i = 0; i<n; i++) {
              int col = z[i];
              probDiv[i] = probAllocation(i, col);
            }
            lambda_est = find_lambda(probDiv, c*m, max_lambda);
            lambda = sma(PAR, L, lambda_est);
          }
        } else {
          lambda = lambda0;
        }
      }
    }
    NumericVector Diversity(n);
    if (DiversityIndex == "Generalized-Entropy") {
      if (q == 1) {
        for (int i = 0; i<n; i++) {
          for (int k = 0; k<K; k++) {
            if (probAllocation(i, k) == 0) {
              Diversity(i) = Diversity(i) + 0; // define log(0) = 0
            } else {
              Diversity(i) = Diversity(i) + probAllocation(i, k)*log2(probAllocation(i, k)); 
            }
          }
          Diversity(i) = -Diversity(i);
        }
      } else {
        for (int i = 0; i<n; i++) {
          for (int k = 0; k<K; k++) {
            Diversity(i) = Diversity(i) + pow(probAllocation(i, k), q);
          }
          Diversity(i) = (1-Diversity(i))/(q-1);
        }
      } 
    } else if (DiversityIndex == "Partial-Generalized-Entropy") {
      if (q == 1) {
        for (int i = 0; i<n; i++) {
          if (probAllocation(i, z(i)) == 0.00) {
            Diversity(i) = 0; // define log(0) = 0
          } else {
            Diversity(i) = probAllocation(i, z(i))*log2(probAllocation(i, z(i))); 
          }
          Diversity(i) = -Diversity(i);
        }
      } else if (q == 0) {
        for (int i = 0; i<n; i++) {
          Diversity(i) = 1 - probAllocation(i, z(i));
        }
      } else {
        for (int i = 0; i<n; i++) {
          Diversity(i) = pow(probAllocation(i, z(i)), q);
        }
        Diversity = (1-Diversity)/(q-1);
      }
    } else if (DiversityIndex == "Exponential") {
      for (int i = 0; i<n; i++) {
        // Exponetial
        Diversity(i) = exp(-lambda*probAllocation(i, z(i)));
      }
    } else if (DiversityIndex == "Pareto") {
      for (int i = 0; i<n; i++) {
        // Pareto
        Diversity(i) = (alphaPareto*pow(xmPareto, alphaPareto))/pow(probAllocation(i, z(i))+0.00001, alphaPareto + 1);
      }
    }
    else if (DiversityIndex == "Weibull") {
      for (int i = 0; i<n; i++) {
        // Weibull
        Diversity(i) = (kWeibull/lambda)*pow(probAllocation(i, z(i)), kWeibull-1)*exp(-pow(probAllocation(i, z(i)/lambda), kWeibull));
      }
    } else if (DiversityIndex == "Hyperbole") {
      for (int i = 0; i<n; i++) {
        // Hyperbole
        Diversity(i) = pow(1/(probAllocation(i, z(i)) + 0.0001), lambda);
      }
    }
    if (sp != 0) {
      if (t <= sp) {
        w_fun = "hyperbolic";
      } else {
        w_fun = "polynomial";
      }
    }
    if (w_fun == "hyperbolic") {
      if (t == 0) {
        alpha = constVal;
      } else {
        alpha = alpha_prec*tanup(t, s, a)+tanlo(t, s, a)*Diversity;
      }
    } else if (w_fun == "polynomial") {
      if (sp == 0) {
        if (t == 0) {
          alpha = constVal;
        } else {
          alpha = alpha_prec*(t/(t+s))+(s/(t+s))*Diversity;
        }
      } else {
        s = 1;
        if (t == 0) {
          alpha = constVal;
        } else {
          alpha = alpha_prec*((t-sp+1)/(t-sp+1+s))+(s/(t-sp+1+s))*Diversity;
        }
      }
    }
    // Normalize the alpha vector
    double sumAlpha = sum(alpha);
    for (int i = 0; i<n; i++) {
      alpha_norm(i) = (alpha(i) / sumAlpha);
    }
    // sample according to alpha
    // rI = csample_num(indI, m, false, alpha_norm);
    rI = csample_num_new(indI, m, alpha_norm);
    // update z
    for (int i = 0; i<m; i++) {
      NumericVector prob_r = as<NumericVector>(wrap(probAllocation(rI[i], _)));
      z(rI[i]) = csample_num_new(indC, 1, prob_r)(0);
      // z(rI[i]) = csample_num(indC, 1, false, probAllocation(rI[i], _))(0);
    }
    // compute N
    arma::irowvec N(K);
    for (int i = 0; i < n; i++) {
      N[z[i]] = N[z[i]]+1;
    }
    // update pi
    arma::rowvec parDirich = concPar.t() + N;
    arma::vec parDirichR = parDirich.t();
    pi = rdirichlet_cpp(1, parDirichR);
    // update params
    // mu
    double muPost;
    double precPost;
    arma::mat sampMean(K, d);
    // Prior Setting
    for (int k = 0; k<K; k++) {
      for (int j = 0; j<d; j++) {
        for (int i = 0; i<n; i++) {
          sampMean(k,j) = sampMean(k,j) + diracF(z(i), k)*X(i, j);
        }
      } 
    }
    for (int j = 0; j<d; j++) {
      for (int k = 0; k<K; k++) {
        muPost = (prec(k,j)*sampMean(k,j)+hyper(2)*hyper(3))/(prec(k,j)*N(k)+hyper(3));
        precPost = prec(k,j)*N(k)+hyper(3);
        mu(k,j) = R::rnorm(muPost, sqrt(1.0/precPost));
      } 
    } 
    // precision
    arma::mat sumNum(K, d);
    for (int k = 0; k<K; k++) {
      for (int j = 0; j<d; j++) {
        for (int i = 0; i<n; i++) {
          sumNum(k,j) = sumNum(k,j) + diracF(z(i), k)*pow(X(i,j)-mu(k,j), 2);
        }
      }
    }
    for (int j = 0; j<d; j++) {
      for (int k = 0; k<K; k++) {
        double shape = hyper(4)+(N(k)/2.0);
        double scale = hyper(5)+(sumNum(k,j)/2.0);
        prec(k,j) = callrgamma(1, shape, 1.0/scale)(0);
      }
    }
    alpha_prec = alpha;
    
    if (pb) {
      ////////////////////////////////////////////////////
      ///////////////// Progress Reporting ///////////////
      ////////////////////////////////////////////////////
      if (t % 100 == 0 || t == iteration - 1) {
        // Calculate progress and time estimates
        progress = static_cast<double>(t + 1) / iteration;
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start).count();
        double remaining = elapsed / progress * (1 - progress);
        
        // Progress bar
        Rcpp::Rcout << "[";
        int pos = barWidth * progress;
        for (int i = 0; i < barWidth; ++i) {
          if (i < pos) Rcpp::Rcout << "=";
          else if (i == pos) Rcpp::Rcout << ">";
          else Rcpp::Rcout << " ";
        }
        
        // Display information
        bool is_adapting = t < sp;
        
        if (is_adapting) {
          Rcpp::Rcout << ANSI_BOLD << ANSI_BG_YELLOW << ANSI_WHITE << " ADAPTING λ " << ANSI_RESET << " ";
        } else { 
          Rcpp::Rcout << ANSI_BOLD << ANSI_BG_CYAN << ANSI_WHITE << " FIXED λ " << ANSI_RESET << " ";
        }
         
        if (t == iteration - 1) {
          Rcpp::Rcout << ANSI_BOLD << ANSI_BRIGHT_GREEN << "✓ " 
                      << ANSI_BG_GREEN << ANSI_WHITE << "COMPLETED" << ANSI_RESET << " "
                      << ANSI_BOLD << ANSI_BRIGHT_GREEN << "100%" << ANSI_RESET << "  "
                      << "Iter: " << ANSI_BOLD << ANSI_BRIGHT_RED << iteration << ANSI_RESET << "  "
                      << "Time: " << ANSI_BOLD << ANSI_BRIGHT_RED << elapsed << "s" << ANSI_RESET
                      << ANSI_BRIGHT_GREEN << " ✔" << ANSI_RESET << "\n";
        } else {
          Rcpp::Rcout << ANSI_BOLD << "[" << ANSI_BRIGHT_GREEN 
                      << std::setw(3) << int(progress * 100.0) << "%" << ANSI_RESET << "] "
                      << ANSI_BOLD << ANSI_BRIGHT_RED << std::setw(5) << t + 1 << ANSI_RESET 
                      << "/" << iteration << " "
                      << ANSI_BRIGHT_RED << "⏱ " << elapsed << "s" 
                      << "<" << remaining << "s" << ANSI_RESET
                      << "   \r";
        }
      }
    }
    ///////////////////////////////////////////////////
    /////////////////// Likelihood ////////////////////
    ///////////////////////////////////////////////////
    if (likelihood) {
      // Observed likelihood
      obs_likelihood = log_likelihood_observed(X, pi, mu, prec);
      // Complete likelihood
      arma::uvec z_uvec = arma::conv_to<arma::uvec>::from(z);
      comp_likelihood = log_likelihood_complete(X, z_uvec, pi, mu, prec);
    }
    ////////////////////////////////////////////////////
    ///////////////// Store Results ///////////////////
    ///////////////////////////////////////////////////
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = durationOld + std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();//1000000;
    durationOld = duration;
    
    if (onlyComp) {
      if(t%thin == 0 && t > burnin-1) {
        TIME[idx] = duration;
        if (likelihood) {
          COMP_LIK[idx] = comp_likelihood;
        }
        idx = idx + 1;
      }
    } else {
      if(t%thin == 0 && t > burnin-1) {
        Z.row(idx) = z;
        PDIST.row(idx) = probDiv;
        TIME[idx] = duration;
        PAR[idx] = lambda;
        PROB[idx] = probAllocation;
        ALPHA.row(idx) = alpha;
        PI.row(idx) = pi;
        D.row(idx) = Diversity;
        MU[idx] = mu;
        PREC[idx] = prec;
        if (likelihood) {
          OBS_LIK[idx] = obs_likelihood;
          COMP_LIK[idx] = comp_likelihood;
        }
        idx = idx + 1;
      }
    }
  } // END MCMC
  
  // Final newline after progress bar
  Rcpp::Rcout << std::endl;
  
  // Counter for threshold exceedances
  int count = 0;
  // Scan iterations
  for(int i = 0; i < sp; ++i) {
    if(PAR[i] >= max_lambda) {
      count++;
    }
  }
  // Calculate percentage
  double percentage = (static_cast<double>(count) / sp) * 100;
  // Generate warning
  if(percentage > 20.0) {
    std::string msg = 
      "Warning: Lambda exceeded max_lambda " + 
      std::to_string(percentage) + 
      "% of the time in the first " + 
      std::to_string(sp) + 
      " iterations. It is recommended to increase the value of max_lambda!";
    Rcpp::warning(msg);
  }
  if (onlyComp) {
    return List::create(Named("Complete_Likelihood") = COMP_LIK,
                        Named("Execution_Time") = TIME);
  } else {
    return List::create(Named("Allocation") = Z,
                        Named("Allocation_Probability_Matrix") = PROB,
                        Named("Lambda_Parameter") = PAR,
                        Named("Probability_Vector_Distribution") = PDIST,
                        Named("Diversity") = D,
                        Named("Proportion_Parameters") = PI,
                        Named("Mu") = MU,
                        Named("Precision") = PREC,
                        Named("Alpha") = ALPHA,
                        Named("Observed_Likelihood") = OBS_LIK,
                        Named("Complete_Likelihood") = COMP_LIK,
                        Named("Execution_Time") = TIME); 
  }
  
}


////////////////////////////////////////////////////
/////////////// Categorical Data //////////////////
///////////////////////////////////////////////////

// [[Rcpp::export]]
List CSSG(arma::mat X, arma::vec hyper, int K, int R, int iteration, int burnin, int thin) {
  // start time
  // We suppose that R1 = R2 = ... = Rd
  auto start = std::chrono::high_resolution_clock::now();
  ////////////////////////////////////////////////////
  ////////////////// Initial settings ///////////////
  ///////////////////////////////////////////////////
  int nout = (iteration-burnin)/thin; // number of sample from the posterior
  int n = X.n_rows; // number of rows
  int d = X.n_cols; // number of columns
  int idx = 0;
  // Index for the cluster
  NumericVector indC(K);
  for (int k = 0; k<K; k++) {
    indC(k) = k;
  }
  // Z
  arma::imat Z(nout, n);
  // PI
  arma::mat PI(nout, K);
  // Probability for each category
  List PROBCAT(nout);
  ////////////////////////////////////////////////////
  ////////////////// Initial value //////////////////
  ///////////////////////////////////////////////////
  // Z
  arma::irowvec z(n);
  NumericVector probC(K);
  for (int k = 0; k<K; k++) {
    probC(k) = hyper(0)/K;
  }  
  for (int i = 0; i<n; i++) {
    z(i) = csample_num(indC, 1, true, probC)(0);
  }  
  // PI
  arma::rowvec pi(K);
  arma::vec concPar = (hyper(1)/K) * arma::ones<arma::vec>(K);
  pi = rdirichlet_cpp(1, concPar);
  // PROBABILITY FOR EACH CATEGORY - Dimension JxKxR (dimension x cluster x category)
  arma::cube probCat(R, d, K);
  arma::vec initR = (hyper(2)/R) * arma::ones<arma::vec>(R);
  for (int j = 0; j<d; j++) {
    for (int k = 0; k<K; k++) {
      probCat.slice(k).col(j) = rdirichlet_cpp(1, initR).t();
    }
  }
  // Probability Matrix
  NumericMatrix probAllocation(n, K);
  arma::vec vecTmp(d);
  ////////////////////////////////////////////////////
  /////////////////// Main Part /////////////////////
  ///////////////////////////////////////////////////
  for (int t = 0; t<iteration; t++) {
    // update probability
    // STEP 1. Compute the probability
    for (int k = 0; k<K; k++) {
      for (int i = 0; i<n; i++) {
        for (int j = 0; j<d; j++) {
          vecTmp(j) = probCat.slice(k)(X(i,j), j);
        } 
        probAllocation(i,k) = pi(k)*myProduct(vecTmp);
      }  
    }
    // update z - DONE
    for (int i = 0; i<n; i++) {
      z(i) = csample_num(indC, 1, false, probAllocation(i, _))(0);
    } 
    // compute N - DONE
    arma::irowvec N(K);
    for (int i = 0; i < n; i++) {
      N[z[i]] = N[z[i]] + 1;
    }
    // compute Nr - DONE
    arma::cube Nr(R, d, K);
    for (int k = 0; k<K; k++) {
      for (int i = 0; i<n; i++) {
        if (z(i) == k) {
          for (int j = 0; j<d; j++) {
            Nr.slice(k)(X(i,j), j) = Nr.slice(k)(X(i,j), j) + 1;
          } 
        }
      }
    }
    // update pi - DONE
    arma::rowvec parDirich = concPar.t() + N;
    pi = rdirichlet_cpp(1, parDirich.t());
    // update category probability - DONE
    for (int j = 0; j<d; j++) {
      for (int k = 0; k<K; k++) {
        arma::rowvec parDirichCat = (initR + Nr.slice(k).col(j)).t();
        probCat.slice(k).col(j) = rdirichlet_cpp(1, parDirichCat.t()).t();
      }
    } 
    ////////////////////////////////////////////////////
    ///////////////// Store Results ///////////////////
    ///////////////////////////////////////////////////
    if(t%thin == 0 && t > burnin-1) {
      Z.row(idx) = z;
      PI.row(idx) = pi;
      PROBCAT[idx] = probCat;
      idx = idx + 1;
    } 
    if (t%1000 == 0 && t > 0) {
      std::cout << "Iteration: " << t << " (of " << iteration << ")\n";
    }
  } 
  std::cout << "End MCMC!\n";
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  return List::create(Named("Allocation") = Z,
                        Named("Proportion_Parameters") = PI,
                        Named("Category_Probability") = PROBCAT,
                        Named("Execution_Time") = duration/1000000);
}



// [[Rcpp::export]]
List CRSG(arma::mat X, arma::vec hyper, int K, int R, int m, int iteration, int burnin, int thin) {
  // start time
  // We suppose that R1 = R2 = ... = Rd
  auto start = std::chrono::high_resolution_clock::now();
  ////////////////////////////////////////////////////
  ////////////////// Initial settings ///////////////
  ///////////////////////////////////////////////////
  int nout = (iteration-burnin)/thin; // number of sample from the posterior
  int n = X.n_rows; // number of rows
  int d = X.n_cols; // number of columns
  int idx = 0;
  // Index for the cluster
  NumericVector indC(K);
  for (int k = 0; k<K; k++) {
    indC(k) = k;
  }
  NumericVector indI(n); // index for the observation
  for (int i = 0; i<n; i++) {
    indI(i) = i;
  }
  // uniform weight vector
  NumericVector constVal(n);
  for (int i = 0; i<n; i++) {
    constVal(i) = 1.0/n;
  } 
  // Z
  arma::imat Z(nout, n);
  // PI
  arma::mat PI(nout, K);
  // Probability for each category
  List PROBCAT(nout);
  if (n < m) {
    cout << "The number of observations should be greater or equal to the number of samples!" << "\n";
  }
  ////////////////////////////////////////////////////
  ////////////////// Initial value //////////////////
  ///////////////////////////////////////////////////
  // Z
  arma::irowvec z(n);
  NumericVector probC(K);
  for (int k = 0; k<K; k++) {
    probC(k) = hyper(0)/K;
  }  
  for (int i = 0; i<n; i++) {
    z(i) = csample_num(indC, 1, true, probC)(0);
  }  
  // PI
  arma::rowvec pi(K);
  arma::vec concPar = (hyper(1)/K) * arma::ones<arma::vec>(K);
  pi = rdirichlet_cpp(1, concPar);
  // PROBABILITY FOR EACH CATEGORY
  arma::cube probCat(R, d, K); 
  arma::vec initR = (hyper(2)/R) * arma::ones<arma::vec>(R);
  for (int j = 0; j<d; j++) {
    for (int k = 0; k<K; k++) {
      probCat.slice(k).col(j) = rdirichlet_cpp(1, initR).t();
    }
  }
  // random sample
  NumericVector rI(m);
  // alpha weight
  NumericVector alpha = constVal;
  // Probability Matrix
  NumericMatrix probAllocation(n, K);
  arma::vec vecTmp(d);
  ////////////////////////////////////////////////////
  /////////////////// Main Part /////////////////////
  ///////////////////////////////////////////////////
  for (int t = 0; t<iteration; t++) {
    // sample according to alpha (uniform)
    rI = csample_num(indI, m, false, alpha);
    // update probability
    // STEP 1. Compute the probability
    for (int i = 0; i<m; i++) {
      for (int k = 0; k<K; k++) {
        for (int j = 0; j<d; j++) {
          vecTmp(j) = probCat.slice(k)(X(rI(i),j), j);
        }
        probAllocation(rI(i),k) = pi(k)*myProduct(vecTmp);
      } 
    }
    // Normalize the rows
    arma::vec rSum = rowSums(probAllocation);
    for (int i = 0; i<m; i++) {
      probAllocation(rI(i), _) = probAllocation(rI(i), _) / rSum(rI(i));
    }
    // update z
    for (int i = 0; i<m; i++) {
      z(rI(i)) = csample_num(indC, 1, false, probAllocation(rI(i), _))(0);
    }
    // compute N
    arma::irowvec N(K);
    for (int i = 0; i < n; i++) {
      N[z[i]] = N[z[i]] + 1;
    }
    // compute Nr
    arma::cube Nr(R, d, K);
    for (int k = 0; k<K; k++) {
      for (int i = 0; i<n; i++) {
        if (z(i) == k) {
          for (int j = 0; j<d; j++) {
            Nr.slice(k)(X(i,j), j) = Nr.slice(k)(X(i,j), j) + 1;
          } 
        }
      }
    }
    // update pi
    arma::rowvec parDirich = concPar.t() + N;
    pi = rdirichlet_cpp(1, parDirich.t());
    // update category probability
    for (int j = 0; j<d; j++) {
      for (int k = 0; k<K; k++) {
        arma::rowvec parDirichCat = (initR + Nr.slice(k).col(j)).t();
        probCat.slice(k).col(j) = rdirichlet_cpp(1, parDirichCat.t()).t();
      }
    } 
    ////////////////////////////////////////////////////
    ///////////////// Store Results ///////////////////
    ///////////////////////////////////////////////////
    if(t%thin == 0 && t > burnin-1) {
      Z.row(idx) = z;
      PI.row(idx) = pi;
      PROBCAT[idx] = probCat;
      idx = idx + 1;
    }
    if (t%1000 == 0 && t > 0) {
      std::cout << "Iteration: " << t << " (of " << iteration << ")\n";
    }
  }
  std::cout << "End MCMC!\n";
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  return List::create(Named("Allocation") = Z,
                        Named("Proportion_Parameters") = PI,
                        Named("Category_Probability") = PROBCAT,
                        Named("Execution_Time") = duration/1000000);
}


// [[Rcpp::export]]
List CDSG(arma::mat X, arma::vec hyper, int K, int R, int m, int iteration, 
               int burnin, int iterTuning, 
               int thin, int updateProbAlloc, String method, 
               double gamma, double q, double lambda, 
               double kWeibull, double alphaPareto, 
               double xmPareto, String DiversityIndex, 
               bool adaptive, double nSD, double lambda0,
               double zeta, double a) {
  // start time
  // We suppose that R1 = R2 = ... = Rd
  auto start = std::chrono::high_resolution_clock::now();
  ////////////////////////////////////////////////////
  ////////////////// Initial settings ///////////////
  ///////////////////////////////////////////////////
  int nout = (iteration-burnin)/thin; // number of sample from the posterior
  int n = X.n_rows; // number of rows
  int d = X.n_cols; // number of columns
  int idx = 0;
  // Index for the cluster
  NumericVector indC(K);
  for (int k = 0; k<K; k++) {
    indC(k) = k;
  }
  NumericVector indI(n); // index for the observation
  for (int i = 0; i<n; i++) {
    indI(i) = i;
  } 
  // uniform weight vector
  NumericVector constVal(n);
  for (int i = 0; i<n; i++) {
    constVal(i) = 1.0/n;
  } 
  // Z
  arma::imat Z(nout, n);
  // PI
  arma::mat PI(nout, K);
  // Probability for each category
  List PROBCAT(nout);
  // Diversity
  NumericMatrix D(nout, n);
  if (q < 0) {
    cout << "q should be greater or equal to 0!" << "\n";
  }
  if (n < m) {
    cout << "The number of observations should be greater or equal to the number of samples!" << "\n";
  }
  ////////////////////////////////////////////////////
  ////////////////// Initial value //////////////////
  ///////////////////////////////////////////////////
  // Z
  arma::irowvec z(n);
  NumericVector probC(K);
  for (int k = 0; k<K; k++) {
    probC(k) = hyper(0)/K;
  }  
  for (int i = 0; i<n; i++) {
    z(i) = csample_num(indC, 1, true, probC)(0);
  }  
  // PI
  arma::rowvec pi(K);
  arma::vec concPar = (hyper(1)/K) * arma::ones<arma::vec>(K);
  pi = rdirichlet_cpp(1, concPar);
  // PROBABILITY FOR EACH CATEGORY - Dimension JxKxR (dimension x cluster x category)
  arma::cube probCat(R, d, K);
  arma::vec initR = (hyper(2)/R) * arma::ones<arma::vec>(R);
  for (int j = 0; j<d; j++) {
    for (int k = 0; k<K; k++) {
      probCat.slice(k).col(j) = rdirichlet_cpp(1, initR).t();
    }
  }
  // random sample
  NumericVector rI(m);
  // alpha weight
  NumericVector alpha(n);
  NumericVector alpha_prec = constVal;
  // Probability Matrix
  NumericMatrix probAllocation(n, K);
  arma::vec vecTmp(d);
  double sds = ceil(sqrt((n/m)*K*(K-1)));
  double s = ceil((n/m)*(K-1)) + nSD*sds;
  ////////////////////////////////////////////////////
  /////////////////// Main Part /////////////////////
  ///////////////////////////////////////////////////
  for (int t = 0; t<iteration; t++) {
    // update probability
    if (t%updateProbAlloc == 0) {
      for (int i = 0; i<n; i++) {
        for (int k = 0; k<K; k++) {
          for (int j = 0; j<d; j++) {
            // ATTENZIONE
            vecTmp(j) = probCat.slice(k)(X(i,j), j);
          }
          probAllocation(i,k) = pi(k)*myProduct(vecTmp);
        }
      }
      // Normalize the rows
      arma::vec rSum = rowSums(probAllocation);
      for (int i = 0; i<n; i++) {
        probAllocation(i, _) = probAllocation(i, _) / rSum(i);
      }
    }
    if (adaptive && t <= iterTuning) {
      DiversityIndex = "Exponential";
      lambda = lambda0*pow(zeta, t) + 1; // 996
    }
    NumericVector Diversity(n);
    if (DiversityIndex == "Generalized-Entropy") {
      if (q == 1) {
        for (int i = 0; i<n; i++) {
          for (int k = 0; k<K; k++) {
            if (probAllocation(i, k) == 0) {
              Diversity(i) = Diversity(i) + 0; // define log(0) = 0
            } else { 
              Diversity(i) = Diversity(i) + probAllocation(i, k)*log2(probAllocation(i, k)); 
            }
          } 
          Diversity(i) = -Diversity(i);
        } 
      } else {
        for (int i = 0; i<n; i++) {
          for (int k = 0; k<K; k++) {
            Diversity(i) = Diversity(i) + pow(probAllocation(i, k), q);
          } 
          Diversity(i) = (1-Diversity(i))/(q-1);
        } 
      } 
    } else if (DiversityIndex == "Partial-Generalized-Entropy") {
      if (q == 1) {
        for (int i = 0; i<n; i++) {
          if (probAllocation(i, z(i)) == 0.00) {
            Diversity(i) = 0; // define log(0) = 0
          } else { 
            Diversity(i) = probAllocation(i, z(i))*log2(probAllocation(i, z(i))); 
          } 
          Diversity(i) = -Diversity(i);
        } 
      } else if (q == 0) {
        for (int i = 0; i<n; i++) {
          Diversity(i) = 1 - probAllocation(i, z(i));
        } 
      } else {
        for (int i = 0; i<n; i++) {
          Diversity(i) = pow(probAllocation(i, z(i)), q);
        } 
        Diversity = (1-Diversity)/(q-1);
      }
    } else if (DiversityIndex == "Exponential") {
      for (int i = 0; i<n; i++) {
        // Exponetial
        Diversity(i) = lambda*exp(-lambda*probAllocation(i, z(i)));
      } 
    } else if (DiversityIndex == "Pareto") {
      for (int i = 0; i<n; i++) {
        // Pareto
        Diversity(i) = (alphaPareto*pow(xmPareto, alphaPareto))/pow(probAllocation(i, z(i))+0.00001, alphaPareto + 1);
      }
    } 
    else if (DiversityIndex == "Weibull") {
      for (int i = 0; i<n; i++) {
        // Weibull
        Diversity(i) = (kWeibull/lambda)*pow(probAllocation(i, z(i)), kWeibull-1)*exp(-pow(probAllocation(i, z(i)/lambda), kWeibull));
      } 
    } else if (DiversityIndex == "Hyperbole") {
      for (int i = 0; i<n; i++) {
        // Hyperbole
        Diversity(i) = pow(1/(probAllocation(i, z(i)) + 0.0001), lambda);
      } 
    }
    // Normalize
    double sumDiv = sum(Diversity);
    for (int i = 0; i<n; i++) {
      Diversity(i) = (Diversity(i) / sumDiv);
    } 
    // update alpha
    if (t == 0 || t == 1) {
      alpha = gamma*Diversity+(1-gamma)*constVal;
    } else { 
      alpha = gamma*(alpha_prec*(t/(t+s))+(s/(t+s))*Diversity)+(1-gamma)*constVal;
      // alpha = gamma*(alpha_prec*tanup(t, s, a)+tanlo(t, s, a)*Diversity)+(1-gamma)*constVal;
    } 
    // sample according to alpha
    rI = csample_num(indI, m, false, alpha); 
    // sample according to alpha
    rI = csample_num(indI, m, false, alpha);
    // update z
    for (int i = 0; i<m; i++) {
      z(rI[i]) = csample_num(indC, 1, false, probAllocation(rI[i], _))(0);
    }
    // compute N
    arma::irowvec N(K);
    for (int i = 0; i < n; i++) {
      N[z[i]] = N[z[i]]+1;
    } 
    // compute Nr
    arma::cube Nr(R, d, K);
    for (int k = 0; k<K; k++) {
      for (int i = 0; i<n; i++) {
        if (z(i) == k) {
          for (int j = 0; j<d; j++) {
            Nr.slice(k)(X(i,j), j) = Nr.slice(k)(X(i,j), j) + 1;
          } 
        }
      }
    }
    // update pi
    arma::rowvec parDirich = concPar.t() + N;
    pi = rdirichlet_cpp(1, parDirich.t());
    // update category probability
    for (int j = 0; j<d; j++) {
      for (int k = 0; k<K; k++) {
        arma::rowvec parDirichCat = (initR + Nr.slice(k).col(j)).t();
        probCat.slice(k).col(j) = rdirichlet_cpp(1, parDirichCat.t()).t();
      }
    } 
    alpha_prec = alpha;
    ////////////////////////////////////////////////////
    ///////////////// Store Results ///////////////////
    ///////////////////////////////////////////////////
    if(t%thin == 0 && t > burnin-1) {
      Z.row(idx) = z;
      PI.row(idx) = pi;
      D.row(idx) = Diversity;
      PROBCAT[idx] = probCat;
      idx = idx + 1;
    } 
    if (t%5000 == 0 && t > 0) {
      std::cout << "Iteration: " << t << " (of " << iteration << ")\n";
    } 
  }
  std::cout << "End MCMC!\n";
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  return List::create(Named("Allocation") = Z,
                        Named("Proportion_Parameters") = PI,
                        Named("Category_Probability") = PROBCAT,
                        Named("Diversity") = D, 
                        Named("Execution_Time") = duration/1000000);
}
