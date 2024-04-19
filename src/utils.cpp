#include <RcppArmadillo.h>
#include <math.h>
#include <R.h>
#include <Rmath.h>
#include <stdlib.h> 
#include <chrono>
#include <RcppArmadilloExtensions/sample.h>

using namespace Rcpp;
using namespace arma;
using namespace std;

// TODO
// [X] data in d-dimension
// [X] How to set the Hyperparameters?
// From Bayesian Regularization for Normal Mixture Estimation and Model-Based Clustering 
// by Fraley and Raftery page 159-160. We have 4 more hyperparameters:
// 1. mu ~ N(muP, sigma^2/kP)
// 2. 1/sigma^2 ~ G(vP^2/2, zetaP^2/2) !!!!Occhio is conditional!!!.
// for the mean the mean of the data. kP = 0.01
// vP = d+2, zetaP = sum(diag(var(data))/d/G^(2/d), where G number of component
// [X] Random Gibbs sampler
// [X] Entropy-Giuded Adaptive Gibbs sampler
// [X] Check the generating mechanism
// [X] Plot with entropy
// Emphasise the color
// [X] How to summarize the posterior? (in C++)
// [ ] Code Optimization
// [X] Check the full conditional with the new Regularization
// [ ] Evaluate Convergence
// [ ] How many time I need to update the Prob Allocation Matrix?

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
int diracF(int a, int b){
  if (a == b) {
    return 1;
  } else {
    return 0;
  }
}

// come campionare dalla distribuzione di Dirichlet
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

// [[Rcpp::export]]
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
  // cout << "Numeratore " << num << "\n";
  // cout << "Denominatore " << den << "\n";
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


////////////////////////////////////////////////////
//////////////// D-Dimensional Data ////////////////
///////////////////////////////////////////////////

// [[Rcpp::export]]
List SSG(arma::mat X, arma::vec hyper, int K, int iteration, int burnin, int thin, String method, arma::irowvec trueAllocation) {
  // precision and not variance!!
  // m: how many observation I want to update
  // start time
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
  // MU
  List MU(nout);
  // SIGMA
  List PREC(nout);
  // ARI Index
  arma::vec LOSS(nout);
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
  // Probability Matrix
  NumericMatrix probAllocation(n, K);
  // Loss index
  double Loss = 0;
  ////////////////////////////////////////////////////
  /////////////////// Main Part /////////////////////
  ///////////////////////////////////////////////////
  for (int t = 0; t<iteration; t++) {
    // update probability
    // STEP 1. Compute the probability
    for (int i = 0; i<n; i++) {
      for (int k = 0; k<K; k++) {
        arma::vec vecTmp(d);
        for (int j = 0; j<d; j++) {
          vecTmp(j) = R::dnorm(X(i,j), mu(k,j), sqrt(1.0/prec(k,j)), FALSE);
        }
        probAllocation(i,k) = pi(k)*myProduct(vecTmp);
      } 
    }
    // Normalize the rows
    arma::vec rSum = rowSums(probAllocation);
    for (int i = 0; i<n; i++) {
      probAllocation(i, _) = probAllocation(i, _) / rSum(i);
    }
    // update z
    for (int i = 0; i<n; i++) {
      z(i) = csample_num(indC, 1, false, probAllocation(i, _))(0);
    }
    // compute N
    arma::irowvec N(K);
    for (int i = 0; i < n; i++) {
      N[z[i]] = N[z[i]]+1;
    }
    // update pi
    arma::rowvec parDirich = ((concPar.t())/K) + N;
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
    ////////////////////////////////////////////////////
    ///////////////// Store Results ///////////////////
    ///////////////////////////////////////////////////
    if(t%thin == 0 && t > burnin-1) {
      Z.row(idx) = z;
      PI.row(idx) = pi;
      MU[idx] = mu;
      PREC[idx] = prec;
      if (sum(trueAllocation) != 0) {
        Loss = ari(z, trueAllocation);
        // Loss = BinderLoss(z, trueAllocation);
        LOSS(idx) = Loss;
      }
      idx = idx + 1;
    }
    if (t%1000 == 0 && t > 0) {
      std::cout << "Iteration: " << t << " (of " << iteration << ")\n";
    }
  }
  std::cout << "End MCMC!\n";
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  if (sum(trueAllocation) != 0) {
    return List::create(Named("Allocation") = Z,
                        Named("Proportion_Parameters") = PI,
                        Named("Mu") = MU,
                        Named("Precision") = PREC,
                        Named("Loss") = LOSS,
                        Named("Execution_Time") = duration/1000000);
  } else {
    return List::create(Named("Allocation") = Z,
                        Named("Proportion_Parameters") = PI,
                        Named("Mu") = MU,
                        Named("Precision") = PREC,
                        Named("Execution_Time") = duration/1000000);
  }
}



////////////////////////////////////////////////////
////////////// Random Gibbs Sampler ///////////////
///////////////////////////////////////////////////

// Random Gibbs sampler d-dimensional
// [[Rcpp::export]]
List RSSG(arma::mat X, arma::vec hyper, int K, int m, int iteration, int burnin, int thin, String method, arma::irowvec trueAllocation) {
  // precision and not variance!!
  // m: how many observation I want to update
  // start time
  auto start = std::chrono::high_resolution_clock::now();
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
  // Entropy
  NumericMatrix D(nout, n);
  // MU
  List MU(nout);
  // SIGMA
  List PREC(nout);
  // ARI Index
  arma::vec LOSS(nout);
  // Adaptive Gamma
  arma::vec GAMMA(nout);
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
  NumericVector Diversity(n);
  // alpha uniform weight 
  NumericVector alpha = constVal;
  // Loss index
  double Loss = 0;
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
        arma::vec vecTmp(d);
        for (int j = 0; j<d; j++) {
          vecTmp(j) = R::dnorm(X(rI[i],j), mu(k,j), sqrt(1.0/prec(k,j)), FALSE);
        }
        probAllocation(rI[i],k) = pi(k)*myProduct(vecTmp);
      } 
    }
    // Normalize the rows
    arma::vec rSum = rowSums(probAllocation);
    for (int i = 0; i<m; i++) {
      probAllocation(rI[i], _) = probAllocation(rI[i], _) / rSum(rI[i]);
    }
    // update z
    for (int i = 0; i<m; i++) {
      z(rI[i]) = csample_num(indC, 1, false, probAllocation(rI[i], _))(0);
    }
    // compute N
    arma::irowvec N(K);
    for (int i = 0; i < n; i++) {
      N[z[i]] = N[z[i]]+1;
    }
    // update pi
    arma::rowvec parDirich = ((concPar.t())/K) + N;
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
    ////////////////////////////////////////////////////
    ///////////////// Store Results ///////////////////
    ///////////////////////////////////////////////////
    if(t%thin == 0 && t > burnin-1) {
      Z.row(idx) = z;
      PI.row(idx) = pi;
      MU[idx] = mu;
      PREC[idx] = prec;
      if (sum(trueAllocation) != 0) {
        Loss = ari(z, trueAllocation);
        // Loss = BinderLoss(z, trueAllocation);
        LOSS(idx) = Loss;
      }
      idx = idx + 1;
    }
    if (t%1000 == 0 && t > 0) {
      std::cout << "Iteration: " << t << " (of " << iteration << ")\n";
    }
  }
  std::cout << "End MCMC!\n";
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  if (sum(trueAllocation) != 0) {
      return List::create(Named("Allocation") = Z,
                          Named("Proportion_Parameters") = PI,
                          Named("Mu") = MU,
                          Named("Precision") = PREC,
                          Named("Loss") = LOSS,
                          Named("Execution_Time") = duration/1000000);
  } else {
      return List::create(Named("Allocation") = Z,
                          Named("Proportion_Parameters") = PI,
                          Named("Mu") = MU,
                          Named("Precision") = PREC,
                          Named("Execution_Time") = duration/1000000);
  }
}

////////////////////////////////////////////////////
////////// Entropy-Guided Gibbs Sampler ///////////
//////////////////////////////////////////////////

// [[Rcpp::export]]
double JS_distance(NumericVector p, NumericVector q) {
  int n = p.size();
  NumericVector D_KL_p(n);
  NumericVector D_KL_q(n);
  // cout << "p " << p << "n";
  // cout << "q " << q << "n";
  NumericVector meanDist = 0.5*(p+q);
  // cout << "mean distance " << meanDist << "\n";
  for (int i = 0; i<n; i++) {
    D_KL_p(i) = p(i)*log2((p(i)/meanDist(i)) + 0.000001);
    D_KL_q(i) = q(i)*log2((q(i)/meanDist(i)) + 0.000001);
  }
  // Levare gli NA
  // cout << D_KL_p << "\n";
  // cout << D_KL_q << "\n";
  double sum_D_KL_p = sum(D_KL_p);
  double sum_D_KL_q = sum(D_KL_q);
  // cout << sum_D_KL_p << "\n";
  // cout << sum_D_KL_q << "\n";
  double distance = 0.5 * (sum_D_KL_p + sum_D_KL_q);
  return(distance);
}


// [[Rcpp::export]]
List DiversityGibbsSamp(arma::mat X, arma::vec hyper, int K, int m, int iteration, int burnin, int thin, String method, double gamma, arma::irowvec trueAllocation, bool adaptiveGamma, double q) {
  // precision and not variance!!
  // m: how many observation I want to update
  // start time
  auto start = std::chrono::high_resolution_clock::now();
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
  // Entropy
  NumericMatrix D(nout, n);
  // MU
  List MU(nout);
  // SIGMA
  List PREC(nout);
  // ARI Index
  arma::vec LOSS(nout);
  // Adaptive Gamma
  arma::vec GAMMA(nout);
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
  if (q < 0) {
    cout << "q should be greater or equal to 0!" << "\n";
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
  NumericVector alpha_prec = constVal;
  // Loss index
  double Loss = 0;
  ////////////////////////////////////////////////////
  /////////////////// Main Part /////////////////////
  ///////////////////////////////////////////////////
  for (int t = 0; t<iteration; t++) {
    // update probability
    for (int i = 0; i<n; i++) {
      for (int k = 0; k<K; k++) {
        arma::vec vecTmp(d);
        for (int j = 0; j<d; j++) {
          vecTmp(j) = R::dnorm(X(i,j), mu(k,j), sqrt(1.0/prec(k,j)), FALSE);
        }
        probAllocation(i,k) = pi(k)*myProduct(vecTmp);
      } 
    }
    // Normalize the rows
    arma::vec rSum = rowSums(probAllocation);
    for (int i = 0; i<n; i++) {
      probAllocation(i, _) = probAllocation(i, _) / rSum(i);
    }
    NumericVector Diversity(n);
    if (q == 1) {
      for (int i = 0; i<n; i++) {
        for (int k = 0; k<K; k++) {
          Diversity(i) = Diversity(i) + probAllocation(i, k)*log2(probAllocation(i, k));
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
    // Normalize
    double sumDiv = sum(Diversity);
    for (int i = 0; i<n; i++) {
      Diversity(i) = (Diversity(i) / sumDiv);
    } 
    if (adaptiveGamma == true) {
      gamma = 1-pow(JS_distance(Diversity, constVal), (1.0/q));
    }
    // update alpha
    if (t == 0 || t == 1) {
      alpha = gamma*Diversity+(1-gamma)*constVal;
    } else {
      alpha = alpha_prec*((t-1.0)/t)+(1.0/t)*(gamma*Diversity+(1-gamma)*constVal);
    }
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
    // update pi
    arma::rowvec parDirich = ((concPar.t())/K) + N;
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
    alpha_prec = alpha;
    ////////////////////////////////////////////////////
    ///////////////// Store Results ///////////////////
    ///////////////////////////////////////////////////
    if(t%thin == 0 && t > burnin-1) {
      Z.row(idx) = z;
      ALPHA.row(idx) = alpha;
      PI.row(idx) = pi;
      D.row(idx) = Diversity;
      MU[idx] = mu;
      PREC[idx] = prec;
      if (sum(trueAllocation) != 0) {
        Loss = ari(z, trueAllocation);
        // Loss = BinderLoss(z, trueAllocation);
        LOSS(idx) = Loss;
        // cout << "Matrice di Contingenza: \n" << contMat(z, trueAllocation) << "\n";
      }
      if (adaptiveGamma == true) {
        GAMMA(idx) = gamma;
      }
      idx = idx + 1;
    }
    if (t%1000 == 0 && t > 0) {
      std::cout << "Iteration: " << t << " (of " << iteration << ")\n";
    }
  }
  std::cout << "End MCMC!\n";
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  if (sum(trueAllocation) != 0) {
    if (adaptiveGamma == true) {
      return List::create(Named("Allocation") = Z,
                          Named("Diversity") = D,
                          Named("Proportion_Parameters") = PI,
                          Named("Mu") = MU,
                          Named("Precision") = PREC,
                          Named("Loss") = LOSS,
                          Named("Alpha") = ALPHA,
                          Named("Gamma") = GAMMA,
                          Named("Execution_Time") = duration/1000000);
    } else {
      return List::create(Named("Allocation") = Z,
                          Named("Diversity") = D,
                          Named("Proportion_Parameters") = PI,
                          Named("Mu") = MU,
                          Named("Precision") = PREC,
                          Named("Loss") = LOSS,
                          Named("Alpha") = ALPHA,
                          Named("Execution_Time") = duration/1000000);
    }
  } else {
    if (adaptiveGamma == true) {
      return List::create(Named("Allocation") = Z,
                          Named("Diversity") = D,
                          Named("Proportion_Parameters") = PI,
                          Named("Mu") = MU,
                          Named("Precision") = PREC,
                          Named("Alpha") = ALPHA,
                          Named("Gamma") = GAMMA,
                          Named("Execution_Time") = duration/1000000);
    } else {
      return List::create(Named("Allocation") = Z,
                          Named("Diversity") = D,
                          Named("Proportion_Parameters") = PI,
                          Named("Mu") = MU,
                          Named("Precision") = PREC,
                          Named("Alpha") = ALPHA,
                          Named("Execution_Time") = duration/1000000);
    }
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
  // cout << "Somma per riga " << sum(sumPost, 0) << "\n";
  // cout << "Somma per colonna " << sum(sumPost, 1) << "\n";
  return(sumPost);
}




////////////////////////////////////////////////////
/////////////// Categorical Data //////////////////
///////////////////////////////////////////////////

