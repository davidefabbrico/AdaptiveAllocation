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
// 2. 1/sigma^2 ~ G(vP^2/2, zetaP^2/2)
// for the mean the mean of the data. kP = 0.01
// vP = d+2, zetaP = sum(diag(var(data))/d/G^(2/d), where G number of component
// [ ] How to summarize the posterior?
// [ ] Random Gibbs sampler
// [ ] Entropy-Giuded Adaptive Gibbs sampler 
// [ ] Code Optimization

// Modello:
// xi | zi, mu, sigma^2 è una normale , da cui R::rnorm(0, 1)
// zi | pi1, ..., piK è una Multinomiale, da cui non sappiamo ancora campionare
// pi1, ..., piK segue una distribuzione di Dirichlet, da cui non sappiamo ancora campionare
// mu segue una Normale con iperparametri mu0 e sigma0^2
// sigma^2 segue una inverse gamma 1/callrgamma

// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::irowvec sum_allocation(arma::irowvec z, int K) {
  int n = z.n_elem;
  arma::irowvec cont(K);
  for (int i = 0; i < n; i++) {
    cont[z[i]]++;
  }
  return cont;
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
                           NumericVector prob
) {
  NumericVector ret = Rcpp::RcppArmadillo::sample(x, size, replace, prob);
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

// FCD z
// [[Rcpp::export]]
arma::irowvec update_allocation(arma::rowvec pi, arma::vec mu, arma::vec prec, arma::vec x) {
  // pi sono le proporzioni di ogni cluster
  // mu è la media a posteriori
  // sigma è la varianza a posteriori
  // x dati
  int K = pi.n_elem; // numero di cluster
  int n = x.n_elem; // numero di osservazioni
  NumericMatrix probCluster(n, K);
  arma::irowvec z(n);
  for (int i = 0; i<n; i++) {
    for (int k = 0; k<K; k++) {
      probCluster(i,k) = (pi(k)*R::dnorm(x(i), mu(k), sqrt(1/prec(k)), FALSE));
    }
  }
  // normalizzo le righe
  for (int i = 0; i<n; i++) {
    probCluster(i, _) = probCluster(i, _) / sum(probCluster(i, _));
  }
  // cout << "Probability Matrix Allocation: " << probCluster;
  NumericVector indC(K);
  for (int k = 0; k<K; k++) {
    indC(k) = k;
  }
  for (int i = 0; i<n; i++) {
    z(i) = csample_num(indC, 1, true, probCluster(i, _))(0);
  }
  return z;
}

// FCD pi
// [[Rcpp::export]]
arma::rowvec update_pi(arma::rowvec alpha, arma::irowvec N) {
  // a is the concentration parameter
  // N is the sum of allocation
  int K = N.n_elem;
  arma::rowvec parDirich = alpha/K + N;
  arma::vec parDirichR = parDirich.t();
  return rdirichlet_cpp(1, parDirichR);
}

// FCD mu
// [[Rcpp::export]]
arma::vec update_mu(double mu0, double p0, arma::vec prec, arma::irowvec z, arma::irowvec N, arma::vec x) {
  // s0 la varianza a priori
  // mu la media a posteriori
  // sigma la varianza a posteriori
  // z la matrice di allocazione
  // N la somma delle allocazioni
  int K = N.n_elem;
  int n = x.n_elem;
  arma::vec MuPosteriori(K);
  arma::vec muPost(K);
  arma::vec precPost(K);
  arma::vec sampMean(K);
  for (int k = 0; k<K; k++) {
    for (int i = 0; i<n; i++) {
      sampMean(k) = sampMean(k) + diracF(z(i), k)*x(i);
    }
  }
  for (int k = 0; k<K; k++) {
    muPost(k) = (prec(k)*sampMean(k)+mu0*p0)/(prec(k)*N(k)+p0);
    precPost(k) = prec(k)*N(k)+p0;
    MuPosteriori(k) = R::rnorm(muPost(k), sqrt(1/precPost(k)));
  }
  return MuPosteriori;
}


// FCD sigma
// [[Rcpp::export]]
arma::vec update_prec(double a0, double b0, arma::vec mu, arma::irowvec z, arma::irowvec N, arma::vec x) {
  // a0 iperparametro a priori della inverse gamma
  // b0 iperparametro a priori della inverse gamma
  // mu vettore delle medie a posteriori
  // z matrice di allocazione
  // N somme matrice allocazione
  // x dati
  int n = x.n_elem;
  int K = N.n_elem;
  arma::vec prec(K);
  arma::vec sumNum(K);
  // cout << "Medie: " << mu << "\n";
  for (int k = 0; k<K; k++) {
    for (int i = 0; i<n; i++) {
      sumNum(k) = sumNum(k) + diracF(z(i), k)*pow(x(i)-mu(k), 2);
    }
  }
  arma::vec shape(K);
  arma::vec scale(K);
  for (int k = 0; k<K; k++) {
    shape(k) = a0+(N(k)/2);
    // compute the sum
    scale(k) = b0+sumNum(k)/2;
    prec(k) = callrgamma(1, shape(k), 1/scale(k))(0);
  }
  return prec;
}

// Gibbs sampler 1-dimension
// [[Rcpp::export]]
List SSG(arma::vec x, arma::vec hyper, int K, int iteration, int burnin, int thin) {
  auto start = std::chrono::high_resolution_clock::now();
  ////////////////////////////////////////////////////
  ////////////////// Initial settings ///////////////
  ///////////////////////////////////////////////////
  int nout = (iteration-burnin)/thin;
  int n = x.n_elem;
  NumericVector indC(K);
  for (int k = 0; k<K; k++) {
    indC(k) = k;
  }
  int idx = 0;
  // Z
  arma::imat Z(nout, n);
  // PI
  arma::mat PI(nout, K);
  // MU
  arma::mat MU(nout, K);
  // SIGMA
  arma::mat PREC(nout, K);
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
  arma::vec concPar = hyper(1) * arma::ones<arma::vec>(K);
  pi = rdirichlet_cpp(1, concPar);
  // MU
  arma::vec mu(K);
  for (int k = 0; k<K; k++) {
    // se 1/ --> precisione
    // se normale --> varianza
    mu(k) = R::rnorm(hyper(2), 1/sqrt(hyper(3)));
  }
  // PRECISION
  arma::vec prec(K);
  prec = callrgamma(K, hyper(4), 1/hyper(5));
  ////////////////////////////////////////////////////
  /////////////////// Main Part /////////////////////
  ///////////////////////////////////////////////////
  for (int t = 0; t<iteration; t++) {
    // update z
    z = update_allocation(pi, mu, prec, x);
    // compute N
    arma::irowvec N = sum_allocation(z, K);
    // update pi
    pi = update_pi(concPar.t(), N);
    // update params
    // mu
    mu = update_mu(hyper(2), hyper(3), prec, z, N, x);
    // precision
    prec = update_prec(hyper(4), hyper(5), mu, z, N, x); 
    ////////////////////////////////////////////////////
    ///////////////// Store Results ///////////////////
    ///////////////////////////////////////////////////
    if(t%thin == 0 && t > burnin-1) {
      Z.row(idx) = z;
      PI.row(idx) = pi;
      MU.row(idx) = mu.t();
      PREC.row(idx) = prec.t();
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
                      Named("Mu") = MU,
                      Named("Precision") = PREC,
                      Named("Execution_Time") = duration/1000000);
  
}


////////////////////////////////////////////////////
//////////////// D-Dimensional Data ////////////////
///////////////////////////////////////////////////
double myProduct(arma::vec a) {
  double prod = 1;
  int n = a.n_elem;
  for (int i = 0; i<n; i++) {
    prod = prod*a(i);
  }
  return prod;
}

// FCD z
// [[Rcpp::export]]
arma::irowvec update_allocationD(arma::rowvec pi, arma::mat mu, arma::mat prec, arma::mat X) {
  // pi sono le proporzioni di ogni cluster
  // mu è la media a posteriori
  // sigma è la varianza a posteriori
  // x dati
  int K = pi.n_elem; // numero di cluster
  int n = X.n_rows; // numero di osservazioni
  int d = X.n_cols; // numero di covariate
  NumericMatrix probCluster(n, K);
  arma::irowvec z(n);
  arma::vec vecTmp(d);
  for (int i = 0; i<n; i++) {
    for (int k = 0; k<K; k++) {
      arma::vec vecTmp = arma::zeros<arma::vec>(d);
      for (int j = 0; j<d; j++) {
        vecTmp(j) = R::dnorm(X(i,j), mu(k,j), sqrt(1/prec(k,j)), FALSE);
      }
      probCluster(i,k) = pi(k)*myProduct(vecTmp);
    }
  }
  // normalizzo le righe
  for (int i = 0; i<n; i++) {
    probCluster(i, _) = probCluster(i, _) / sum(probCluster(i, _));
  }
  // cout << "Probability Matrix Allocation: " << probCluster;
  NumericVector indC(K);
  for (int k = 0; k<K; k++) {
    indC(k) = k;
  }
  for (int i = 0; i<n; i++) {
    z(i) = csample_num(indC, 1, true, probCluster(i, _))(0);
  }
  return z;
}


// FCD mu
// [[Rcpp::export]]
arma::mat update_muD(double mu0, double p0, arma::mat prec, arma::irowvec z, arma::irowvec N, arma::mat X) {
  // La struttura è la seguente: un vettore media per ogni cluster e una matrice precisione
  // per ogni cluster. Dato che assumiamo indipendenza possiamo mettere la diagonale
  // all'interno di una matrice Kxd
  // p0 la precisione a priori 
  // mu il vettore delle medie a posteriori (un vettore per ogni cluster)
  // prec la precisione a posteriori 
  // z la matrice di allocazione
  // N la somma delle allocazioni
  int K = N.n_elem;
  int n = X.n_rows;
  int d = X.n_cols;
  arma::mat MuPosteriori(K, d);
  arma::mat muPost(K, d);
  arma::mat precPost(K, d);
  arma::mat sampMean(K, d);
  // Prior Setting (potrei anche non fare questa specifica, guardare precisione)
  arma::mat muPrior = mu0 * arma::ones<arma::mat>(K, d);
  arma::mat pPrior = p0 * arma::ones<arma::mat>(K, d);
  
  for (int k = 0; k<K; k++) {
    for (int j = 0; j<d; j++) {
      for (int i = 0; i<n; i++) {
        sampMean(k,j) = sampMean(k,j) + diracF(z(i), k)*X(i, j);
      }
    }
  }
  
  for (int j = 0; j<d; j++) {
    for (int k = 0; k<K; k++) {
      muPost(k,j) = (prec(k,j)*sampMean(k,j)+muPrior(k,j)*pPrior(k,j))/(prec(k,j)*N(k)+pPrior(k,j));
      precPost(k,j) = prec(k,j)*N(k)+pPrior(k,j);
      MuPosteriori(k,j) = R::rnorm(muPost(k,j), sqrt(1/precPost(k,j)));
    }
  }
  
  return MuPosteriori;
}


// FCD sigma
// [[Rcpp::export]]
arma::mat update_precD(double a0, double b0, arma::mat mu, arma::irowvec z, arma::irowvec N, arma::mat X) {
  // a0 iperparametro a priori della inverse gamma
  // b0 iperparametro a priori della inverse gamma
  // mu vettore delle medie a posteriori
  // z matrice di allocazione
  // N somme matrice allocazione
  // x dati
  int n = X.n_rows;
  int K = N.n_elem;
  int d = X.n_cols;
  arma::mat prec(K,d);
  arma::mat sumNum(K,d);
  // cout << "Medie: " << mu << "\n";
  for (int k = 0; k<K; k++) {
    for (int j = 0; j<d; j++) {
      for (int i = 0; i<n; i++) {
        sumNum(k,j) = sumNum(k,j) + diracF(z(i), k)*pow(X(i,j)-mu(k,j), 2);
      }
    }
  }
  for (int j = 0; j<d; j++) {
    for (int k = 0; k<K; k++) {
      double shape = a0+(N(k)/2);
      // compute the sum
      double scale = b0+sumNum(k,j)/2;
      prec(k,j) = callrgamma(1, shape, 1/scale)(0);
    }
  }
  return prec;
}


// Gibbs sampler d-dimension
// [[Rcpp::export]]
List DSSG(arma::mat X, arma::vec hyper, int K, int iteration, int burnin, int thin, String method) {
  auto start = std::chrono::high_resolution_clock::now();
  ////////////////////////////////////////////////////
  ////////////////// Initial settings ///////////////
  ///////////////////////////////////////////////////
  int nout = (iteration-burnin)/thin;
  int n = X.n_rows; // number of rows
  int d = X.n_cols; // number of columns
  NumericVector indC(K);
  for (int k = 0; k<K; k++) {
    indC(k) = k;
  }
  int idx = 0;
  // Z
  arma::imat Z(nout, n);
  // PI
  arma::mat PI(nout, K);
  // MU
  List MU(nout);
  // SIGMA
  List PREC(nout);
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
  arma::vec concPar = hyper(1) * arma::ones<arma::vec>(K);
  pi = rdirichlet_cpp(1, concPar);
  // MU
  arma::mat mu(K, d);
  for (int j = 0; j<d; j++) {
    for (int k = 0; k<K; k++) {
      mu(k,j) = R::rnorm(hyper(2), 1/sqrt(hyper(3)));
    }
  }
  // PRECISION
  arma::mat prec(K, d);
  for (int j = 0; j<d; j++) {
    for (int k = 0; k<K; k++) {
      prec(k,j) = callrgamma(1, hyper(4), 1/hyper(5))(0);
    }
  }
  ////////////////////////////////////////////////////
  /////////////////// Main Part /////////////////////
  ///////////////////////////////////////////////////
  for (int t = 0; t<iteration; t++) {
    // update z
    z = update_allocationD(pi, mu, prec, X);
    // compute N
    arma::irowvec N = sum_allocation(z, K);
    // update pi
    pi = update_pi(concPar.t(), N);
    // update params
    // mu
    mu = update_muD(hyper(2), hyper(3), prec, z, N, X);
    // precision
    prec = update_precD(hyper(4), hyper(5), mu, z, N, X);
    ////////////////////////////////////////////////////
    ///////////////// Store Results ///////////////////
    ///////////////////////////////////////////////////
    if(t%thin == 0 && t > burnin-1) {
      Z.row(idx) = z;
      PI.row(idx) = pi;
      MU[idx] = mu;
      PREC[idx] = prec;
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
                      Named("Mu") = MU,
                      Named("Precision") = PREC,
                      Named("Execution_Time") = duration/1000000);
  
}



