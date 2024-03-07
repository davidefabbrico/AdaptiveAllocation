#include <RcppArmadillo.h>
#include <math.h>
#include <R.h>
#include <Rmath.h>
#include <stdlib.h> 
#include <chrono>

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

// come campionare da una distribuzione Gamma
// [[Rcpp::export]]
NumericVector callrgamma(int n, double shape, double scale) { 
  // n sono i campioni che voglio
  // shape e scale sono i parametri della gamma
  return(rgamma(n, shape, scale)); 
}

// come campionare da una distribuzione multinomiale
// [[Rcpp::export]]
IntegerVector rmultinom_1(unsigned int &size, NumericVector &probs, unsigned int &N) {
  IntegerVector outcome(N);
  rmultinom(size, probs.begin(), N, outcome.begin());
  return outcome;
}

// [[Rcpp::export]]
IntegerMatrix rmultinom_rcpp(unsigned int &n, unsigned int &size, NumericVector &probs) {
  unsigned int N = probs.length();
  IntegerMatrix sim(N, n);
  for (unsigned int i = 0; i < n; i++) {
    sim(_,i) = rmultinom_1(size, probs, N);
  }
  return sim;
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
      double cur = Rcpp::as<double>(callrgamma(1, alpha_m[j], 1.0));
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

// come calcolare N (somma delle allocazioni)
arma::vec sum_allocation(arma::mat z) {
  // z è la matrice delle allocazioni
  int K = z.n_cols;
  arma::vec sAllocation(K);
  sAllocation = arma::sum(z, 0);
  return sAllocation;
}

// FCD z
// [[Rcpp::export]]
arma::mat update_allocation(arma::vec pi, double mu, double sigma, arma::vec x) {
  // pi sono le proporzioni di ogni cluster
  // mu è la media a posteriori
  // sigma è la varianza a posteriori
  // x dati
  int K = pi.n_elem; // numero di cluster
  int n = x.n_elem; // numero di osservazioni
  arma::mat probCluster(n, K);
  arma::mat z(n, K);
  // calcola il denominatore della probabilità (vettore i-dim)
  arma::rowvec denSum = arma::zeros<arma::rowvec>(n);
  for (int i = 0; i<n; i++) {
    for (int kp = 0; kp<K; kp++) {
      denSum(i) = denSum(i) + pi(kp)*R::dnorm(x(i), mu, sigma, FALSE);
    }
  }
  for (int k = 0; k<K; k++) {
    for (int i = 0; i<n; i++) {
      probCluster(i,k) = pi(k)*R::dnorm(x(i), mu, sigma, FALSE) / denSum(i);
    }
  }
  // li aggiorno per riga (faccio variare le colonne)
  // for (int i = 0; i<n; i++) {
  //   z.row(i) = rmultinom_rcpp(1, K, probCluster.row(i));
  // }
  return z;
}

// FCD pi
// [[Rcpp::export]]
arma::rowvec update_pi(arma::vec alpha, arma::vec N) {
  // a is the concentration parameter
  // N is the sum of allocation
  int K = N.n_elem;
  arma::vec n_alpha(K);
  // invece di calcolarmelo per ogni funzione glielo passo come parametro in input
  // arma::vec N = sum_allocation(z);
  arma::vec parDirich = alpha/K + N;
  return rdirichlet_cpp(1, parDirich);
}

// FCD mu
// [[Rcpp::export]]
arma::vec update_mu(double mu0, double s0, arma::vec sigma, arma::mat z, arma::vec N, arma::vec x) {
  // s0 la varianza a priori
  // mu la media a posteriori
  // sigma la varianza a posteriori
  // z la matrice di allocazione
  // N la somma delle allocazioni
  int K = z.n_cols;
  int n = z.n_rows;
  arma::vec MuPosteriori(K);
  arma::vec muPost(K);
  arma::vec sigmaPost(K);
  arma::vec sampMean(K);
  for (int k = 0; k<K; k++) {
    for (int i = 0; i<n; i++) {
      sampMean(k) = sampMean(k) + z(i,k)*x(i)/N(k);
    }
    muPost(k) = (N(k)*s0*sampMean(k)+mu0*sigma(k))/(sigma(k)+N(k)*s0);
    sigmaPost(k) = (s0*sigma(k))/(sigma(k)+N(k)*s0);
    MuPosteriori(k) = R::rnorm(muPost(k), sigmaPost(k));
  }
  return MuPosteriori;
}


// FCD sigma
// [[Rcpp::export]]
arma::vec update_sigma(double a0, double b0, arma::vec mu, arma::mat z, arma::vec N, arma::vec x) {
  // a0 iperparametro a priori della inverse gamma
  // b0 iperparametro a priori della inverse gamma
  // mu vettore delle medie a posteriori
  // z matrice di allocazione
  // N somme matrice allocazione
  // x dati
  int n = x.n_rows;
  int K = N.n_elem;
  arma::vec sigma(K);
  double sumNum;
  for (int k = 0; k<K; k++) {
    double shape = a0+N(k)/2;
    // compute the sum
    sumNum = 0;
    for (int i = 0; i<n; i++) {
      sumNum = sumNum + z(i,k)*pow((x(i)-mu(k)), 2);
    }
    double scale = b0+sumNum/2;
    sigma(k) = 1/callrgamma(1, shape, scale)(0);
  }
  return sigma;
}

// Gibbs sampler
// [[Rcpp::export]]
List SSG(arma::vec x, arma::vec hyper, int K, int iteration, int burnin, int thin) {
  auto start = std::chrono::high_resolution_clock::now();
  ////////////////////////////////////////////////////
  ////////////////// Initial settings ///////////////
  ///////////////////////////////////////////////////
  int nout = (iteration-burnin)/thin;
  int n = x.n_elem;
  int idx = 0;
  // Z
  List Z(nout);
  // PI
  arma::mat PI(nout, K);
  // MU
  arma::mat MU(nout, K);
  // SIGMA
  arma::mat SIGMA(nout, K);
  ////////////////////////////////////////////////////
  ////////////////// Initial value //////////////////
  ///////////////////////////////////////////////////
  // Z
  arma::mat z(n, K);
  arma::vec probC = hyper(1)/K * arma::ones<arma::rowvec>(K);
  for (int i = 0; i<n; i++) {
    // Rcpp::NumericVector probCl = Rcpp::wrap(probC);
    // z.row(i) = Rcpp::transpose(rmultinom_rcpp(1, K, probC));
  }
  // PI
  arma::vec pi(K);
  arma::vec concPar = hyper(2) * arma::ones<arma::rowvec>(K);
  pi = rdirichlet_cpp(1, concPar);
  // MU
  arma::vec mu(K);
  for (int k = 0; k<K; k++) {
    mu(k) = R::rnorm(hyper(3), hyper(4));
  }
  // SIGMA
  arma::vec sigma(K);
  sigma = 1/callrgamma(K, hyper(5), hyper(6));
  ////////////////////////////////////////////////////
  /////////////////// Main Part /////////////////////
  ///////////////////////////////////////////////////
  for (int t = 0; t<iteration; t++) {
    // update z
    // z = update_allocation(pi, mu, sigma, x);
    // compute N
    arma::vec N = sum_allocation(z);
    // update pi
    pi = update_pi(concPar, N);
    // update params
    // mu
    mu = update_mu(hyper(3), hyper(4), sigma, z, N, x);
    // sigma
    sigma = update_sigma(hyper(5), hyper(6), mu, z, N, x);
    ////////////////////////////////////////////////////
    ///////////////// Store Results ///////////////////
    ///////////////////////////////////////////////////
    if(t%thin == 0 && t > burnin-1) {
      Z(idx) = z;
      PI.row(idx) = pi;
      MU.row(idx) = mu;
      SIGMA.row(idx) = sigma;
      idx = idx + 1;
    }
    if (t%500 == 0 && t > 0) {
      std::cout << "Iteration: " << t << " (of " << iteration << ")\n";
    }
  }
  std::cout << "End MCMC!\n";
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  return List::create(Named("Allocation_Matrix") = Z,
                      Named("Proportion_Parameters") = PI,
                      Named("Mu") = MU,
                      Named("Sigma") = SIGMA,
                      Named("Execution_Time") = duration/1000000);
}





