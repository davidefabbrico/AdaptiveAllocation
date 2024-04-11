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
  arma::vec rSum = rowSums(probCluster);
  for (int i = 0; i<n; i++) {
    probCluster(i, _) = probCluster(i, _) / rSum(i);
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

// Contingency Table
// [[Rcpp::export]]
arma::mat contMat(arma::irowvec eAlloc, arma::irowvec tAlloc) {
  int n = tAlloc.n_elem; 
  arma::irowvec nx = arma::unique(tAlloc);
  int nE = nx.n_elem;
  arma::mat cT(nE, nE);
  // cout << "TRUE " << tAlloc;
  // cout << "EST " << eAlloc;
  for (int i = 0; i<n; i++) {
    cT(tAlloc(i), eAlloc(i)) += 1;
  } 
  return(cT);
} 

int coefBinom(int n, int k) {
  if (k > n)
    return 0;
  if (k == 0 || k == n)
    return 1;
  return coefBinom(n - 1, k - 1) + coefBinom(n - 1, k);
}

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
  arma::vec rSum = rowSums(probCluster); 
  for (int i = 0; i<n; i++) {
    probCluster(i, _) = probCluster(i, _) / rSum(i);
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


// FCD sigma
// [[Rcpp::export]]
arma::mat update_precDH(double a0, double b0, arma::mat mu, arma::irowvec z, arma::irowvec N, arma::mat X) {
  // a0 iperparametro a priori della inverse gamma
  // b0 iperparametro a priori della inverse gamma
  // mu vettore delle medie a posteriori
  // z matrice di allocazione
  // N somme matrice allocazione
  // x dati
  int n = X.n_rows;
  int K = N.n_elem;
  int d = X.n_cols;
  arma::mat prec(K, d);
  arma::mat sumNum(K, d);
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

// FCD mu
// [[Rcpp::export]]
arma::mat update_muDH(arma::rowvec mu0, arma::mat p0, arma::mat prec, arma::irowvec z, arma::irowvec N, arma::mat X) {
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
  // Prior Setting (tutti i cluster partono con la stessa hyperpriori per la media)
  arma::mat muPrior(K, d);
  for (int k = 0; k<K; k++) {
    muPrior.row(k) = mu0;
  }
  
  for (int k = 0; k<K; k++) {
    for (int j = 0; j<d; j++) {
      for (int i = 0; i<n; i++) {
        sampMean(k,j) = sampMean(k,j) + diracF(z(i), k)*X(i, j);
      }
    } 
  } 
  
  for (int j = 0; j<d; j++) {
    for (int k = 0; k<K; k++) {
      muPost(k,j) = (prec(k,j)*sampMean(k,j)+muPrior(k,j)*p0(k,j))/(prec(k,j)*N(k)+p0(k,j));
      precPost(k,j) = prec(k,j)*N(k)+p0(k,j);
      MuPosteriori(k,j) = R::rnorm(muPost(k,j), sqrt(1/precPost(k,j)));
    } 
  } 
  
  return MuPosteriori;
}  


// FCD mu
// [[Rcpp::export]]
arma::mat update_muD(arma::rowvec mu0, double p0, arma::mat prec, arma::irowvec z, arma::irowvec N, arma::mat X) {
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
  // Prior Setting (tutti i cluster partono con la stessa hyperpriori per la media)
  arma::mat muPrior(K, d);
  for (int k = 0; k<K; k++) {
    muPrior.row(k) = mu0;
  }
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


// Gibbs sampler d-dimension
// [[Rcpp::export]]
List DSSG(arma::mat X, arma::vec hyper, int K, int iteration, int burnin, int thin, String method, arma::irowvec trueAllocation) {
  // precision and not variance!!
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
  // LOSS
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
  arma::vec concPar = hyper(1) * arma::ones<arma::vec>(K);
  pi = rdirichlet_cpp(1, concPar);
  // PRECISION
  arma::mat prec(K, d);
  for (int j = 0; j<d; j++) {
    for (int k = 0; k<K; k++) {
      prec(k,j) = callrgamma(1, hyper_prec_a, 1/hyper_prec_b)(0);
    }
  }
  // MU
  arma::mat mu(K, d);
  if (method == "EB") {
    double kP = 0.01;
    for (int j = 0; j<d; j++) {
      for (int k = 0; k<K; k++) {
        hyper_mu_prec(k,j) = prec(k,j);
        mu(k,j) = R::rnorm(hyper_mu_mean(j), 1/sqrt(kP*hyper_mu_prec(k,j)));
      }
    } 
  } else {
    for (int j = 0; j<d; j++) {
      for (int k = 0; k<K; k++) {
        mu(k,j) = R::rnorm(hyper_mu_mean(j), 1/sqrt(hyper_mu_prec(k,j)));
      }
    } 
  }
  double Loss = 0;
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
    mu = update_muDH(hyper_mu_mean, hyper_mu_prec, prec, z, N, X);
    // precision
    prec = update_precDH(hyper_prec_a, hyper_prec_b, mu, z, N, X);
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
        LOSS(idx) = Loss;
        // cout << "Matrice di Contingenza: \n" << contMat(z, trueAllocation) << "\n";
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

// [[Rcpp::export]]
arma::irowvec update_allocationRD(NumericMatrix probAllocation, arma::rowvec pi, arma::mat mu, arma::mat prec, arma::mat X, int m, arma::irowvec z, NumericVector alpha) {
  // pi sono le proporzioni di ogni cluster
  // mu è la media a posteriori
  // sigma è la varianza a posteriori
  // x dati
  // m sono quanti ne voglio aggiornare
  // z è il "vecchio" vettore di allocazione per ogni individuo
  int K = pi.n_elem; // numero di cluster
  int n = X.n_rows;
  int d = X.n_cols; // numero di covariate
  arma::vec vecTmp(d);
  // NumericMatrix probAllocation(n, K);
  NumericVector indI(n);
  for (int i = 0; i<n; i++) {
    indI(i) = i;
  }
  NumericVector indC(K);
  for (int k = 0; k<K; k++) {
    indC(k) = k;
  }
  NumericVector rI(m);
  rI = csample_num(indI, m, false, alpha); // without replace
  // cout << "Lunghezza rI " << rI.size() << "\n";
  // cout << "M " << m << "\n";
  for (int i = 0; i<m; i++) {
    for (int k = 0; k<K; k++) {
      // arma::vec vecTmp = arma::zeros<arma::vec>(d);
      for (int j = 0; j<d; j++) {
        vecTmp(j) = R::dnorm(X(rI[i],j), mu(k,j), sqrt(1/prec(k,j)), FALSE);
      }
      probAllocation(rI[i],k) = pi(k)*myProduct(vecTmp);
    }
  }
  // normalizzo le righe
  arma::vec rSum = rowSums(probAllocation);
  for (int i = 0; i<m; i++) {
    probAllocation(rI[i], _) = probAllocation(rI[i], _) / rSum(rI[i]);
  }
  // cout << "Somma per colonna Random " << colSums(probAllocation) << "\n";
  for (int i = 0; i<m; i++) {
    z(rI[i]) = csample_num(indC, 1, true, probAllocation(rI[i], _))(0);
  }
  // cout << "Matrice di Prob: " << probAllocation << "\n";
  // cout << "Allocation: " << z << "\n";
  return(z);
}


// Random Gibbs sampler d-dimensional
// [[Rcpp::export]]
List RSSG(arma::mat X, arma::vec hyper, int K, int m, int iteration, int burnin, int thin, String method, arma::irowvec trueAllocation) {
  // precision and not variance!!
  // m: how many observation I want to update
  
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
  // LOSS 
  arma::vec LOSS(nout);
  ////////////////////////////////////////////////////
  /////////////// Random Gibbs Sampler ///////////////
  ///////////////////////////////////////////////////
  // Alpha weight
  NumericVector alpha(n); // in this case the weights are setting equally to 1/n
  for (int i = 0; i<n; i++) {
    alpha(i) = 1.0/n;
  }
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
  arma::vec concPar = hyper(1) * arma::ones<arma::vec>(K);
  pi = rdirichlet_cpp(1, concPar);
  // PRECISION
  arma::mat prec(K, d);
  for (int j = 0; j<d; j++) {
    for (int k = 0; k<K; k++) {
      prec(k,j) = callrgamma(1, hyper_prec_a, 1/hyper_prec_b)(0);
    }
  }  
  // MU
  arma::mat mu(K, d);
  if (method == "EB") {
    double kP = 0.01;
    for (int j = 0; j<d; j++) {
      for (int k = 0; k<K; k++) {
        hyper_mu_prec(k,j) = prec(k,j);
        mu(k,j) = R::rnorm(hyper_mu_mean(j), 1/sqrt(kP*hyper_mu_prec(k,j)));
      }  
    } 
  } else {
    for (int j = 0; j<d; j++) {
      for (int k = 0; k<K; k++) {
        mu(k,j) = R::rnorm(hyper_mu_mean(j), 1/sqrt(hyper_mu_prec(k,j)));
      }
    }  
  } 
  // Store allocation
  List allocation(2);
  // Probability Matrix
  NumericMatrix probAllocation(n, K);
  // for (int i = 0; i<n; i++) {
  //   for (int k = 0; k<K; k++) {
  //     probAllocation(i,k) = 1.0/K;
  //   }
  // }
  ////////////////////////////////////////////////////
  /////////////////// Main Part /////////////////////
  ///////////////////////////////////////////////////
  
  double Loss = 0;
  for (int t = 0; t<iteration; t++) {
    // update probability matrix allocation and z
    z = update_allocationRD(probAllocation, pi, mu, prec, X, m, z, alpha);
    // compute N
    arma::irowvec N = sum_allocation(z, K);
    // update pi
    pi = update_pi(concPar.t(), N);
    // update params
    // mu
    mu = update_muDH(hyper_mu_mean, hyper_mu_prec, prec, z, N, X);
    // precision
    prec = update_precDH(hyper_prec_a, hyper_prec_b, mu, z, N, X);
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
        LOSS(idx) = Loss;
        // cout << "Matrice di Contingenza: \n" << contMat(z, trueAllocation) << "\n";
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


NumericMatrix comp_ProbAlloc(arma::rowvec pi, arma::mat mu, arma::mat prec, arma::mat X) {
  int K = pi.n_elem; // numero di cluster
  int n = X.n_rows; // numero di osservazioni
  int d = X.n_cols; // numero di covariate
  NumericMatrix probAllocation(n, K);
  arma::vec vecTmp(d);
  // STEP 1. Compute the probability
  for (int i = 0; i<n; i++) {
    for (int k = 0; k<K; k++) {
      // arma::vec vecTmp = arma::zeros<arma::vec>(d);
      for (int j = 0; j<d; j++) {
        vecTmp(j) = R::dnorm(X(i,j), mu(k,j), sqrt(1/prec(k,j)), FALSE);
      }
      probAllocation(i,k) = pi(k)*myProduct(vecTmp);
    } 
  }
  // cout << "Prob Allocation " << probAllocation << "\n";
  // normalizzo le righe
  arma::vec rSum = rowSums(probAllocation);
  for (int i = 0; i<n; i++) {
    probAllocation(i, _) = probAllocation(i, _) / rSum(i);
  }
  // cout << "Somma per colonna Adaptive " << colSums(probAllocation) << "\n";
  return(probAllocation);
}

// [[Rcpp::export]]
NumericVector GSIndex(NumericMatrix probAllocation) {
  int K = probAllocation.ncol(); // numero di cluster
  int n = probAllocation.nrow(); // numero di osservazioni
  NumericVector GSimpson(n);
  for (int i = 0; i<n; i++) {
    for (int k = 0; k<K; k++) {
      GSimpson(i) = GSimpson(i) + pow(probAllocation(i, k), 2);
    }
    GSimpson(i) = 1 - GSimpson(i);
  }
  // Normalize
  double sumSimp = sum(GSimpson);
  // check the entropy values
  for (int i = 0; i<n; i++) {
    GSimpson(i) = GSimpson(i) / sumSimp;
  } 
  return(GSimpson);
}

// [[Rcpp::export]]
NumericVector entropy(NumericMatrix probAllocation) {
  int K = probAllocation.ncol(); // numero di cluster
  int n = probAllocation.nrow(); // numero di osservazioni
  // STEP 2. Entropy
  NumericVector Entropy(n);
  for (int i = 0; i<n; i++) {
    for (int k = 0; k<K; k++) {
      Entropy(i) = Entropy(i) + probAllocation(i, k)*log(probAllocation(i, k));
    }
    Entropy(i) = -Entropy(i);
  }
  // STEP 3. Normalize the entropy
  double sommaEntropia = sum(Entropy);
  // check the entropy values
  for (int i = 0; i<n; i++) {
    Entropy(i) = Entropy(i) / sommaEntropia;
  } 
  // cout << "Somma Entropia: " << sum(Entropy) << "\n";
  return(Entropy);
}


// [[Rcpp::export]]
NumericVector update_alpha(int iter, double gamma, NumericVector alpha_prec, NumericVector Diversity) {
  int n = alpha_prec.size();
  NumericVector constVal(n); // in this case the weights are setting equally to 1/n
  NumericVector alpha(n);
  for (int i = 0; i<n; i++) {
    constVal(i) = 1.0/n;
  }
  // cout << "Questo è alpha prima: " << alpha_prec << "\n";
  // cout << Diversity << "\n";
  if (iter == 0 || iter == 1) {
    alpha = gamma*Diversity+(1-gamma)*constVal;// constVal;
  } else {
    alpha = alpha_prec*((iter-1.0)/iter)+(1.0/iter)*(gamma*Diversity+(1-gamma)*constVal);
  }
  // NumericVector a = (1.0/iter)*(gamma*Diversity+(1-gamma)*constVal);
  // cout << "Questo è quello che gli aggiungo: " << a << "\n";
  // cout << "Questo è alpha dopo: " << alpha << "\n";
  // Normalizzo alpha
  // double sommaAlpha = sum(alpha);
  // // check the entropy values
  // for (int i = 0; i<n; i++) {
  //   alpha(i) = alpha(i) / sommaAlpha;
  // } 
  return(alpha);
}

// [[Rcpp::export]]
arma::irowvec diversity_allocation(NumericMatrix probAllocation, int m, arma::irowvec z, NumericVector alpha) {
  // pi sono le proporzioni di ogni cluster
  // mu è la media a posteriori
  // sigma è la varianza a posteriori
  // x dati
  int K = probAllocation.ncol(); // numero di cluster
  int n = probAllocation.nrow(); // numero di osservazioni
  // Update the allocation
  NumericVector indI(n);
  for (int i = 0; i<n; i++) {
    indI(i) = i;
  }
  NumericVector indC(K);
  for (int k = 0; k<K; k++) {
    indC(k) = k;
  } 
  NumericVector rI(m);
  rI = csample_num(indI, m, false, alpha); // without replace
  // cout << rI << "\n";
  for (int i = 0; i<m; i++) {
    z(rI[i]) = csample_num(indC, 1, true, probAllocation(rI[i], _))(0);
  }
  return(z);
}


// Random Gibbs sampler d-dimensional
// [[Rcpp::export]]
List DiversityGibbsSamp(arma::mat X, arma::vec hyper, int K, int m, int iteration, int burnin, int thin, String method, double gamma, String diversity, arma::irowvec trueAllocation) {
  // precision and not variance!!
  // m: how many observation I want to update
  // start time
  auto start = std::chrono::high_resolution_clock::now();
  ////////////////////////////////////////////////////
  ////////////////// Initial settings ///////////////
  ///////////////////////////////////////////////////
  int nout = (iteration-burnin)/thin; // number of sample a posteriori
  int n = X.n_rows; // number of rows
  int d = X.n_cols; // number of columns
  NumericVector indC(K); // indici per i cluster
  for (int k = 0; k<K; k++) {
    indC(k) = k;
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
  // Alpha weight
  NumericVector alpha(n); // start from 1/n
  for (int i = 0; i<n; i++) {
    alpha(i) = 1.0/n; // sample
  } 
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
    // hyper_mu_mean = mean(X, 0);
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
  arma::vec concPar = hyper(1) * arma::ones<arma::vec>(K);
  pi = rdirichlet_cpp(1, concPar);
  // PRECISION
  arma::mat prec(K, d);
  for (int j = 0; j<d; j++) {
    for (int k = 0; k<K; k++) {
      prec(k,j) = callrgamma(1, hyper_prec_a, 1/hyper_prec_b)(0);
    }
  } 
  // MU
  arma::mat mu(K, d);
  if (method == "EB") {
    double kP = 0.01;
    for (int j = 0; j<d; j++) {
      for (int k = 0; k<K; k++) {
        hyper_mu_prec(k,j) = prec(k,j);
        mu(k,j) = R::rnorm(hyper_mu_mean(j), 1/sqrt(kP*hyper_mu_prec(k,j)));
      } 
    } 
  } else {
    for (int j = 0; j<d; j++) {
      for (int k = 0; k<K; k++) {
        mu(k,j) = R::rnorm(hyper_mu_mean(j), 1/sqrt(hyper_mu_prec(k,j)));
      }
    } 
  } 
  // cout << "Mean \n " << hyper_mu_mean << "\n";
  // cout  << "Variance \n " << sqrt(hyper_prec) << "\n";
  // Store allocation
  List allocation(2);
  // Probability Matrix
  NumericMatrix probAllocation(n, K);
  NumericVector Diversity(n);
  arma::irowvec N(K);
  // Loss index
  double Loss = 0;
  ////////////////////////////////////////////////////
  /////////////////// Main Part /////////////////////
  ///////////////////////////////////////////////////
  for (int t = 0; t<iteration; t++) {
    // update probability 
    probAllocation = comp_ProbAlloc(pi, mu, prec, X);
    if (diversity == "Entropy") {
      // compute entropy
      Diversity = entropy(probAllocation); 
    } else if (diversity == "Gini-Simpson") {
      Diversity = GSIndex(probAllocation);
    } else {
      cout << "You need to specify a Diversity Index!" << "\n";
    }
    // update alpha
    alpha = update_alpha(t, gamma, alpha, Diversity);
    // update z
    z = diversity_allocation(probAllocation, m, z, alpha);
    // compute N
    N = sum_allocation(z, K);
    // update pi
    pi = update_pi(concPar.t(), N);
    // update params
    // mu
    mu = update_muDH(hyper_mu_mean, hyper_mu_prec, prec, z, N, X);
    // precision
    prec = update_precDH(hyper_prec_a, hyper_prec_b, mu, z, N, X);
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
        LOSS(idx) = Loss;
        // cout << "Matrice di Contingenza: \n" << contMat(z, trueAllocation) << "\n";
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
                        Named("Diversity") = D,
                        Named("Proportion_Parameters") = PI,
                        Named("Mu") = MU,
                        Named("Precision") = PREC,
                        Named("Loss") = LOSS,
                        Named("Alpha") = ALPHA,
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

