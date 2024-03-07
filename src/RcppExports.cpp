// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// callrgamma
NumericVector callrgamma(int n, double shape, double scale);
RcppExport SEXP _AdaptiveAllocation_callrgamma(SEXP nSEXP, SEXP shapeSEXP, SEXP scaleSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< double >::type shape(shapeSEXP);
    Rcpp::traits::input_parameter< double >::type scale(scaleSEXP);
    rcpp_result_gen = Rcpp::wrap(callrgamma(n, shape, scale));
    return rcpp_result_gen;
END_RCPP
}
// rmultinom_1
IntegerVector rmultinom_1(unsigned int& size, NumericVector& probs, unsigned int& N);
RcppExport SEXP _AdaptiveAllocation_rmultinom_1(SEXP sizeSEXP, SEXP probsSEXP, SEXP NSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int& >::type size(sizeSEXP);
    Rcpp::traits::input_parameter< NumericVector& >::type probs(probsSEXP);
    Rcpp::traits::input_parameter< unsigned int& >::type N(NSEXP);
    rcpp_result_gen = Rcpp::wrap(rmultinom_1(size, probs, N));
    return rcpp_result_gen;
END_RCPP
}
// rmultinom_rcpp
IntegerMatrix rmultinom_rcpp(unsigned int& n, unsigned int& size, NumericVector& probs);
RcppExport SEXP _AdaptiveAllocation_rmultinom_rcpp(SEXP nSEXP, SEXP sizeSEXP, SEXP probsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< unsigned int& >::type n(nSEXP);
    Rcpp::traits::input_parameter< unsigned int& >::type size(sizeSEXP);
    Rcpp::traits::input_parameter< NumericVector& >::type probs(probsSEXP);
    rcpp_result_gen = Rcpp::wrap(rmultinom_rcpp(n, size, probs));
    return rcpp_result_gen;
END_RCPP
}
// rdirichlet_cpp
arma::mat rdirichlet_cpp(int num_samples, arma::vec alpha_m);
RcppExport SEXP _AdaptiveAllocation_rdirichlet_cpp(SEXP num_samplesSEXP, SEXP alpha_mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type num_samples(num_samplesSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha_m(alpha_mSEXP);
    rcpp_result_gen = Rcpp::wrap(rdirichlet_cpp(num_samples, alpha_m));
    return rcpp_result_gen;
END_RCPP
}
// update_allocation
arma::mat update_allocation(arma::vec pi, double mu, double sigma, arma::vec x);
RcppExport SEXP _AdaptiveAllocation_update_allocation(SEXP piSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type pi(piSEXP);
    Rcpp::traits::input_parameter< double >::type mu(muSEXP);
    Rcpp::traits::input_parameter< double >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(update_allocation(pi, mu, sigma, x));
    return rcpp_result_gen;
END_RCPP
}
// update_pi
arma::rowvec update_pi(arma::vec alpha, arma::vec N);
RcppExport SEXP _AdaptiveAllocation_update_pi(SEXP alphaSEXP, SEXP NSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type N(NSEXP);
    rcpp_result_gen = Rcpp::wrap(update_pi(alpha, N));
    return rcpp_result_gen;
END_RCPP
}
// update_mu
arma::vec update_mu(double mu0, double s0, arma::vec sigma, arma::mat z, arma::vec N, arma::vec x);
RcppExport SEXP _AdaptiveAllocation_update_mu(SEXP mu0SEXP, SEXP s0SEXP, SEXP sigmaSEXP, SEXP zSEXP, SEXP NSEXP, SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type mu0(mu0SEXP);
    Rcpp::traits::input_parameter< double >::type s0(s0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type z(zSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type N(NSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(update_mu(mu0, s0, sigma, z, N, x));
    return rcpp_result_gen;
END_RCPP
}
// update_sigma
arma::vec update_sigma(double a0, double b0, arma::vec mu, arma::mat z, arma::vec N, arma::vec x);
RcppExport SEXP _AdaptiveAllocation_update_sigma(SEXP a0SEXP, SEXP b0SEXP, SEXP muSEXP, SEXP zSEXP, SEXP NSEXP, SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type a0(a0SEXP);
    Rcpp::traits::input_parameter< double >::type b0(b0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu(muSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type z(zSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type N(NSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(update_sigma(a0, b0, mu, z, N, x));
    return rcpp_result_gen;
END_RCPP
}
// SSG
List SSG(arma::vec x, arma::vec hyper, int K, int iteration, int burnin, int thin);
RcppExport SEXP _AdaptiveAllocation_SSG(SEXP xSEXP, SEXP hyperSEXP, SEXP KSEXP, SEXP iterationSEXP, SEXP burninSEXP, SEXP thinSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type hyper(hyperSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type iteration(iterationSEXP);
    Rcpp::traits::input_parameter< int >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< int >::type thin(thinSEXP);
    rcpp_result_gen = Rcpp::wrap(SSG(x, hyper, K, iteration, burnin, thin));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_AdaptiveAllocation_callrgamma", (DL_FUNC) &_AdaptiveAllocation_callrgamma, 3},
    {"_AdaptiveAllocation_rmultinom_1", (DL_FUNC) &_AdaptiveAllocation_rmultinom_1, 3},
    {"_AdaptiveAllocation_rmultinom_rcpp", (DL_FUNC) &_AdaptiveAllocation_rmultinom_rcpp, 3},
    {"_AdaptiveAllocation_rdirichlet_cpp", (DL_FUNC) &_AdaptiveAllocation_rdirichlet_cpp, 2},
    {"_AdaptiveAllocation_update_allocation", (DL_FUNC) &_AdaptiveAllocation_update_allocation, 4},
    {"_AdaptiveAllocation_update_pi", (DL_FUNC) &_AdaptiveAllocation_update_pi, 2},
    {"_AdaptiveAllocation_update_mu", (DL_FUNC) &_AdaptiveAllocation_update_mu, 6},
    {"_AdaptiveAllocation_update_sigma", (DL_FUNC) &_AdaptiveAllocation_update_sigma, 6},
    {"_AdaptiveAllocation_SSG", (DL_FUNC) &_AdaptiveAllocation_SSG, 6},
    {NULL, NULL, 0}
};

RcppExport void R_init_AdaptiveAllocation(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
