// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// sum_allocation
arma::irowvec sum_allocation(arma::irowvec z, int K);
RcppExport SEXP _AdaptiveAllocation_sum_allocation(SEXP zSEXP, SEXP KSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::irowvec >::type z(zSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    rcpp_result_gen = Rcpp::wrap(sum_allocation(z, K));
    return rcpp_result_gen;
END_RCPP
}
// callrgamma
Rcpp::NumericVector callrgamma(int n, double shape, double scale);
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
// csample_num
NumericVector csample_num(NumericVector x, int size, bool replace, NumericVector prob);
RcppExport SEXP _AdaptiveAllocation_csample_num(SEXP xSEXP, SEXP sizeSEXP, SEXP replaceSEXP, SEXP probSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type size(sizeSEXP);
    Rcpp::traits::input_parameter< bool >::type replace(replaceSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type prob(probSEXP);
    rcpp_result_gen = Rcpp::wrap(csample_num(x, size, replace, prob));
    return rcpp_result_gen;
END_RCPP
}
// diracF
int diracF(int a, int b);
RcppExport SEXP _AdaptiveAllocation_diracF(SEXP aSEXP, SEXP bSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type a(aSEXP);
    Rcpp::traits::input_parameter< int >::type b(bSEXP);
    rcpp_result_gen = Rcpp::wrap(diracF(a, b));
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
arma::irowvec update_allocation(arma::rowvec pi, arma::vec mu, arma::vec prec, arma::vec x);
RcppExport SEXP _AdaptiveAllocation_update_allocation(SEXP piSEXP, SEXP muSEXP, SEXP precSEXP, SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::rowvec >::type pi(piSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu(muSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prec(precSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(update_allocation(pi, mu, prec, x));
    return rcpp_result_gen;
END_RCPP
}
// update_pi
arma::rowvec update_pi(arma::rowvec alpha, arma::irowvec N);
RcppExport SEXP _AdaptiveAllocation_update_pi(SEXP alphaSEXP, SEXP NSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::rowvec >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< arma::irowvec >::type N(NSEXP);
    rcpp_result_gen = Rcpp::wrap(update_pi(alpha, N));
    return rcpp_result_gen;
END_RCPP
}
// update_mu
arma::vec update_mu(double mu0, double p0, arma::vec prec, arma::irowvec z, arma::irowvec N, arma::vec x);
RcppExport SEXP _AdaptiveAllocation_update_mu(SEXP mu0SEXP, SEXP p0SEXP, SEXP precSEXP, SEXP zSEXP, SEXP NSEXP, SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type mu0(mu0SEXP);
    Rcpp::traits::input_parameter< double >::type p0(p0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type prec(precSEXP);
    Rcpp::traits::input_parameter< arma::irowvec >::type z(zSEXP);
    Rcpp::traits::input_parameter< arma::irowvec >::type N(NSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(update_mu(mu0, p0, prec, z, N, x));
    return rcpp_result_gen;
END_RCPP
}
// update_prec
arma::vec update_prec(double a0, double b0, arma::vec mu, arma::irowvec z, arma::irowvec N, arma::vec x);
RcppExport SEXP _AdaptiveAllocation_update_prec(SEXP a0SEXP, SEXP b0SEXP, SEXP muSEXP, SEXP zSEXP, SEXP NSEXP, SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type a0(a0SEXP);
    Rcpp::traits::input_parameter< double >::type b0(b0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu(muSEXP);
    Rcpp::traits::input_parameter< arma::irowvec >::type z(zSEXP);
    Rcpp::traits::input_parameter< arma::irowvec >::type N(NSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(update_prec(a0, b0, mu, z, N, x));
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
// update_allocationD
arma::irowvec update_allocationD(arma::rowvec pi, arma::mat mu, arma::mat prec, arma::mat X);
RcppExport SEXP _AdaptiveAllocation_update_allocationD(SEXP piSEXP, SEXP muSEXP, SEXP precSEXP, SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::rowvec >::type pi(piSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type mu(muSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type prec(precSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(update_allocationD(pi, mu, prec, X));
    return rcpp_result_gen;
END_RCPP
}
// update_muD
arma::vec update_muD(arma::vec mu0, arma::mat p0, arma::mat prec, arma::irowvec z, arma::irowvec N, arma::mat X);
RcppExport SEXP _AdaptiveAllocation_update_muD(SEXP mu0SEXP, SEXP p0SEXP, SEXP precSEXP, SEXP zSEXP, SEXP NSEXP, SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type mu0(mu0SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type p0(p0SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type prec(precSEXP);
    Rcpp::traits::input_parameter< arma::irowvec >::type z(zSEXP);
    Rcpp::traits::input_parameter< arma::irowvec >::type N(NSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(update_muD(mu0, p0, prec, z, N, X));
    return rcpp_result_gen;
END_RCPP
}
// update_precD
arma::mat update_precD(double a0, double b0, arma::mat mu, arma::irowvec z, arma::irowvec N, arma::mat X);
RcppExport SEXP _AdaptiveAllocation_update_precD(SEXP a0SEXP, SEXP b0SEXP, SEXP muSEXP, SEXP zSEXP, SEXP NSEXP, SEXP XSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type a0(a0SEXP);
    Rcpp::traits::input_parameter< double >::type b0(b0SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type mu(muSEXP);
    Rcpp::traits::input_parameter< arma::irowvec >::type z(zSEXP);
    Rcpp::traits::input_parameter< arma::irowvec >::type N(NSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    rcpp_result_gen = Rcpp::wrap(update_precD(a0, b0, mu, z, N, X));
    return rcpp_result_gen;
END_RCPP
}
// DSSG
List DSSG(arma::mat X, arma::vec hyper, int K, int iteration, int burnin, int thin);
RcppExport SEXP _AdaptiveAllocation_DSSG(SEXP XSEXP, SEXP hyperSEXP, SEXP KSEXP, SEXP iterationSEXP, SEXP burninSEXP, SEXP thinSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type hyper(hyperSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type iteration(iterationSEXP);
    Rcpp::traits::input_parameter< int >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< int >::type thin(thinSEXP);
    rcpp_result_gen = Rcpp::wrap(DSSG(X, hyper, K, iteration, burnin, thin));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_AdaptiveAllocation_sum_allocation", (DL_FUNC) &_AdaptiveAllocation_sum_allocation, 2},
    {"_AdaptiveAllocation_callrgamma", (DL_FUNC) &_AdaptiveAllocation_callrgamma, 3},
    {"_AdaptiveAllocation_csample_num", (DL_FUNC) &_AdaptiveAllocation_csample_num, 4},
    {"_AdaptiveAllocation_diracF", (DL_FUNC) &_AdaptiveAllocation_diracF, 2},
    {"_AdaptiveAllocation_rdirichlet_cpp", (DL_FUNC) &_AdaptiveAllocation_rdirichlet_cpp, 2},
    {"_AdaptiveAllocation_update_allocation", (DL_FUNC) &_AdaptiveAllocation_update_allocation, 4},
    {"_AdaptiveAllocation_update_pi", (DL_FUNC) &_AdaptiveAllocation_update_pi, 2},
    {"_AdaptiveAllocation_update_mu", (DL_FUNC) &_AdaptiveAllocation_update_mu, 6},
    {"_AdaptiveAllocation_update_prec", (DL_FUNC) &_AdaptiveAllocation_update_prec, 6},
    {"_AdaptiveAllocation_SSG", (DL_FUNC) &_AdaptiveAllocation_SSG, 6},
    {"_AdaptiveAllocation_update_allocationD", (DL_FUNC) &_AdaptiveAllocation_update_allocationD, 4},
    {"_AdaptiveAllocation_update_muD", (DL_FUNC) &_AdaptiveAllocation_update_muD, 6},
    {"_AdaptiveAllocation_update_precD", (DL_FUNC) &_AdaptiveAllocation_update_precD, 6},
    {"_AdaptiveAllocation_DSSG", (DL_FUNC) &_AdaptiveAllocation_DSSG, 6},
    {NULL, NULL, 0}
};

RcppExport void R_init_AdaptiveAllocation(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
