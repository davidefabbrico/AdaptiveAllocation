// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// customMatrix
arma::mat customMatrix(arma::irowvec z);
RcppExport SEXP _AdaptiveAllocation_customMatrix(SEXP zSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::irowvec >::type z(zSEXP);
    rcpp_result_gen = Rcpp::wrap(customMatrix(z));
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
// summary_Posterior
arma::mat summary_Posterior(arma::imat z);
RcppExport SEXP _AdaptiveAllocation_summary_Posterior(SEXP zSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::imat >::type z(zSEXP);
    rcpp_result_gen = Rcpp::wrap(summary_Posterior(z));
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
// BinderLoss
double BinderLoss(arma::irowvec eAlloc, arma::irowvec tAlloc);
RcppExport SEXP _AdaptiveAllocation_BinderLoss(SEXP eAllocSEXP, SEXP tAllocSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::irowvec >::type eAlloc(eAllocSEXP);
    Rcpp::traits::input_parameter< arma::irowvec >::type tAlloc(tAllocSEXP);
    rcpp_result_gen = Rcpp::wrap(BinderLoss(eAlloc, tAlloc));
    return rcpp_result_gen;
END_RCPP
}
// coefBinom
int coefBinom(int n, int k);
RcppExport SEXP _AdaptiveAllocation_coefBinom(SEXP nSEXP, SEXP kSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    rcpp_result_gen = Rcpp::wrap(coefBinom(n, k));
    return rcpp_result_gen;
END_RCPP
}
// ari
double ari(arma::irowvec eAlloc, arma::irowvec tAlloc);
RcppExport SEXP _AdaptiveAllocation_ari(SEXP eAllocSEXP, SEXP tAllocSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::irowvec >::type eAlloc(eAllocSEXP);
    Rcpp::traits::input_parameter< arma::irowvec >::type tAlloc(tAllocSEXP);
    rcpp_result_gen = Rcpp::wrap(ari(eAlloc, tAlloc));
    return rcpp_result_gen;
END_RCPP
}
// myProduct
double myProduct(arma::vec a);
RcppExport SEXP _AdaptiveAllocation_myProduct(SEXP aSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type a(aSEXP);
    rcpp_result_gen = Rcpp::wrap(myProduct(a));
    return rcpp_result_gen;
END_RCPP
}
// SSG
List SSG(arma::mat X, arma::vec hyper, int K, int iteration, int burnin, int thin, String method, arma::irowvec trueAllocation);
RcppExport SEXP _AdaptiveAllocation_SSG(SEXP XSEXP, SEXP hyperSEXP, SEXP KSEXP, SEXP iterationSEXP, SEXP burninSEXP, SEXP thinSEXP, SEXP methodSEXP, SEXP trueAllocationSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type hyper(hyperSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type iteration(iterationSEXP);
    Rcpp::traits::input_parameter< int >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< int >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< String >::type method(methodSEXP);
    Rcpp::traits::input_parameter< arma::irowvec >::type trueAllocation(trueAllocationSEXP);
    rcpp_result_gen = Rcpp::wrap(SSG(X, hyper, K, iteration, burnin, thin, method, trueAllocation));
    return rcpp_result_gen;
END_RCPP
}
// RSSG
List RSSG(arma::mat X, arma::vec hyper, int K, int m, int iteration, int burnin, int thin, String method, arma::irowvec trueAllocation);
RcppExport SEXP _AdaptiveAllocation_RSSG(SEXP XSEXP, SEXP hyperSEXP, SEXP KSEXP, SEXP mSEXP, SEXP iterationSEXP, SEXP burninSEXP, SEXP thinSEXP, SEXP methodSEXP, SEXP trueAllocationSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type hyper(hyperSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< int >::type iteration(iterationSEXP);
    Rcpp::traits::input_parameter< int >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< int >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< String >::type method(methodSEXP);
    Rcpp::traits::input_parameter< arma::irowvec >::type trueAllocation(trueAllocationSEXP);
    rcpp_result_gen = Rcpp::wrap(RSSG(X, hyper, K, m, iteration, burnin, thin, method, trueAllocation));
    return rcpp_result_gen;
END_RCPP
}
// JS_distance
double JS_distance(NumericVector p, NumericVector q);
RcppExport SEXP _AdaptiveAllocation_JS_distance(SEXP pSEXP, SEXP qSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type p(pSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type q(qSEXP);
    rcpp_result_gen = Rcpp::wrap(JS_distance(p, q));
    return rcpp_result_gen;
END_RCPP
}
// DiversityGibbsSamp
List DiversityGibbsSamp(arma::mat X, arma::vec hyper, int K, int m, int iteration, int burnin, int thin, String method, double gamma, arma::irowvec trueAllocation, bool adaptiveGamma, double q);
RcppExport SEXP _AdaptiveAllocation_DiversityGibbsSamp(SEXP XSEXP, SEXP hyperSEXP, SEXP KSEXP, SEXP mSEXP, SEXP iterationSEXP, SEXP burninSEXP, SEXP thinSEXP, SEXP methodSEXP, SEXP gammaSEXP, SEXP trueAllocationSEXP, SEXP adaptiveGammaSEXP, SEXP qSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type hyper(hyperSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< int >::type iteration(iterationSEXP);
    Rcpp::traits::input_parameter< int >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< int >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< String >::type method(methodSEXP);
    Rcpp::traits::input_parameter< double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< arma::irowvec >::type trueAllocation(trueAllocationSEXP);
    Rcpp::traits::input_parameter< bool >::type adaptiveGamma(adaptiveGammaSEXP);
    Rcpp::traits::input_parameter< double >::type q(qSEXP);
    rcpp_result_gen = Rcpp::wrap(DiversityGibbsSamp(X, hyper, K, m, iteration, burnin, thin, method, gamma, trueAllocation, adaptiveGamma, q));
    return rcpp_result_gen;
END_RCPP
}
// CSSG
List CSSG(arma::mat X, arma::vec hyper, int K, int R, int iteration, int burnin, int thin, arma::irowvec trueAllocation);
RcppExport SEXP _AdaptiveAllocation_CSSG(SEXP XSEXP, SEXP hyperSEXP, SEXP KSEXP, SEXP RSEXP, SEXP iterationSEXP, SEXP burninSEXP, SEXP thinSEXP, SEXP trueAllocationSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type hyper(hyperSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type R(RSEXP);
    Rcpp::traits::input_parameter< int >::type iteration(iterationSEXP);
    Rcpp::traits::input_parameter< int >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< int >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< arma::irowvec >::type trueAllocation(trueAllocationSEXP);
    rcpp_result_gen = Rcpp::wrap(CSSG(X, hyper, K, R, iteration, burnin, thin, trueAllocation));
    return rcpp_result_gen;
END_RCPP
}
// CRSG
List CRSG(arma::mat X, arma::vec hyper, int K, int R, int m, int iteration, int burnin, int thin, arma::irowvec trueAllocation);
RcppExport SEXP _AdaptiveAllocation_CRSG(SEXP XSEXP, SEXP hyperSEXP, SEXP KSEXP, SEXP RSEXP, SEXP mSEXP, SEXP iterationSEXP, SEXP burninSEXP, SEXP thinSEXP, SEXP trueAllocationSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type hyper(hyperSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type R(RSEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< int >::type iteration(iterationSEXP);
    Rcpp::traits::input_parameter< int >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< int >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< arma::irowvec >::type trueAllocation(trueAllocationSEXP);
    rcpp_result_gen = Rcpp::wrap(CRSG(X, hyper, K, R, m, iteration, burnin, thin, trueAllocation));
    return rcpp_result_gen;
END_RCPP
}
// CDSG
List CDSG(arma::mat X, arma::vec hyper, int K, int R, int m, int iteration, int burnin, int thin, double gamma, arma::irowvec trueAllocation, bool adaptiveGamma, int q);
RcppExport SEXP _AdaptiveAllocation_CDSG(SEXP XSEXP, SEXP hyperSEXP, SEXP KSEXP, SEXP RSEXP, SEXP mSEXP, SEXP iterationSEXP, SEXP burninSEXP, SEXP thinSEXP, SEXP gammaSEXP, SEXP trueAllocationSEXP, SEXP adaptiveGammaSEXP, SEXP qSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type hyper(hyperSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type R(RSEXP);
    Rcpp::traits::input_parameter< int >::type m(mSEXP);
    Rcpp::traits::input_parameter< int >::type iteration(iterationSEXP);
    Rcpp::traits::input_parameter< int >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< int >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< double >::type gamma(gammaSEXP);
    Rcpp::traits::input_parameter< arma::irowvec >::type trueAllocation(trueAllocationSEXP);
    Rcpp::traits::input_parameter< bool >::type adaptiveGamma(adaptiveGammaSEXP);
    Rcpp::traits::input_parameter< int >::type q(qSEXP);
    rcpp_result_gen = Rcpp::wrap(CDSG(X, hyper, K, R, m, iteration, burnin, thin, gamma, trueAllocation, adaptiveGamma, q));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_AdaptiveAllocation_customMatrix", (DL_FUNC) &_AdaptiveAllocation_customMatrix, 1},
    {"_AdaptiveAllocation_callrgamma", (DL_FUNC) &_AdaptiveAllocation_callrgamma, 3},
    {"_AdaptiveAllocation_csample_num", (DL_FUNC) &_AdaptiveAllocation_csample_num, 4},
    {"_AdaptiveAllocation_diracF", (DL_FUNC) &_AdaptiveAllocation_diracF, 2},
    {"_AdaptiveAllocation_summary_Posterior", (DL_FUNC) &_AdaptiveAllocation_summary_Posterior, 1},
    {"_AdaptiveAllocation_rdirichlet_cpp", (DL_FUNC) &_AdaptiveAllocation_rdirichlet_cpp, 2},
    {"_AdaptiveAllocation_BinderLoss", (DL_FUNC) &_AdaptiveAllocation_BinderLoss, 2},
    {"_AdaptiveAllocation_coefBinom", (DL_FUNC) &_AdaptiveAllocation_coefBinom, 2},
    {"_AdaptiveAllocation_ari", (DL_FUNC) &_AdaptiveAllocation_ari, 2},
    {"_AdaptiveAllocation_myProduct", (DL_FUNC) &_AdaptiveAllocation_myProduct, 1},
    {"_AdaptiveAllocation_SSG", (DL_FUNC) &_AdaptiveAllocation_SSG, 8},
    {"_AdaptiveAllocation_RSSG", (DL_FUNC) &_AdaptiveAllocation_RSSG, 9},
    {"_AdaptiveAllocation_JS_distance", (DL_FUNC) &_AdaptiveAllocation_JS_distance, 2},
    {"_AdaptiveAllocation_DiversityGibbsSamp", (DL_FUNC) &_AdaptiveAllocation_DiversityGibbsSamp, 12},
    {"_AdaptiveAllocation_CSSG", (DL_FUNC) &_AdaptiveAllocation_CSSG, 8},
    {"_AdaptiveAllocation_CRSG", (DL_FUNC) &_AdaptiveAllocation_CRSG, 9},
    {"_AdaptiveAllocation_CDSG", (DL_FUNC) &_AdaptiveAllocation_CDSG, 12},
    {NULL, NULL, 0}
};

RcppExport void R_init_AdaptiveAllocation(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
