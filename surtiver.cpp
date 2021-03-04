#define ARMA_NO_DEBUG
#define ARMA_DONT_USE_OPENMP
#define STRICT_R_HEADERS // needed on Windows, not macOS
#include <RcppArmadillo.h>
#include <omp.h>
#include <RcppArmadilloExtensions/sample.h> // for Rcpp::RcppArmadillo::sample
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;
using namespace std;

tuple<vector<uvec>, vector<uvec>> ranstra(const vec &event, 
                                          const unsigned int &n_stra) {
  vector<uvec> idx_stra, idx_fail;
  uvec id_stra = Rcpp::RcppArmadillo::sample(regspace<uvec>(0, n_stra-1),
                                             event.n_rows, true);
  for (unsigned int i = 0; i < n_stra; ++i) {
    uvec idx_stra_tmp = find(id_stra==i);
    uvec idx_fail_tmp = find(event.rows(idx_stra_tmp)==1);
    idx_stra.push_back(idx_stra_tmp);
    idx_fail.push_back(idx_fail_tmp);
  }
  return make_tuple(idx_stra, idx_fail);
}

// [[Rcpp::export]]
double logplkd_fixtra(const vec &event, const IntegerVector &count_strata,
                      const mat &Z_tv, const mat &B_spline, const mat &theta,
                      const mat &Z_ti, const vec &beta_ti,
                      const bool &parallel=false, const unsigned int &threads=1) {
  // data should be ordered by strata first and then time
  // Z_tv: design matrix corresponding to time-varying coefs
  // B_spline: a matrix of B-splines, # of time points as row ct,
  //           and # of knots as col ct
  // theta: p by k; B_spline: t by k
  // Z_ti: design matrix corresponding to time-invariant coefs
  // beta_ti: a vector of initial values of time-invariant coefs
  bool ti = norm(Z_ti, "inf") > sqrt(datum::eps);
  IntegerVector cumsum_strata = cumsum(count_strata);
  unsigned int n_strata = cumsum_strata.length();
  cumsum_strata.push_front(0);
  double logplkd = 0.0;
  for (unsigned int i = 0; i < n_strata; ++i) {
    unsigned int start = cumsum_strata[i], end = cumsum_strata[i+1];
    uvec idx_fail = find(event.rows(start, end-1)==1);
    mat Z_tv_theta = Z_tv.rows(start+idx_fail(0), end-1) * theta;
    vec Z_ti_beta_ti;
    if (ti) {
      Z_ti_beta_ti = Z_ti.rows(start+idx_fail(0), end-1) * beta_ti;
    }
    mat B_sp = B_spline.rows(start+idx_fail);
    idx_fail -= idx_fail(0); // index within the strata
    unsigned int n_fail = idx_fail.n_elem, n_Z_tv_theta = Z_tv_theta.n_rows;
    vec lincomb_fail(n_fail);
    if (parallel) {
      double scale_fac = as_scalar(idx_fail.tail(1));
      vec cumsum_ar = (double)n_Z_tv_theta / scale_fac * regspace(1,n_fail) -
        cumsum(conv_to<vec>::from(idx_fail)/scale_fac); // cum sum of at risk cts
      omp_set_num_threads(threads);
      #pragma omp parallel
      {
        unsigned int id, j, istart, iend;
        id = omp_get_thread_num();
        istart = as_scalar(
          find(cumsum_ar >= cumsum_ar(n_fail-1)/(double)threads*id, 1));
        // exclusive
        iend = as_scalar(
          find(cumsum_ar >= cumsum_ar(n_fail-1)/(double)threads*(id+1), 1));
        if (id==threads-1) iend = n_fail;
        double val_tmp = 0;
        if (ti) {
          for (j = istart; j < iend; ++j) {
            vec lincomb =
              Z_tv_theta.rows(idx_fail(j), n_Z_tv_theta-1)*B_sp.row(j).t() +
              Z_ti_beta_ti.subvec(idx_fail(j), n_Z_tv_theta-1);
            lincomb_fail(j) = lincomb(0);
            val_tmp += log(sum(exp(lincomb)));
          }
        } else {
          for (j = istart; j < iend; ++j) {
            vec lincomb =
              Z_tv_theta.rows(idx_fail(j), n_Z_tv_theta-1) * B_sp.row(j).t();
            lincomb_fail(j) = lincomb(0);
            val_tmp += log(sum(exp(lincomb)));
          }
        }
        #pragma omp atomic
        logplkd -= val_tmp;
      }
    } else {
      if (ti) {
        for (unsigned int j = 0; j < n_fail; ++j) {
          vec lincomb =
            Z_tv_theta.rows(idx_fail(j), n_Z_tv_theta-1) * B_sp.row(j).t() +
            Z_ti_beta_ti.subvec(idx_fail(j), n_Z_tv_theta-1);
          lincomb_fail(j) = lincomb(0);
          logplkd -= log(sum(exp(lincomb)));
        }
      } else {
        for (unsigned int j = 0; j < n_fail; ++j) {
          vec lincomb =
            Z_tv_theta.rows(idx_fail(j), n_Z_tv_theta-1) * B_sp.row(j).t();
          lincomb_fail(j) = lincomb(0);
          logplkd -= log(sum(exp(lincomb)));
        }
      }
    }
    logplkd += accu(lincomb_fail);
  }
  logplkd /= Z_tv.n_rows;
  return logplkd;
}

// [[Rcpp::export]]
double logplkd_sim(const vec &theta_vec, const vec &event, 
                   const IntegerVector &count_strata,
                   const mat &Z_tv, const mat &B_spline,
                   const mat &Z_ti, const vec &beta_ti,
                   const bool &parallel=false, const unsigned int &threads=1) {
  // data should be ordered by strata first and then time
  // Z_tv: design matrix corresponding to time-varying coefs
  // B_spline: a matrix of B-splines, # of time points as row ct,
  //           and # of knots as col ct
  // theta: p by k; B_spline: t by k
  // Z_ti: design matrix corresponding to time-invariant coefs
  // beta_ti: a vector of initial values of time-invariant coefs
  mat theta = reshape(theta_vec, Z_tv.n_cols, B_spline.n_cols);
  bool ti = norm(Z_ti, "inf") > sqrt(datum::eps);
  IntegerVector cumsum_strata = cumsum(count_strata);
  unsigned int n_strata = cumsum_strata.length();
  cumsum_strata.push_front(0);
  double logplkd = 0.0;
  for (unsigned int i = 0; i < n_strata; ++i) {
    unsigned int start = cumsum_strata[i], end = cumsum_strata[i+1];
    uvec idx_fail = find(event.rows(start, end-1)==1);
    mat Z_tv_theta = Z_tv.rows(start+idx_fail(0), end-1) * theta;
    vec Z_ti_beta_ti;
    if (ti) {
      Z_ti_beta_ti = Z_ti.rows(start+idx_fail(0), end-1) * beta_ti;
    }
    mat B_sp = B_spline.rows(start+idx_fail);
    idx_fail -= idx_fail(0); // index within the strata
    unsigned int n_fail = idx_fail.n_elem, n_Z_tv_theta = Z_tv_theta.n_rows;
    vec lincomb_fail(n_fail);
    if (parallel) {
      double scale_fac = as_scalar(idx_fail.tail(1));
      vec cumsum_ar = (double)n_Z_tv_theta / scale_fac * regspace(1,n_fail) -
        cumsum(conv_to<vec>::from(idx_fail)/scale_fac); // cum sum of at risk cts
      omp_set_num_threads(threads);
      #pragma omp parallel
      {
        unsigned int id, j, istart, iend;
        id = omp_get_thread_num();
        istart = as_scalar(
          find(cumsum_ar >= cumsum_ar(n_fail-1)/(double)threads*id, 1));
        // exclusive
        iend = as_scalar(
          find(cumsum_ar >= cumsum_ar(n_fail-1)/(double)threads*(id+1), 1));
        if (id==threads-1) iend = n_fail;
        double val_tmp = 0;
        if (ti) {
          for (j = istart; j < iend; ++j) {
            vec lincomb =
              Z_tv_theta.rows(idx_fail(j), n_Z_tv_theta-1)*B_sp.row(j).t() +
              Z_ti_beta_ti.subvec(idx_fail(j), n_Z_tv_theta-1);
            lincomb_fail(j) = lincomb(0);
            val_tmp += log(sum(exp(lincomb)));
          }
        } else {
          for (j = istart; j < iend; ++j) {
            vec lincomb =
              Z_tv_theta.rows(idx_fail(j), n_Z_tv_theta-1) * B_sp.row(j).t();
            lincomb_fail(j) = lincomb(0);
            val_tmp += log(sum(exp(lincomb)));
          }
        }
        #pragma omp atomic
        logplkd -= val_tmp;
      }
    } else {
      if (ti) {
        for (unsigned int j = 0; j < n_fail; ++j) {
          vec lincomb =
            Z_tv_theta.rows(idx_fail(j), n_Z_tv_theta-1) * B_sp.row(j).t() +
            Z_ti_beta_ti.subvec(idx_fail(j), n_Z_tv_theta-1);
          lincomb_fail(j) = lincomb(0);
          logplkd -= log(sum(exp(lincomb)));
        }
      } else {
        for (unsigned int j = 0; j < n_fail; ++j) {
          vec lincomb =
            Z_tv_theta.rows(idx_fail(j), n_Z_tv_theta-1) * B_sp.row(j).t();
          lincomb_fail(j) = lincomb(0);
          logplkd -= log(sum(exp(lincomb)));
        }
      }
    }
    logplkd += accu(lincomb_fail);
  }
  logplkd /= Z_tv.n_rows;
  return logplkd;
}


// [[Rcpp::export]]
double logplkd_ranstra(const vec &event, const unsigned int &n_stra,
               const mat &Z_tv, const mat &B_spline, const mat &theta,
               const mat &Z_ti, const vec &beta_ti,
               const bool parallel=false, const unsigned int threads=1) {
  
  bool ti = norm(Z_ti, "inf") > sqrt(datum::eps);
  double logplkd = 0.0;
  auto tuple_idx = ranstra(event, n_stra);
  vector<uvec> idx_stra, idx_fail;
  tie(idx_stra, idx_fail) = tuple_idx;
  if (parallel) {
    uvec quant = conv_to<uvec>::from(
      floor(quantile(regspace<uvec>(0,n_stra),regspace(0.0,1.0/threads,1.0))));
    omp_set_num_threads(threads);
    #pragma omp parallel
    {
      unsigned id = omp_get_thread_num();
      double val_tmp = 0;
      for (unsigned int i = quant[id]; i < quant[id+1]; ++i) {
        uvec idx_stra_tmp = idx_stra[i];
        uvec idx_fail_tmp = idx_fail[i];
        vec Z_tv_theta = Z_tv.rows(idx_stra_tmp) * theta.col(0);
        vec Z_ti_beta_ti;
        if (ti) Z_ti_beta_ti = Z_ti.rows(idx_stra_tmp) * beta_ti;
        mat B_sp = B_spline.rows(idx_stra_tmp.elem(idx_fail_tmp));
        unsigned int n_Z_stra = Z_tv_theta.n_rows;
        vec lincomb_fail(idx_fail[i].n_elem);
        if (ti) {
          for (unsigned int j = 0; j < idx_fail[i].n_elem; ++j) {
            vec lincomb =
              Z_tv_theta.rows(idx_fail[i](j), n_Z_stra-1) *
              B_sp.row(j).t() +
              Z_ti_beta_ti.subvec(idx_fail[i](j),n_Z_stra-1);
            lincomb_fail(j) = lincomb(0);
            val_tmp += log(sum(exp(lincomb)));
          }
        } else {
          for (unsigned int j = 0; j < idx_fail[i].n_elem; ++j) {
            vec lincomb =
              Z_tv_theta.rows(idx_fail[i](j), n_Z_stra-1) *
              B_sp.row(j).t();
            lincomb_fail(j) = lincomb(0);
            val_tmp += log(sum(exp(lincomb)));
          }
        }
        logplkd += accu(lincomb_fail);
      }
      #pragma omp atomic
      logplkd -= val_tmp;
    }
  } else {
    for (unsigned int i = 0; i < n_stra; ++i) {
      uvec idx_stra_tmp = idx_stra[i];
      uvec idx_fail_tmp = idx_fail[i];
      vec Z_tv_theta = Z_tv.rows(idx_stra_tmp) * theta.col(0);
      vec Z_ti_beta_ti;
      if (ti) Z_ti_beta_ti = Z_ti.rows(idx_stra_tmp) * beta_ti;
      mat B_sp = B_spline.rows(idx_stra_tmp.elem(idx_fail_tmp));
      unsigned int n_Z_stra = Z_tv_theta.n_rows;
      vec lincomb_fail(idx_fail[i].n_elem);
      if (ti) {
        for (unsigned int j = 0; j < idx_fail[i].n_elem; ++j) {
          vec lincomb =
            Z_tv_theta.rows(idx_fail[i](j), n_Z_stra-1) *
            B_sp.row(j).t() +
            Z_ti_beta_ti.subvec(idx_fail[i](j),n_Z_stra-1);
          lincomb_fail(j) = lincomb(0);
          logplkd -= log(sum(exp(lincomb)));
        }
      } else {
        for (unsigned int j = 0; j < idx_fail[i].n_elem; ++j) {
          vec lincomb =
            Z_tv_theta.rows(idx_fail[i](j), n_Z_stra-1) *
            B_sp.row(j).t();
          lincomb_fail(j) = lincomb(0);
          logplkd -= log(sum(exp(lincomb)));
        }
      }
      logplkd += accu(lincomb_fail);
    }
  }
  logplkd /= Z_tv.n_rows;
  return logplkd;
}

// [[Rcpp::export]]
List gradinfo_fixtra(const vec &event, const IntegerVector &count_strata,
                     const mat &Z_tv, const mat &B_spline, const mat &theta,
                     const mat &Z_ti, const vec &beta_ti,
                     const string &method="Newton", const double &lambda=1e8,
                     const bool &parallel=false, const unsigned int &threads=1) {
  // data should be ordered by strata first and then time
  // theta: p by k; B_spline: t by k

  bool ti = norm(Z_ti, "inf") > sqrt(datum::eps);
  IntegerVector cumsum_strata = cumsum(count_strata);
  unsigned int n_strata = cumsum_strata.length();
  cumsum_strata.push_front(0);
  vec grad; mat info, gradd(theta.n_elem, int(accu(event)),fill::zeros);
  if (ti) {
    grad = zeros<vec>(theta.n_elem+beta_ti.n_elem);
    info = zeros<mat>(theta.n_elem+beta_ti.n_elem, theta.n_elem+beta_ti.n_elem);
  } else {
    grad = zeros<vec>(theta.n_elem);
    info = zeros<mat>(theta.n_elem, theta.n_elem);
  }
  for (unsigned int i = 0; i < n_strata; ++i) {
    uvec idx_fail =
      find(event.rows(cumsum_strata[i], cumsum_strata[i+1]-1)==1);
    mat Z_tv_strata =
      Z_tv.rows(cumsum_strata[i]+idx_fail(0), cumsum_strata[i+1]-1);
    mat Z_ti_strata; vec Z_ti_beta_ti;
    if (ti) {
      Z_ti_strata =
        Z_ti.rows(cumsum_strata[i]+idx_fail(0), cumsum_strata[i+1]-1);
      Z_ti_beta_ti = Z_ti_strata * beta_ti;
    }
    mat B_sp = B_spline.rows(cumsum_strata[i]+idx_fail);
    mat Z_tv_theta = Z_tv_strata * theta;
    idx_fail -= idx_fail(0); // index within the strata
    unsigned int n_fail = idx_fail.n_elem, n_Z_tv_theta = Z_tv_theta.n_rows;
    if (parallel) {
      double scale_fac = as_scalar(idx_fail.tail(1));
      vec cumsum_ar = (double)n_Z_tv_theta / scale_fac * regspace(1,n_fail) -
        cumsum(conv_to<vec>::from(idx_fail)/scale_fac); // cum sum of at risk counts
      omp_set_num_threads(threads);
      #pragma omp parallel
      {
        unsigned int id, j, istart, iend;
        id = omp_get_thread_num();
        istart =
          as_scalar(find(cumsum_ar>=cumsum_ar(n_fail-1)/(double)threads*id, 1));
        iend =
          as_scalar(find(cumsum_ar>=cumsum_ar(n_fail-1)/(double)threads*(id+1), 1)); // exclusive
        if (id == threads-1) iend = n_fail;
        vec grad_tmp(size(grad), fill::zeros);
        mat info_tmp(size(info), fill::zeros);
        if (ti) {
          for (j = istart; j < iend; ++j) {
            vec B_sp_tmp = B_sp.row(j).t();
            unsigned int arstart = idx_fail(j), arend = n_Z_tv_theta-1; // at risk set indices
            vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp +
                                  Z_ti_beta_ti.subvec(arstart,arend));
            double S0 = accu(exp_lincomb);
            mat Z_tv_exp =
              Z_tv_strata.rows(arstart,arend).each_col() % exp_lincomb;
            mat Z_ti_exp =
              Z_ti_strata.rows(arstart,arend).each_col() % exp_lincomb;
            vec S1_tv = sum(Z_tv_exp).t(), S1_ti = sum(Z_ti_exp).t();
            vec S1 = join_cols(kron(S1_tv,B_sp_tmp), S1_ti)/S0;
            mat S2_11 =
              kron(Z_tv_strata.rows(arstart,arend).t()*Z_tv_exp,
                   B_sp_tmp*B_sp_tmp.t())/S0,
                S2_12 =
                  kron(Z_tv_strata.rows(arstart,arend).t()*Z_ti_exp,
                       B_sp_tmp)/S0,
                S2_22 = Z_ti_strata.rows(arstart,arend).t()*Z_ti_exp/S0;
            grad_tmp += join_cols(kron(Z_tv_strata.row(arstart).t(), B_sp_tmp),
                                  Z_ti_strata.row(arstart).t()) - S1;
            info_tmp += join_cols(join_rows(S2_11, S2_12),
                                  join_rows(S2_12.t(), S2_22)) - S1*S1.t();
          }
        } else {
          for (j = istart; j < iend; ++j) {
            vec B_sp_tmp = B_sp.row(j).t();
            unsigned int arstart = idx_fail(j), arend = n_Z_tv_theta-1; // at risk set indices
            vec exp_lincomb =
              exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp);
            double S0 = accu(exp_lincomb);
            mat Z_tv_exp =
              Z_tv_strata.rows(arstart,arend).each_col() % exp_lincomb;
            vec S1_tv = sum(Z_tv_exp).t();
            mat S2 = Z_tv_strata.rows(arstart,arend).t() * Z_tv_exp;
            grad_tmp += kron(Z_tv_strata.row(arstart).t()-S1_tv/S0, B_sp_tmp);
            info_tmp +=
              kron(S2/S0-S1_tv*S1_tv.t()/pow(S0,2),B_sp_tmp*B_sp_tmp.t());
          }
        }
        #pragma omp critical (gradient)
        grad += grad_tmp;
        #pragma omp critical (information)
        info += info_tmp;
      }
    } else {
      if (ti) {
        for (unsigned int j = 0; j < n_fail; ++j) {
          vec B_sp_tmp = B_sp.row(j).t();
          unsigned int arstart = idx_fail(j), arend = n_Z_tv_theta-1; // at risk set indices
          vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp +
                                Z_ti_beta_ti.subvec(arstart,arend));
          double S0 = accu(exp_lincomb);
          mat Z_tv_exp =
            Z_tv_strata.rows(arstart,arend).each_col() % exp_lincomb;
          mat Z_ti_exp =
            Z_ti_strata.rows(arstart,arend).each_col() % exp_lincomb;
          vec S1_tv = sum(Z_tv_exp).t(), S1_ti = sum(Z_ti_exp).t();
          vec S1 = join_cols(kron(S1_tv,B_sp_tmp), S1_ti)/S0;
          mat S2_11 =
            kron(Z_tv_strata.rows(arstart,arend).t()*Z_tv_exp,
                 B_sp_tmp*B_sp_tmp.t())/S0,
              S2_12 =
                kron(Z_tv_strata.rows(arstart,arend).t()*Z_ti_exp,
                     B_sp_tmp)/S0,
              S2_22 = Z_ti_strata.rows(arstart,arend).t()*Z_ti_exp/S0;
          grad += join_cols(kron(Z_tv_strata.row(arstart).t(), B_sp_tmp),
                            Z_ti_strata.row(arstart).t()) - S1;
          info += join_cols(join_rows(S2_11, S2_12),
                            join_rows(S2_12.t(), S2_22)) - S1*S1.t();
        }
      } else {
        for (unsigned int j = 0; j < n_fail; ++j) {
          vec B_sp_tmp = B_sp.row(j).t();
          unsigned int arstart = idx_fail(j), arend = n_Z_tv_theta-1; // at risk set indices
          vec exp_lincomb =
            exp(Z_tv_theta.rows(arstart, arend) * B_sp_tmp);
          double S0 = accu(exp_lincomb);
          mat Z_tv_exp =
            Z_tv_strata.rows(arstart,arend).each_col() % exp_lincomb;
          vec S1_tv = sum(Z_tv_exp).t();
          mat S2 = Z_tv_strata.rows(arstart,arend).t() * Z_tv_exp;
          grad += kron(Z_tv_strata.row(arstart).t() - S1_tv/S0, B_sp_tmp);
          gradd.col(j) += kron(Z_tv_strata.row(arstart).t() - S1_tv/S0, B_sp_tmp);
          info += kron(S2/S0-S1_tv*S1_tv.t()/pow(S0,2),B_sp_tmp*B_sp_tmp.t());
        }
      }
    }
  }

  vec step; // Newton step
  if (method=="ProxN") {
    info.diag() += Z_tv.n_rows / lambda;
    step = solve(info, grad, solve_opts::fast+solve_opts::likely_sympd);
    info.diag() -= Z_tv.n_rows / lambda;
  } else if (method=="Newton") {
    step = solve(info, grad, solve_opts::fast+solve_opts::likely_sympd);
  }
  double inc = dot(grad, step) / Z_tv.n_rows; // square of Newton increment
  if (ti) {
    return List::create(_["step_tv"]=reshape(step.head(theta.n_elem),
                                      size(theta.t())).t(),
                                      _["step_ti"]=step.tail(beta_ti.n_elem),
                                      _["grad"]=grad, _["info"]=info,
                                      _["inc"]=inc);
  } else {
    return List::create(_["step"]=reshape(step, size(theta.t())).t(),
                        _["grad"]=grad, _["gradd"]=gradd, _["info"]=info, _["inc"]=inc);
  }
}

// [[Rcpp::export]]
mat info_sim(const mat &theta, const vec &event, 
             const IntegerVector &count_strata,
             const mat &Z_tv, const mat &B_spline, 
             const mat &Z_ti, const vec &beta_ti,
             const bool &parallel=true, const unsigned int &threads=1) {
  // data should be ordered by strata first and then time
  // theta: p by k; B_spline: t by k
  
  bool ti = norm(Z_ti, "inf") > sqrt(datum::eps);
  IntegerVector cumsum_strata = cumsum(count_strata);
  unsigned int n_strata = cumsum_strata.length();
  cumsum_strata.push_front(0);
  mat info;
  if (ti) {
    info = zeros<mat>(theta.n_elem+beta_ti.n_elem, theta.n_elem+beta_ti.n_elem);
  } else {
    info = zeros<mat>(theta.n_elem, theta.n_elem);
  }
  for (unsigned int i = 0; i < n_strata; ++i) {
    uvec idx_fail =
      find(event.rows(cumsum_strata[i], cumsum_strata[i+1]-1)==1);
    mat Z_tv_strata =
      Z_tv.rows(cumsum_strata[i]+idx_fail(0), cumsum_strata[i+1]-1);
    mat Z_ti_strata; vec Z_ti_beta_ti;
    if (ti) {
      Z_ti_strata =
        Z_ti.rows(cumsum_strata[i]+idx_fail(0), cumsum_strata[i+1]-1);
      Z_ti_beta_ti = Z_ti_strata * beta_ti;
    }
    mat B_sp = B_spline.rows(cumsum_strata[i]+idx_fail);
    mat Z_tv_theta = Z_tv_strata * theta;
    idx_fail -= idx_fail(0); // index within the strata
    unsigned int n_fail = idx_fail.n_elem, n_Z_tv_theta = Z_tv_theta.n_rows;
    if (parallel) {
      double scale_fac = as_scalar(idx_fail.tail(1));
      vec cumsum_ar = (double)n_Z_tv_theta / scale_fac * regspace(1,n_fail) -
        cumsum(conv_to<vec>::from(idx_fail)/scale_fac); // cum sum of at risk counts
      omp_set_num_threads(threads);
      #pragma omp parallel
      {
        unsigned int id, j, istart, iend;
        id = omp_get_thread_num();
        istart =
          as_scalar(find(cumsum_ar>=cumsum_ar(n_fail-1)/(double)threads*id, 1));
        iend =
          as_scalar(find(cumsum_ar>=cumsum_ar(n_fail-1)/(double)threads*(id+1), 1)); // exclusive
        if (id == threads-1) iend = n_fail;
        mat info_tmp(size(info), fill::zeros);
        if (ti) {
          for (j = istart; j < iend; ++j) {
            vec B_sp_tmp = B_sp.row(j).t();
            unsigned int arstart = idx_fail(j), arend = n_Z_tv_theta-1; // at risk set indices
            vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp +
              Z_ti_beta_ti.subvec(arstart,arend));
            double S0 = accu(exp_lincomb);
            mat Z_tv_exp =
              Z_tv_strata.rows(arstart,arend).each_col() % exp_lincomb;
            mat Z_ti_exp =
              Z_ti_strata.rows(arstart,arend).each_col() % exp_lincomb;
            vec S1_tv = sum(Z_tv_exp).t(), S1_ti = sum(Z_ti_exp).t();
            vec S1 = join_cols(kron(S1_tv,B_sp_tmp), S1_ti)/S0;
            mat S2_11 =
              kron(Z_tv_strata.rows(arstart,arend).t()*Z_tv_exp,
                   B_sp_tmp*B_sp_tmp.t())/S0,
                   S2_12 =
                     kron(Z_tv_strata.rows(arstart,arend).t()*Z_ti_exp,
                          B_sp_tmp)/S0,
                          S2_22 = Z_ti_strata.rows(arstart,arend).t()*Z_ti_exp/S0;
            info_tmp += join_cols(join_rows(S2_11, S2_12),
                                  join_rows(S2_12.t(), S2_22)) - S1*S1.t();
          }
        } else {
          for (j = istart; j < iend; ++j) {
            vec B_sp_tmp = B_sp.row(j).t();
            unsigned int arstart = idx_fail(j), arend = n_Z_tv_theta-1; // at risk set indices
            vec exp_lincomb =
              exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp);
            double S0 = accu(exp_lincomb);
            mat Z_tv_exp =
              Z_tv_strata.rows(arstart,arend).each_col() % exp_lincomb;
            vec S1_tv = sum(Z_tv_exp).t();
            mat S2 = Z_tv_strata.rows(arstart,arend).t() * Z_tv_exp;
            info_tmp +=
              kron(S2/S0-S1_tv*S1_tv.t()/pow(S0,2),B_sp_tmp*B_sp_tmp.t());
          }
        }
        #pragma omp critical (information)
        info += info_tmp;
      }
    } else {
      if (ti) {
        for (unsigned int j = 0; j < n_fail; ++j) {
          vec B_sp_tmp = B_sp.row(j).t();
          unsigned int arstart = idx_fail(j), arend = n_Z_tv_theta-1; // at risk set indices
          vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp +
            Z_ti_beta_ti.subvec(arstart,arend));
          double S0 = accu(exp_lincomb);
          mat Z_tv_exp =
            Z_tv_strata.rows(arstart,arend).each_col() % exp_lincomb;
          mat Z_ti_exp =
            Z_ti_strata.rows(arstart,arend).each_col() % exp_lincomb;
          vec S1_tv = sum(Z_tv_exp).t(), S1_ti = sum(Z_ti_exp).t();
          vec S1 = join_cols(kron(S1_tv,B_sp_tmp), S1_ti)/S0;
          mat S2_11 =
            kron(Z_tv_strata.rows(arstart,arend).t()*Z_tv_exp,
                 B_sp_tmp*B_sp_tmp.t())/S0,
                 S2_12 =
                   kron(Z_tv_strata.rows(arstart,arend).t()*Z_ti_exp,
                        B_sp_tmp)/S0,
                        S2_22 = Z_ti_strata.rows(arstart,arend).t()*Z_ti_exp/S0;
          info += join_cols(join_rows(S2_11, S2_12),
                            join_rows(S2_12.t(), S2_22)) - S1*S1.t();
        }
      } else {
        for (unsigned int j = 0; j < n_fail; ++j) {
          vec B_sp_tmp = B_sp.row(j).t();
          unsigned int arstart = idx_fail(j), arend = n_Z_tv_theta-1; // at risk set indices
          vec exp_lincomb =
            exp(Z_tv_theta.rows(arstart, arend) * B_sp_tmp);
          double S0 = accu(exp_lincomb);
          mat Z_tv_exp =
            Z_tv_strata.rows(arstart,arend).each_col() % exp_lincomb;
          vec S1_tv = sum(Z_tv_exp).t();
          mat S2 = Z_tv_strata.rows(arstart,arend).t() * Z_tv_exp;
          info += kron(S2/S0-S1_tv*S1_tv.t()/pow(S0,2),B_sp_tmp*B_sp_tmp.t());
        }
      }
    }
  }
  info.diag() += sqrt(datum::eps);
  return info;
}

// [[Rcpp::export]]
vec grad_sim(const vec &theta_vec, const vec &event,
             const IntegerVector &count_strata,
             const mat &Z_tv, const mat &B_spline, 
             const mat &Z_ti, const vec &beta_ti,
             const bool &parallel=true, const unsigned int &threads=1) {
  // data should be ordered by strata first and then time
  // theta: p by k; B_spline: t by k
  mat theta = reshape(theta_vec, Z_tv.n_cols, B_spline.n_cols);
  bool ti = norm(Z_ti, "inf") > sqrt(datum::eps);
  IntegerVector cumsum_strata = cumsum(count_strata);
  unsigned int n_strata = cumsum_strata.length();
  cumsum_strata.push_front(0);
  vec grad;
  if (ti) {
    grad = zeros<vec>(theta.n_elem+beta_ti.n_elem);
  } else {
    grad = zeros<vec>(theta.n_elem);
  }
  for (unsigned int i = 0; i < n_strata; ++i) {
    uvec idx_fail =
      find(event.rows(cumsum_strata[i], cumsum_strata[i+1]-1)==1);
    mat Z_tv_strata =
      Z_tv.rows(cumsum_strata[i]+idx_fail(0), cumsum_strata[i+1]-1);
    mat Z_ti_strata; vec Z_ti_beta_ti;
    if (ti) {
      Z_ti_strata =
        Z_ti.rows(cumsum_strata[i]+idx_fail(0), cumsum_strata[i+1]-1);
      Z_ti_beta_ti = Z_ti_strata * beta_ti;
    }
    mat B_sp = B_spline.rows(cumsum_strata[i]+idx_fail);
    mat Z_tv_theta = Z_tv_strata * theta;
    idx_fail -= idx_fail(0); // index within the strata
    unsigned int n_fail = idx_fail.n_elem, n_Z_tv_theta = Z_tv_theta.n_rows;
    if (parallel) {
      double scale_fac = as_scalar(idx_fail.tail(1));
      vec cumsum_ar = (double)n_Z_tv_theta / scale_fac * regspace(1,n_fail) -
        cumsum(conv_to<vec>::from(idx_fail)/scale_fac); // cum sum of at risk counts
      omp_set_num_threads(threads);
      #pragma omp parallel
      {
        unsigned int id, j, istart, iend;
        id = omp_get_thread_num();
        istart =
          as_scalar(find(cumsum_ar>=cumsum_ar(n_fail-1)/(double)threads*id, 1));
        iend =
          as_scalar(find(cumsum_ar>=cumsum_ar(n_fail-1)/(double)threads*(id+1), 1)); // exclusive
        if (id == threads-1) iend = n_fail;
        vec grad_tmp(size(grad), fill::zeros);
        if (ti) {
          for (j = istart; j < iend; ++j) {
            vec B_sp_tmp = B_sp.row(j).t();
            unsigned int arstart = idx_fail(j), arend = n_Z_tv_theta-1; // at risk set indices
            vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp +
              Z_ti_beta_ti.subvec(arstart,arend));
            double S0 = accu(exp_lincomb);
            mat Z_tv_exp =
              Z_tv_strata.rows(arstart,arend).each_col() % exp_lincomb;
            mat Z_ti_exp =
              Z_ti_strata.rows(arstart,arend).each_col() % exp_lincomb;
            vec S1_tv = sum(Z_tv_exp).t(), S1_ti = sum(Z_ti_exp).t();
            vec S1 = join_cols(kron(S1_tv,B_sp_tmp), S1_ti)/S0;
            grad_tmp += join_cols(kron(Z_tv_strata.row(arstart).t(), B_sp_tmp),
                                  Z_ti_strata.row(arstart).t()) - S1;
          }
        } else {
          for (j = istart; j < iend; ++j) {
            vec B_sp_tmp = B_sp.row(j).t();
            unsigned int arstart = idx_fail(j), arend = n_Z_tv_theta-1; // at risk set indices
            vec exp_lincomb =
              exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp);
            double S0 = accu(exp_lincomb);
            mat Z_tv_exp =
              Z_tv_strata.rows(arstart,arend).each_col() % exp_lincomb;
            vec S1_tv = sum(Z_tv_exp).t();
            grad_tmp += kron(Z_tv_strata.row(arstart).t()-S1_tv/S0, B_sp_tmp);
          }
        }
        #pragma omp critical (gradient)
        grad += grad_tmp;
      }
    } else {
      if (ti) {
        for (unsigned int j = 0; j < n_fail; ++j) {
          vec B_sp_tmp = B_sp.row(j).t();
          unsigned int arstart = idx_fail(j), arend = n_Z_tv_theta-1; // at risk set indices
          vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp +
            Z_ti_beta_ti.subvec(arstart,arend));
          double S0 = accu(exp_lincomb);
          mat Z_tv_exp =
            Z_tv_strata.rows(arstart,arend).each_col() % exp_lincomb;
          mat Z_ti_exp =
            Z_ti_strata.rows(arstart,arend).each_col() % exp_lincomb;
          vec S1_tv = sum(Z_tv_exp).t(), S1_ti = sum(Z_ti_exp).t();
          vec S1 = join_cols(kron(S1_tv,B_sp_tmp), S1_ti)/S0;
          grad += join_cols(kron(Z_tv_strata.row(arstart).t(), B_sp_tmp),
                            Z_ti_strata.row(arstart).t()) - S1;
        }
      } else {
        for (unsigned int j = 0; j < n_fail; ++j) {
          vec B_sp_tmp = B_sp.row(j).t();
          unsigned int arstart = idx_fail(j), arend = n_Z_tv_theta-1; // at risk set indices
          vec exp_lincomb =
            exp(Z_tv_theta.rows(arstart, arend) * B_sp_tmp);
          double S0 = accu(exp_lincomb);
          mat Z_tv_exp =
            Z_tv_strata.rows(arstart,arend).each_col() % exp_lincomb;
          vec S1_tv = sum(Z_tv_exp).t();
          grad += kron(Z_tv_strata.row(arstart).t() - S1_tv/S0, B_sp_tmp);
        }
      }
    }
  }
  return grad;
}

// [[Rcpp::export]]
List gradinfo_ranstra(const vec &event, const unsigned int &n_stra,
                      const mat &Z_tv, const mat &B_spline, const mat &theta,
                      const mat &Z_ti, const vec &beta_ti,
                      const string &method="Newton", const double &lambda=1e8,
                      const bool parallel=false, const unsigned int threads=1) {
  // data should be ordered by strata first and then time
  // theta: p by k; B_spline: t by k
  
  bool ti = norm(Z_ti, "inf") > sqrt(datum::eps);
  // IntegerVector cumsum_strata = cumsum(Rcpp::table(strata));
  // unsigned int n_strata = cumsum_strata.length();
  // cumsum_strata.push_front(0);
  vec grad; mat info;
  if (ti) {
    grad = zeros<vec>(theta.n_elem+beta_ti.n_elem);
    info = zeros<mat>(theta.n_elem+beta_ti.n_elem, theta.n_elem+beta_ti.n_elem);
  } else {
    grad = zeros<vec>(theta.n_elem);
    info = zeros<mat>(theta.n_elem, theta.n_elem);
  }
  auto tuple_idx = ranstra(event, n_stra);
  vector<uvec> idx_stra, idx_fail;
  tie(idx_stra, idx_fail) = tuple_idx;
  if (parallel) {
    uvec quant = conv_to<uvec>::from(
      floor(quantile(regspace<uvec>(0,n_stra),regspace(0.0,1.0/threads,1.0))));
    omp_set_num_threads(threads);
    #pragma omp parallel
    {
      unsigned int id = omp_get_thread_num();
      vec grad_tmp(size(grad), fill::zeros);
      mat info_tmp(size(info), fill::zeros);
      for (unsigned int i = quant[id]; i < quant[id+1]; ++i) {
        uvec idx_stra_tmp = idx_stra[i];
        uvec idx_fail_tmp = idx_fail[i];
        mat Z_tv_stra = Z_tv.rows(idx_stra_tmp);
        mat Z_ti_stra; vec Z_ti_beta_ti;
        if (ti) {
          Z_ti_stra = Z_ti.rows(idx_stra_tmp);
          Z_ti_beta_ti = Z_ti_stra * beta_ti;
        }
        mat Z_tv_theta = Z_tv_stra * theta;
        mat B_sp = B_spline.rows(idx_stra_tmp.elem(idx_fail_tmp));
        if (ti) {
          for (unsigned int j = 0; j < idx_fail_tmp.n_elem; ++j) {
            vec B_sp_tmp = B_sp.row(j).t();
            unsigned int arstart = idx_fail_tmp(j), arend = Z_tv_theta.n_rows-1;
            vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp +
              Z_ti_beta_ti.subvec(arstart,arend));
            double S0 = accu(exp_lincomb);
            mat Z_tv_exp =
              Z_tv_stra.rows(arstart,arend).each_col() % exp_lincomb;
            mat Z_ti_exp =
              Z_ti_stra.rows(arstart,arend).each_col() % exp_lincomb;
            vec S1_tv = sum(Z_tv_exp).t(), S1_ti = sum(Z_ti_exp).t();
            vec S1 = join_cols(kron(S1_tv,B_sp_tmp), S1_ti)/S0;
            mat S2_11 = kron(Z_tv_stra.rows(arstart,arend).t()*Z_tv_exp,
                             B_sp_tmp*B_sp_tmp.t())/S0,
                S2_12 = kron(Z_tv_stra.rows(arstart,arend).t()*Z_ti_exp,
                             B_sp_tmp)/S0,
                S2_22 = Z_ti_stra.rows(arstart,arend).t()*Z_ti_exp/S0;
            grad_tmp += join_cols(kron(Z_tv_stra.row(arstart).t(), B_sp_tmp),
                                  Z_ti_stra.row(arstart).t()) - S1;
            info_tmp += join_cols(join_rows(S2_11, S2_12),
                                  join_rows(S2_12.t(), S2_22)) - S1*S1.t();
          }
        } else {
          for (unsigned int j = 0; j < idx_fail_tmp.n_elem; ++j) {
            vec B_sp_tmp = B_sp.row(j).t();
            unsigned int arstart = idx_fail_tmp(j), arend = Z_tv_theta.n_rows-1;
            vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp);
            double S0 = accu(exp_lincomb);
            mat Z_tv_exp =
              Z_tv_stra.rows(arstart,arend).each_col() % exp_lincomb;
            vec S1_tv = sum(Z_tv_exp).t();
            mat S2 = Z_tv_stra.rows(arstart,arend).t() * Z_tv_exp;
            grad_tmp += kron(Z_tv_stra.row(arstart).t()-S1_tv/S0, B_sp_tmp);
            info_tmp += kron(S2/S0-S1_tv*S1_tv.t()/pow(S0,2),B_sp_tmp*B_sp_tmp.t());
          }
        }
      }
    #pragma omp critical (gradient)
      grad += grad_tmp;
    #pragma omp critical (information)
      info += info_tmp;
    }
  } else {
    for (unsigned int i = 0; i < n_stra; ++i) {
      uvec idx_stra_tmp = idx_stra[i];
      uvec idx_fail_tmp = idx_fail[i];
      mat Z_tv_stra = Z_tv.rows(idx_stra_tmp);
      mat Z_ti_stra; vec Z_ti_beta_ti;
      if (ti) {
        Z_ti_stra = Z_ti.rows(idx_stra_tmp);
        Z_ti_beta_ti = Z_ti_stra * beta_ti;
      }
      mat Z_tv_theta = Z_tv_stra * theta;
      mat B_sp = B_spline.rows(idx_stra_tmp.elem(idx_fail_tmp));
      if (ti) {
        for (unsigned int j = 0; j < idx_fail_tmp.n_elem; ++j) {
          vec B_sp_tmp = B_sp.row(j).t();
          unsigned int arstart = idx_fail_tmp(j), arend = Z_tv_theta.n_rows-1;
          vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp +
            Z_ti_beta_ti.subvec(arstart,arend));
          double S0 = accu(exp_lincomb);
          mat Z_tv_exp =
            Z_tv_stra.rows(arstart,arend).each_col() % exp_lincomb;
          mat Z_ti_exp =
            Z_ti_stra.rows(arstart,arend).each_col() % exp_lincomb;
          vec S1_tv = sum(Z_tv_exp).t(), S1_ti = sum(Z_ti_exp).t();
          vec S1 = join_cols(kron(S1_tv,B_sp_tmp), S1_ti)/S0;
          mat S2_11 = kron(Z_tv_stra.rows(arstart,arend).t()*Z_tv_exp,
                           B_sp_tmp*B_sp_tmp.t())/S0,
              S2_12 = kron(Z_tv_stra.rows(arstart,arend).t()*Z_ti_exp,
                           B_sp_tmp)/S0,
              S2_22 = Z_ti_stra.rows(arstart,arend).t()*Z_ti_exp/S0;
          grad += join_cols(kron(Z_tv_stra.row(arstart).t(), B_sp_tmp),
                            Z_ti_stra.row(arstart).t()) - S1;
          info += join_cols(join_rows(S2_11, S2_12),
                            join_rows(S2_12.t(), S2_22)) - S1*S1.t();
        }
      } else {
        for (unsigned int j = 0; j < idx_fail_tmp.n_elem; ++j) {
          vec B_sp_tmp = B_sp.row(j).t();
          unsigned int arstart = idx_fail_tmp(j), arend = Z_tv_theta.n_rows-1;
          vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp);
          double S0 = accu(exp_lincomb);
          mat Z_tv_exp =
            Z_tv_stra.rows(arstart,arend).each_col() % exp_lincomb;
          vec S1_tv = sum(Z_tv_exp).t();
          mat S2 = Z_tv_stra.rows(arstart,arend).t() * Z_tv_exp;
          grad += kron(Z_tv_stra.row(arstart).t()-S1_tv/S0, B_sp_tmp);
          info += kron(S2/S0-S1_tv*S1_tv.t()/pow(S0,2),B_sp_tmp*B_sp_tmp.t());
        }
      }
    }
  }
  vec step; // Newton step
  if (method=="ProxN") {
    info.diag() += Z_tv.n_rows / lambda;
    step = solve(info, grad, solve_opts::fast+solve_opts::likely_sympd);
    info.diag() -= Z_tv.n_rows / lambda;
  } else if (method=="Newton") {
    step = solve(info, grad, solve_opts::fast+solve_opts::likely_sympd);
  }
  double inc = dot(grad, step) / Z_tv.n_rows; // square of Newton increment
  if (ti) {
    return List::create(_["step_tv"]=reshape(step.head(theta.n_elem),
                                      size(theta.t())).t(),
                                      _["step_ti"]=step.tail(beta_ti.n_elem),
                                      _["grad"]=grad, _["info"]=info,
                                      _["inc"]=inc);
  } else {
    return List::create(_["step"]=reshape(step, size(theta.t())).t(),
                        _["grad"]=grad, _["info"]=info, _["inc"]=inc);
  }
}

List objfun_fixtra(const mat &Z_tv, const mat &B_spline, const mat &theta,
                   const mat &Z_ti, const vec &beta_ti, const bool &ti,
                   const unsigned int n_strata,
                   vector<uvec> &idx_B_sp, vector<uvec> &idx_fail,
                   vector<unsigned int> n_Z_strata,
                   vector<vector<unsigned int>> &idx_Z_strata,
                   vector<vector<unsigned int>> &istart,
                   vector<vector<unsigned int>> &iend,
                   const bool &parallel=false, const unsigned int &threads=1) {

  double norm_parm;
  vector<vec> hazard;
  if (ti) {
    norm_parm = max(norm(theta, "inf"), norm(beta_ti, "inf"));
  } else {
    norm_parm = norm(theta, "inf");
  }
  double logplkd = 0.0;
  if (norm_parm < sqrt(datum::eps)) { // theta and beta_ti are 0
    if (parallel) {
      for (unsigned int i = 0; i < n_strata; ++i) {
        vec S0_fail(idx_fail[i].n_elem);
        omp_set_num_threads(threads);
        #pragma omp parallel
        {
          unsigned int id, j;
          id = omp_get_thread_num();
          double val_tmp = 0;
          for (j = istart[i][id]; j < iend[i][id]; ++j) {
            S0_fail(j) = n_Z_strata[i]-idx_fail[i](j);
            val_tmp += log(S0_fail(j));
          }
          #pragma omp atomic
          logplkd -= val_tmp;
        }
        hazard.push_back(1/S0_fail);
      }
    } else {
      for (unsigned int i = 0; i < n_strata; ++i) {
        vec S0_fail(idx_fail[i].n_elem);
        for (unsigned int j = 0; j < idx_fail[i].n_elem; ++j) {
          S0_fail(j) = n_Z_strata[i]-idx_fail[i](j);
          logplkd -= log(S0_fail(j));
        }
        hazard.push_back(1/S0_fail);
      }
    }
  } else if (max(var(theta, 0, 1)) < sqrt(datum::eps)) { // each row of theta is const
    for (unsigned int i = 0; i < n_strata; ++i) {
      vec Z_tv_theta =
        Z_tv.rows(idx_Z_strata[i][0], idx_Z_strata[i][1]) * theta.col(0);
      vec Z_ti_beta_ti;
      if (ti) {
        Z_ti_beta_ti =
          Z_ti.rows(idx_Z_strata[i][0], idx_Z_strata[i][1]) * beta_ti;
      }
      mat B_sp = B_spline.rows(idx_B_sp[i]);
      unsigned int n_Z_tv_theta = Z_tv_theta.n_rows;
      vec lincomb_fail(idx_fail[i].n_elem), S0_fail(idx_fail[i].n_elem);
      if (parallel) {
        omp_set_num_threads(threads);
        #pragma omp parallel
        {
          unsigned int id, j;
          id = omp_get_thread_num();
          double val_tmp = 0;
          if (ti) {
            for (j = istart[i][id]; j < iend[i][id]; ++j) {
              vec lincomb =
                Z_tv_theta.subvec(idx_fail[i](j), n_Z_tv_theta-1) *
                accu(B_sp.row(j)) +
                Z_ti_beta_ti.subvec(idx_fail[i](j),n_Z_tv_theta-1);
              lincomb_fail(j) = lincomb(0);
              S0_fail(j) = sum(exp(lincomb));
              val_tmp += log(S0_fail(j));
            }
          } else {
            for (j = istart[i][id]; j < iend[i][id]; ++j) {
              vec lincomb =
                Z_tv_theta.subvec(idx_fail[i](j), n_Z_tv_theta-1) *
                accu(B_sp.row(j));
              lincomb_fail(j) = lincomb(0);
              S0_fail(j) = sum(exp(lincomb));
              val_tmp += log(S0_fail(j));
            }
          }
          #pragma omp atomic
          logplkd -= val_tmp;
        }
      } else {
        if (ti) {
          for (unsigned int j = 0; j < idx_fail[i].n_elem; ++j) {
            vec lincomb =
              Z_tv_theta.subvec(idx_fail[i](j),n_Z_tv_theta-1) *
              accu(B_sp.row(j)) +
              Z_ti_beta_ti.subvec(idx_fail[i](j),n_Z_tv_theta-1);
            lincomb_fail(j) = lincomb(0);
            S0_fail(j) = sum(exp(lincomb));
            logplkd -= log(S0_fail(j));
          }
        } else {
          for (unsigned int j = 0; j < idx_fail[i].n_elem; ++j) {
            vec lincomb =
              Z_tv_theta.subvec(idx_fail[i](j), n_Z_tv_theta-1) *
              accu(B_sp.row(j));
            lincomb_fail(j) = lincomb(0);
            S0_fail(j) = sum(exp(lincomb));
            logplkd -= log(S0_fail(j));
          }
        }
      }
      logplkd += accu(lincomb_fail);
      hazard.push_back(1/S0_fail);
    }
  } else { // general theta
    for (unsigned int i = 0; i < n_strata; ++i) {
      mat Z_tv_theta =
        Z_tv.rows(idx_Z_strata[i][0], idx_Z_strata[i][1]) * theta;
      vec Z_ti_beta_ti;
      if (ti) {
        Z_ti_beta_ti =
          Z_ti.rows(idx_Z_strata[i][0], idx_Z_strata[i][1]) * beta_ti;
      }
      mat B_sp = B_spline.rows(idx_B_sp[i]);
      unsigned int n_Z_tv_theta = Z_tv_theta.n_rows;
      vec lincomb_fail(idx_fail[i].n_elem), S0_fail(idx_fail[i].n_elem);
      if (parallel) {
        omp_set_num_threads(threads);
        #pragma omp parallel
        {
          unsigned int id, j;
          id = omp_get_thread_num();
          double val_tmp = 0;
          if (ti) {
            for (j = istart[i][id]; j < iend[i][id]; ++j) {
              vec lincomb =
                Z_tv_theta.rows(idx_fail[i](j), n_Z_tv_theta-1) *
                B_sp.row(j).t() +
                Z_ti_beta_ti.subvec(idx_fail[i](j),n_Z_tv_theta-1);
              lincomb_fail(j) = lincomb(0);
              S0_fail(j) = sum(exp(lincomb));
              val_tmp += log(S0_fail(j));
            }
          } else {
            for (j = istart[i][id]; j < iend[i][id]; ++j) {
              vec lincomb =
                Z_tv_theta.rows(idx_fail[i](j), n_Z_tv_theta-1) *
                B_sp.row(j).t();
              lincomb_fail(j) = lincomb(0);
              S0_fail(j) = sum(exp(lincomb));
              val_tmp += log(S0_fail(j));
            }
          }
          #pragma omp atomic
          logplkd -= val_tmp;
        }
      } else {
        if (ti) {
          for (unsigned int j = 0; j < idx_fail[i].n_elem; ++j) {
            vec lincomb =
              Z_tv_theta.rows(idx_fail[i](j), n_Z_tv_theta-1) * 
              B_sp.row(j).t() +
              Z_ti_beta_ti.subvec(idx_fail[i](j),n_Z_tv_theta-1);
            lincomb_fail(j) = lincomb(0);
            S0_fail(j) = sum(exp(lincomb));
            logplkd -= log(S0_fail(j));
          }
        } else {
          for (unsigned int j = 0; j < idx_fail[i].n_elem; ++j) {
            vec lincomb =
              Z_tv_theta.rows(idx_fail[i](j), n_Z_tv_theta-1) * 
              B_sp.row(j).t();
            lincomb_fail(j) = lincomb(0);
            S0_fail(j) = sum(exp(lincomb));
            logplkd -= log(S0_fail(j));
          }
        }
      }
      logplkd += accu(lincomb_fail);
      hazard.push_back(1/S0_fail);
    }
  }
  logplkd /= Z_tv.n_rows;
  return List::create(_["logplkd"]=logplkd, _["hazard"]=hazard);
}

List obj_fixtra_bresties(const mat &Z_tv, const mat &B_spline, const mat &theta,
                         const mat &Z_ti, const vec &beta_ti, const bool &ti,
                         const unsigned int n_strata,
                         vector<uvec> &idx_B_sp, vector<vector<uvec>> &idx_fail,
                         vector<unsigned int> n_Z_strata,
                         vector<vector<unsigned int>> &idx_Z_strata,
                         vector<vector<unsigned int>> &istart,
                         vector<vector<unsigned int>> &iend,
                         const bool &parallel=false, const unsigned int &threads=1) {
  
  double norm_parm;
  vector<vec> hazard;
  if (ti) {
    norm_parm = max(norm(theta, "inf"), norm(beta_ti, "inf"));
  } else {
    norm_parm = norm(theta, "inf");
  }
  double logplkd = 0.0;
  if (norm_parm < sqrt(datum::eps)) { // theta and beta_ti are 0
    if (parallel) {
      for (unsigned int i = 0; i < n_strata; ++i) {
        vec hazard_tmp(idx_fail[i].size());
        omp_set_num_threads(threads);
        #pragma omp parallel
        {
          unsigned int id, j;
          id = omp_get_thread_num();
          double val_tmp = 0;
          for (j = istart[i][id]; j < iend[i][id]; ++j) {
            double tmp = n_Z_strata[i]-idx_fail[i][j](0);
            hazard_tmp(j) = idx_fail[i][j].n_elem/tmp;
            val_tmp += idx_fail[i][j].n_elem*log(tmp);
          }
          #pragma omp atomic
          logplkd -= val_tmp;
        }
        hazard.push_back(hazard_tmp);
      }
    } else {
      for (unsigned int i = 0; i < n_strata; ++i) {
        vec hazard_tmp(idx_fail[i].size());
        for (unsigned int j = 0; j < idx_fail[i].size(); ++j) {
          double tmp = n_Z_strata[i]-idx_fail[i][j](0);
          hazard_tmp(j) = idx_fail[i][j].n_elem/tmp;
          logplkd -= idx_fail[i][j].n_elem*log(tmp);
        }
        hazard.push_back(hazard_tmp);
      }
    }
  } else if (max(var(theta, 0, 1)) < sqrt(datum::eps)) { // each row of theta is const
    for (unsigned int i = 0; i < n_strata; ++i) {
      vec Z_tv_theta =
        Z_tv.rows(idx_Z_strata[i][0], idx_Z_strata[i][1]) * theta.col(0);
      vec Z_ti_beta_ti;
      if (ti) {
        Z_ti_beta_ti =
          Z_ti.rows(idx_Z_strata[i][0], idx_Z_strata[i][1]) * beta_ti;
      }
      mat B_sp = B_spline.rows(idx_B_sp[i]);
      unsigned int n_Z_tv_theta = Z_tv_theta.n_rows;
      vec lincomb_fail(idx_fail[i].size()), hazard_tmp(idx_fail[i].size());
      if (parallel) {
        omp_set_num_threads(threads);
        #pragma omp parallel
        {
          unsigned int id, j;
          id = omp_get_thread_num();
          double val_tmp = 0;
          if (ti) {
            for (j = istart[i][id]; j < iend[i][id]; ++j) {
              vec lincomb =
                Z_tv_theta.subvec(idx_fail[i][j](0), n_Z_tv_theta-1) *
                accu(B_sp.row(j)) +
                Z_ti_beta_ti.subvec(idx_fail[i][j](0),n_Z_tv_theta-1);
              lincomb_fail(j) = sum(lincomb.head(idx_fail[i][j].n_elem));
              double tmp = sum(exp(lincomb));
              hazard_tmp(j) = idx_fail[i][j].n_elem/tmp;
              val_tmp += idx_fail[i][j].n_elem*log(tmp);
            }
          } else {
            for (j = istart[i][id]; j < iend[i][id]; ++j) {
              vec lincomb =
                Z_tv_theta.subvec(idx_fail[i][j](0), n_Z_tv_theta-1) *
                accu(B_sp.row(j));
              lincomb_fail(j) = sum(lincomb.head(idx_fail[i][j].n_elem));
              double tmp = sum(exp(lincomb));
              hazard_tmp(j) = idx_fail[i][j].n_elem/tmp;
              val_tmp += idx_fail[i][j].n_elem*log(tmp);
            }
          }
          #pragma omp atomic
          logplkd -= val_tmp;
        }
      } else {
        if (ti) {
          for (unsigned int j = 0; j < idx_fail[i].size(); ++j) {
            vec lincomb =
              Z_tv_theta.subvec(idx_fail[i][j](0),n_Z_tv_theta-1) *
              accu(B_sp.row(j)) +
              Z_ti_beta_ti.subvec(idx_fail[i][j](0),n_Z_tv_theta-1);
            lincomb_fail(j) = sum(lincomb.head(idx_fail[i][j].n_elem));
            double tmp = sum(exp(lincomb));
            hazard_tmp(j) = idx_fail[i][j].n_elem/tmp;
            logplkd -= idx_fail[i][j].n_elem*log(tmp);
          }
        } else {
          for (unsigned int j = 0; j < idx_fail[i].size(); ++j) {
            vec lincomb =
              Z_tv_theta.subvec(idx_fail[i][j](0), n_Z_tv_theta-1) *
              accu(B_sp.row(j));
            lincomb_fail(j) = sum(lincomb.head(idx_fail[i][j].n_elem));
            double tmp = sum(exp(lincomb));
            hazard_tmp(j) = idx_fail[i][j].n_elem/tmp;
            logplkd -= idx_fail[i][j].n_elem*log(tmp);
          }
        }
      }
      logplkd += accu(lincomb_fail);
      hazard.push_back(hazard_tmp);
    }
  } else { // general theta
    for (unsigned int i = 0; i < n_strata; ++i) {
      mat Z_tv_theta =
        Z_tv.rows(idx_Z_strata[i][0], idx_Z_strata[i][1]) * theta;
      vec Z_ti_beta_ti;
      if (ti) {
        Z_ti_beta_ti =
          Z_ti.rows(idx_Z_strata[i][0], idx_Z_strata[i][1]) * beta_ti;
      }
      mat B_sp = B_spline.rows(idx_B_sp[i]);
      unsigned int n_Z_tv_theta = Z_tv_theta.n_rows;
      vec lincomb_fail(idx_fail[i].size()), hazard_tmp(idx_fail[i].size());
      if (parallel) {
        omp_set_num_threads(threads);
        #pragma omp parallel
        {
          unsigned int id, j;
          id = omp_get_thread_num();
          double val_tmp = 0;
          if (ti) {
            for (j = istart[i][id]; j < iend[i][id]; ++j) {
              vec lincomb =
                Z_tv_theta.rows(idx_fail[i][j](0), n_Z_tv_theta-1) *
                B_sp.row(j).t() +
                Z_ti_beta_ti.subvec(idx_fail[i][j](0),n_Z_tv_theta-1);
              lincomb_fail(j) = sum(lincomb.head(idx_fail[i][j].n_elem));
              double tmp = sum(exp(lincomb));
              hazard_tmp(j) = idx_fail[i][j].n_elem/tmp;
              val_tmp += idx_fail[i][j].n_elem*log(tmp);
            }
          } else {
            for (j = istart[i][id]; j < iend[i][id]; ++j) {
              vec lincomb =
                Z_tv_theta.rows(idx_fail[i][j](0), n_Z_tv_theta-1) *
                B_sp.row(j).t();
              lincomb_fail(j) = sum(lincomb.head(idx_fail[i][j].n_elem));
              double tmp = sum(exp(lincomb));
              hazard_tmp(j) = idx_fail[i][j].n_elem/tmp;
              val_tmp += idx_fail[i][j].n_elem*log(tmp);
            }
          }
          #pragma omp atomic
          logplkd -= val_tmp;
        }
      } else {
        if (ti) {
          for (unsigned int j = 0; j < idx_fail[i].size(); ++j) {
            vec lincomb =
              Z_tv_theta.rows(idx_fail[i][j](0), n_Z_tv_theta-1) * 
              B_sp.row(j).t() +
              Z_ti_beta_ti.subvec(idx_fail[i][j](0),n_Z_tv_theta-1);
            lincomb_fail(j) = sum(lincomb.head(idx_fail[i][j].n_elem));
            double tmp = sum(exp(lincomb));
            hazard_tmp(j) = idx_fail[i][j].n_elem/tmp;
            logplkd -= idx_fail[i][j].n_elem*log(tmp);
          }
        } else {
          for (unsigned int j = 0; j < idx_fail[i].size(); ++j) {
            vec lincomb =
              Z_tv_theta.rows(idx_fail[i][j](0), n_Z_tv_theta-1) * 
              B_sp.row(j).t();
            lincomb_fail(j) = sum(lincomb.head(idx_fail[i][j].n_elem));
            double tmp = sum(exp(lincomb));
            hazard_tmp(j) = idx_fail[i][j].n_elem/tmp;
            logplkd -= idx_fail[i][j].n_elem*log(tmp);
          }
        }
      }
      logplkd += accu(lincomb_fail);
      hazard.push_back(hazard_tmp);
    }
  }
  logplkd /= Z_tv.n_rows;
  return List::create(_["logplkd"]=logplkd, _["hazard"]=hazard);
}

double objfun_ranstra(const mat &Z_tv, const mat &B_spline, const mat &theta,
                      const mat &Z_ti, const vec &beta_ti, const bool &ti,
                      const unsigned int &n_stra, vector<uvec> &idx_stra,
                      vector<uvec> &idx_fail,
                      const bool &parallel=false, const unsigned int &threads=1) {
  
  double norm_parm;
  if (ti) {
    norm_parm = max(norm(theta, "inf"), norm(beta_ti, "inf"));
  } else {
    norm_parm = norm(theta, "inf");
  }
  double logplkd = 0.0;
  if (norm_parm < sqrt(datum::eps)) { // theta = 0
    if (parallel) {
      uvec quant = conv_to<uvec>::from(
        floor(quantile(regspace<uvec>(0,n_stra),regspace(0.0,1.0/threads,1.0))));
      omp_set_num_threads(threads);
      #pragma omp parallel
      {
        unsigned int id = omp_get_thread_num();
        double val_tmp = 0;
        for (unsigned int i = quant[id]; i < quant[id+1]; ++i) {
          unsigned int n_Z_stra = idx_stra[i].n_elem-idx_fail[i](0);
          for (unsigned int j = 0; j < idx_fail[i].n_elem; ++j) {
            val_tmp += log(n_Z_stra-idx_fail[i](j));
          }
        }
        #pragma omp atomic
          logplkd -= val_tmp;
      }
    } else {
      for (unsigned int i = 0; i < n_stra; ++i) {
        unsigned int n_Z_stra = idx_stra[i].n_elem-idx_fail[i](0);
        for (unsigned int j = 0; j < idx_fail[i].n_elem; ++j) {
          logplkd -= log(n_Z_stra-idx_fail[i](j));
        }
      }
    }
  } else if (max(var(theta,0,1)) < sqrt(datum::eps)) { // each row of theta is const
    if (parallel) {
      uvec quant = conv_to<uvec>::from(
        floor(quantile(regspace<uvec>(0,n_stra),regspace(0.0,1.0/threads,1.0))));
      omp_set_num_threads(threads);
      #pragma omp parallel
      {
        unsigned id = omp_get_thread_num();
        double val_tmp = 0;
        for (unsigned int i = quant[id]; i < quant[id+1]; ++i) {
          uvec idx_stra_tmp = idx_stra[i];
          uvec idx_fail_tmp = idx_fail[i];
          vec Z_tv_theta = Z_tv.rows(idx_stra_tmp) * theta.col(0);
          vec Z_ti_beta_ti;
          if (ti) Z_ti_beta_ti = Z_ti.rows(idx_stra_tmp) * beta_ti;
          mat B_sp = B_spline.rows(idx_stra_tmp.elem(idx_fail_tmp));
          unsigned int n_Z_stra = Z_tv_theta.n_rows;
          vec lincomb_fail(idx_fail[i].n_elem);
          if (ti) {
            for (unsigned int j = 0; j < idx_fail[i].n_elem; ++j) {
              vec lincomb =
                Z_tv_theta.subvec(idx_fail[i](j), n_Z_stra-1) *
                accu(B_sp.row(j)) +
                Z_ti_beta_ti.subvec(idx_fail[i](j),n_Z_stra-1);
              lincomb_fail(j) = lincomb(0);
              val_tmp += log(sum(exp(lincomb)));
            }
          } else {
            for (unsigned int j = 0; j < idx_fail[i].n_elem; ++j) {
              vec lincomb =
                Z_tv_theta.subvec(idx_fail[i](j), n_Z_stra-1) *
                accu(B_sp.row(j));
              lincomb_fail(j) = lincomb(0);
              val_tmp += log(sum(exp(lincomb)));
            }
          }
          logplkd += accu(lincomb_fail);
        }
        #pragma omp atomic
        logplkd -= val_tmp;
      }
    } else {
      for (unsigned int i = 0; i < n_stra; ++i) {
        uvec idx_stra_tmp = idx_stra[i];
        uvec idx_fail_tmp = idx_fail[i];
        vec Z_tv_theta = Z_tv.rows(idx_stra_tmp) * theta.col(0);
        vec Z_ti_beta_ti;
        if (ti) Z_ti_beta_ti = Z_ti.rows(idx_stra_tmp) * beta_ti;
        mat B_sp = B_spline.rows(idx_stra_tmp.elem(idx_fail_tmp));
        unsigned int n_Z_stra = Z_tv_theta.n_rows;
        vec lincomb_fail(idx_fail[i].n_elem);
        if (ti) {
          for (unsigned int j = 0; j < idx_fail[i].n_elem; ++j) {
            vec lincomb =
              Z_tv_theta.subvec(idx_fail[i](j),n_Z_stra-1) *
              accu(B_sp.row(j)) +
              Z_ti_beta_ti.subvec(idx_fail[i](j),n_stra-1);
            lincomb_fail(j) = lincomb(0);
            logplkd -= log(sum(exp(lincomb)));
          }
        } else {
          for (unsigned int j = 0; j < idx_fail[i].n_elem; ++j) {
            vec lincomb =
              Z_tv_theta.subvec(idx_fail[i](j), n_Z_stra-1) *
              accu(B_sp.row(j));
            lincomb_fail(j) = lincomb(0);
            logplkd -= log(sum(exp(lincomb)));
          }
        }
        logplkd += accu(lincomb_fail);
      }
    }
  } else { // general theta
    if (parallel) {
      uvec quant = conv_to<uvec>::from(
        floor(quantile(regspace<uvec>(0,n_stra),regspace(0.0,1.0/threads,1.0))));
      omp_set_num_threads(threads);
      #pragma omp parallel
      {
        unsigned id = omp_get_thread_num();
        double val_tmp = 0;
        for (unsigned int i = quant[id]; i < quant[id+1]; ++i) {
          uvec idx_stra_tmp = idx_stra[i];
          uvec idx_fail_tmp = idx_fail[i];
          vec Z_tv_theta = Z_tv.rows(idx_stra_tmp) * theta.col(0);
          vec Z_ti_beta_ti;
          if (ti) Z_ti_beta_ti = Z_ti.rows(idx_stra_tmp) * beta_ti;
          mat B_sp = B_spline.rows(idx_stra_tmp.elem(idx_fail_tmp));
          unsigned int n_Z_stra = Z_tv_theta.n_rows;
          vec lincomb_fail(idx_fail[i].n_elem);
          if (ti) {
            for (unsigned int j = 0; j < idx_fail[i].n_elem; ++j) {
              vec lincomb =
                Z_tv_theta.rows(idx_fail[i](j), n_Z_stra-1) *
                B_sp.row(j).t() +
                Z_ti_beta_ti.subvec(idx_fail[i](j),n_Z_stra-1);
              lincomb_fail(j) = lincomb(0);
              val_tmp += log(sum(exp(lincomb)));
            }
          } else {
            for (unsigned int j = 0; j < idx_fail[i].n_elem; ++j) {
              vec lincomb =
                Z_tv_theta.rows(idx_fail[i](j), n_Z_stra-1) *
                B_sp.row(j).t();
              lincomb_fail(j) = lincomb(0);
              val_tmp += log(sum(exp(lincomb)));
            }
          }
          logplkd += accu(lincomb_fail);
        }
        #pragma omp atomic
        logplkd -= val_tmp;
      }
    } else {
      for (unsigned int i = 0; i < n_stra; ++i) {
        uvec idx_stra_tmp = idx_stra[i];
        uvec idx_fail_tmp = idx_fail[i];
        vec Z_tv_theta = Z_tv.rows(idx_stra_tmp) * theta.col(0);
        vec Z_ti_beta_ti;
        if (ti) Z_ti_beta_ti = Z_ti.rows(idx_stra_tmp) * beta_ti;
        mat B_sp = B_spline.rows(idx_stra_tmp.elem(idx_fail_tmp));
        unsigned int n_Z_stra = Z_tv_theta.n_rows;
        vec lincomb_fail(idx_fail[i].n_elem);
        if (ti) {
          for (unsigned int j = 0; j < idx_fail[i].n_elem; ++j) {
            vec lincomb =
              Z_tv_theta.rows(idx_fail[i](j), n_Z_stra-1) *
              B_sp.row(j).t() +
              Z_ti_beta_ti.subvec(idx_fail[i](j),n_Z_stra-1);
            lincomb_fail(j) = lincomb(0);
            logplkd -= log(sum(exp(lincomb)));
          }
        } else {
          for (unsigned int j = 0; j < idx_fail[i].n_elem; ++j) {
            vec lincomb =
              Z_tv_theta.rows(idx_fail[i](j), n_Z_stra-1) *
              B_sp.row(j).t();
            lincomb_fail(j) = lincomb(0);
            logplkd -= log(sum(exp(lincomb)));
          }
        }
        logplkd += accu(lincomb_fail);
      }
    }
  }
  logplkd /= Z_tv.n_rows;
  return logplkd;
}

List stepinc_fixtra(const mat &Z_tv, const mat &B_spline, const mat &theta,
                    const mat &Z_ti, const vec &beta_ti, const bool &ti,
                    const unsigned int n_strata,
                    vector<uvec> &idx_B_sp, vector<uvec> &idx_fail,
                    vector<vector<unsigned int>> &idx_Z_strata,
                    vector<vector<unsigned int>> &istart,
                    vector<vector<unsigned int>> &iend,
                    const string &method="Newton", const double &lambda=1e8,
                    const bool &parallel=false, const unsigned int &threads=1) {

  vec grad; mat info; // gradient and info matrix
  if (ti) {
    grad = zeros<vec>(theta.n_elem+beta_ti.n_elem);
    info = zeros<mat>(theta.n_elem+beta_ti.n_elem, theta.n_elem+beta_ti.n_elem);
  } else {
    grad = zeros<vec>(theta.n_elem);
    info = zeros<mat>(theta.n_elem, theta.n_elem);
  }
  for (unsigned int i = 0; i < n_strata; ++i) {
    mat Z_tv_strata = Z_tv.rows(idx_Z_strata[i][0], idx_Z_strata[i][1]);
    mat Z_ti_strata; vec Z_ti_beta_ti;
    if (ti) {
      Z_ti_strata = Z_ti.rows(idx_Z_strata[i][0], idx_Z_strata[i][1]);
      Z_ti_beta_ti = Z_ti_strata * beta_ti;
    }
    mat Z_tv_theta = Z_tv_strata * theta;
    mat B_sp = B_spline.rows(idx_B_sp[i]);
    unsigned int n_Z_tv_theta = Z_tv_theta.n_rows;
    if (parallel) {
      omp_set_num_threads(threads);
      #pragma omp parallel
      {
        unsigned int id, j;
        id = omp_get_thread_num();
        vec grad_tmp(size(grad), fill::zeros);
        mat info_tmp(size(info), fill::zeros);
        if (ti) {
          for (j = istart[i][id]; j < iend[i][id]; ++j) {
            vec B_sp_tmp = B_sp.row(j).t();
            unsigned int arstart = idx_fail[i](j), arend = n_Z_tv_theta-1;
            vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp +
              Z_ti_beta_ti.subvec(arstart,arend));
            double S0 = accu(exp_lincomb);
            mat Z_tv_exp =
              Z_tv_strata.rows(arstart,arend).each_col() % exp_lincomb;
            mat Z_ti_exp =
              Z_ti_strata.rows(arstart,arend).each_col() % exp_lincomb;
            vec S1_tv = sum(Z_tv_exp).t(), S1_ti = sum(Z_ti_exp).t();
            vec S1 = join_cols(kron(S1_tv,B_sp_tmp), S1_ti)/S0;
            mat S2_11 = kron(Z_tv_strata.rows(arstart,arend).t()*Z_tv_exp,
                             B_sp_tmp*B_sp_tmp.t())/S0,
                S2_12 = kron(Z_tv_strata.rows(arstart,arend).t()*Z_ti_exp,
                             B_sp_tmp)/S0,
                S2_22 = Z_ti_strata.rows(arstart,arend).t()*Z_ti_exp/S0;
            grad_tmp += join_cols(kron(Z_tv_strata.row(arstart).t(), B_sp_tmp),
                                  Z_ti_strata.row(arstart).t()) - S1;
            info_tmp += join_cols(join_rows(S2_11, S2_12),
                                  join_rows(S2_12.t(), S2_22)) - S1*S1.t();
          }
        } else {
          for (j = istart[i][id]; j < iend[i][id]; ++j) {
            vec B_sp_tmp = B_sp.row(j).t();
            unsigned int arstart = idx_fail[i](j), arend = n_Z_tv_theta-1;
            vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp);
            double S0 = accu(exp_lincomb);
            mat Z_tv_exp =
              Z_tv_strata.rows(arstart,arend).each_col() % exp_lincomb;
            vec S1_tv = sum(Z_tv_exp).t();
            mat S2 = Z_tv_strata.rows(arstart,arend).t() * Z_tv_exp;
            grad_tmp += kron(Z_tv_strata.row(arstart).t()-S1_tv/S0, B_sp_tmp);
            info_tmp +=
              kron(S2/S0-S1_tv*S1_tv.t()/pow(S0,2),B_sp_tmp*B_sp_tmp.t());
          }
        }
        #pragma omp critical (gradient)
          grad += grad_tmp;
        #pragma omp critical (information)
          info += info_tmp;
      }
    } else {
      if (ti) {
        for (unsigned int j = 0; j < idx_fail[i].n_elem; ++j) {
          vec B_sp_tmp = B_sp.row(j).t();
          unsigned int arstart = idx_fail[i](j), arend = n_Z_tv_theta-1;
          vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp +
            Z_ti_beta_ti.subvec(arstart,arend));
          double S0 = accu(exp_lincomb);
          mat Z_tv_exp =
            Z_tv_strata.rows(arstart,arend).each_col() % exp_lincomb;
          mat Z_ti_exp =
            Z_ti_strata.rows(arstart,arend).each_col() % exp_lincomb;
          vec S1_tv = sum(Z_tv_exp).t(), S1_ti = sum(Z_ti_exp).t();
          vec S1 = join_cols(kron(S1_tv,B_sp_tmp), S1_ti)/S0;
          mat S2_11 = kron(Z_tv_strata.rows(arstart,arend).t()*Z_tv_exp,
                           B_sp_tmp*B_sp_tmp.t())/S0,
              S2_12 = kron(Z_tv_strata.rows(arstart,arend).t()*Z_ti_exp,
                           B_sp_tmp)/S0,
              S2_22 = Z_ti_strata.rows(arstart,arend).t()*Z_ti_exp/S0;
          grad += join_cols(kron(Z_tv_strata.row(arstart).t(), B_sp_tmp),
                            Z_ti_strata.row(arstart).t()) - S1;
          info += join_cols(join_rows(S2_11, S2_12),
                            join_rows(S2_12.t(), S2_22)) - S1*S1.t();
        }
      } else {
        for (unsigned int j = 0; j < idx_fail[i].n_elem; ++j) {
          vec B_sp_tmp = B_sp.row(j).t();
          unsigned int arstart = idx_fail[i](j), arend = n_Z_tv_theta-1;
          vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp);
          double S0 = accu(exp_lincomb);
          mat Z_tv_exp =
            Z_tv_strata.rows(arstart,arend).each_col() % exp_lincomb;
          vec S1_tv = sum(Z_tv_exp).t();
          mat S2 = Z_tv_strata.rows(arstart,arend).t() * Z_tv_exp;
          grad += kron(Z_tv_strata.row(arstart).t()-S1_tv/S0, B_sp_tmp);
          info += kron(S2/S0-S1_tv*S1_tv.t()/pow(S0,2),B_sp_tmp*B_sp_tmp.t());
        }
      }
    }
  }
  vec step; // Newton step
  if (method=="ProxN") {
    info.diag() += Z_tv.n_rows / lambda;
    step = solve(info, grad, solve_opts::fast+solve_opts::likely_sympd);
    info.diag() -= Z_tv.n_rows / lambda;
  } else if (method=="Newton") {
    step = solve(info, grad, solve_opts::fast+solve_opts::likely_sympd);
  }
  double inc = dot(grad, step) / Z_tv.n_rows; // square of Newton increment
  if (ti) {
    return List::create(_["step_tv"]=reshape(step.head(theta.n_elem),
                                      size(theta.t())).t(),
                        _["step_ti"]=step.tail(beta_ti.n_elem),
                        _["grad"]=grad, _["info"]=info, _["inc"]=inc);
  } else {
    return List::create(_["step"]=reshape(step, size(theta.t())).t(),
                        _["grad"]=grad, _["info"]=info, _["inc"]=inc);
  }
}

List stepinc_fixtra_bresties(const mat &Z_tv, const mat &B_spline,
                             const mat &theta, const mat &Z_ti,
                             const vec &beta_ti, const bool &ti,
                             const unsigned int n_strata,
                             vector<uvec> &idx_B_sp,
                             vector<vector<uvec>> &idx_fail,
                             vector<vector<unsigned int>> &idx_Z_strata,
                             vector<vector<unsigned int>> &istart,
                             vector<vector<unsigned int>> &iend,
                             const string &method="Newton",
                             const double &lambda=1e8,
                             const bool &parallel=false,
                             const unsigned int &threads=1) {
  
  vec grad; mat info; // gradient and info matrix
  if (ti) {
    grad = zeros<vec>(theta.n_elem+beta_ti.n_elem);
    info = zeros<mat>(theta.n_elem+beta_ti.n_elem, theta.n_elem+beta_ti.n_elem);
  } else {
    grad = zeros<vec>(theta.n_elem);
    info = zeros<mat>(theta.n_elem, theta.n_elem);
  }
  for (unsigned int i = 0; i < n_strata; ++i) {
    mat Z_tv_strata = Z_tv.rows(idx_Z_strata[i][0], idx_Z_strata[i][1]);
    mat Z_ti_strata; vec Z_ti_beta_ti;
    if (ti) {
      Z_ti_strata = Z_ti.rows(idx_Z_strata[i][0], idx_Z_strata[i][1]);
      Z_ti_beta_ti = Z_ti_strata * beta_ti;
    }
    mat Z_tv_theta = Z_tv_strata * theta;
    mat B_sp = B_spline.rows(idx_B_sp[i]);
    unsigned int n_Z_tv_theta = Z_tv_theta.n_rows;
    if (parallel) {
      omp_set_num_threads(threads);
      #pragma omp parallel
      {
        unsigned int id, j;
        id = omp_get_thread_num();
        vec grad_tmp(size(grad), fill::zeros);
        mat info_tmp(size(info), fill::zeros);
        if (ti) {
          for (j = istart[i][id]; j < iend[i][id]; ++j) {
            vec B_sp_tmp = B_sp.row(j).t();
            unsigned int arstart = idx_fail[i][j](0), arend = n_Z_tv_theta-1, 
              nar = idx_fail[i][j].n_elem;
            vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp +
              Z_ti_beta_ti.subvec(arstart,arend));
            double S0 = accu(exp_lincomb);
            mat Z_tv_exp =
              Z_tv_strata.rows(arstart,arend).each_col() % exp_lincomb;
            mat Z_ti_exp =
              Z_ti_strata.rows(arstart,arend).each_col() % exp_lincomb;
            vec S1_tv = sum(Z_tv_exp).t(), S1_ti = sum(Z_ti_exp).t();
            vec S1 = join_cols(kron(S1_tv,B_sp_tmp), S1_ti)/S0;
            mat S2_11 = kron(Z_tv_strata.rows(arstart,arend).t()*Z_tv_exp,
                             B_sp_tmp*B_sp_tmp.t())/S0,
                S2_12 = kron(Z_tv_strata.rows(arstart,arend).t()*Z_ti_exp,
                             B_sp_tmp)/S0,
                S2_22 = Z_ti_strata.rows(arstart,arend).t()*Z_ti_exp/S0;
            grad_tmp += join_cols(kron(sum(Z_tv_strata.rows(idx_fail[i][j])).t(), B_sp_tmp),
                                  sum(Z_ti_strata.rows(idx_fail[i][j])).t()) - nar*S1;
            info_tmp += nar*(join_cols(join_rows(S2_11, S2_12),
                                       join_rows(S2_12.t(), S2_22)) - S1*S1.t());
          }
        } else {
          for (j = istart[i][id]; j < iend[i][id]; ++j) {
            vec B_sp_tmp = B_sp.row(j).t();
            unsigned int arstart = idx_fail[i][j](0), arend = n_Z_tv_theta-1,
              nar = idx_fail[i][j].n_elem;
            vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp);
            double S0 = accu(exp_lincomb);
            mat Z_tv_exp =
              Z_tv_strata.rows(arstart,arend).each_col() % exp_lincomb;
            vec S1_tv = sum(Z_tv_exp).t();
            mat S2 = Z_tv_strata.rows(arstart,arend).t() * Z_tv_exp;
            grad_tmp += kron(sum(Z_tv_strata.rows(idx_fail[i][j])).t()-nar*S1_tv/S0, B_sp_tmp);
            info_tmp += nar*
              kron(S2/S0-S1_tv*S1_tv.t()/pow(S0,2),B_sp_tmp*B_sp_tmp.t());
          }
        }
        #pragma omp critical (gradient)
        grad += grad_tmp;
        #pragma omp critical (information)
        info += info_tmp;
      }
    } else {
      if (ti) {
        for (unsigned int j = 0; j < idx_fail[i].size(); ++j) {
          vec B_sp_tmp = B_sp.row(j).t();
          unsigned int arstart = idx_fail[i][j](0), arend = n_Z_tv_theta-1,
            nar = idx_fail[i][j].n_elem;
          vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp +
            Z_ti_beta_ti.subvec(arstart,arend));
          double S0 = accu(exp_lincomb);
          mat Z_tv_exp =
            Z_tv_strata.rows(arstart,arend).each_col() % exp_lincomb;
          mat Z_ti_exp =
            Z_ti_strata.rows(arstart,arend).each_col() % exp_lincomb;
          vec S1_tv = sum(Z_tv_exp).t(), S1_ti = sum(Z_ti_exp).t();
          vec S1 = join_cols(kron(S1_tv,B_sp_tmp), S1_ti)/S0;
          mat S2_11 = kron(Z_tv_strata.rows(arstart,arend).t()*Z_tv_exp,
                           B_sp_tmp*B_sp_tmp.t())/S0,
              S2_12 = kron(Z_tv_strata.rows(arstart,arend).t()*Z_ti_exp,
                           B_sp_tmp)/S0,
              S2_22 = Z_ti_strata.rows(arstart,arend).t()*Z_ti_exp/S0;
          grad += join_cols(kron(sum(Z_tv_strata.rows(idx_fail[i][j])).t(), B_sp_tmp),
                            sum(Z_ti_strata.rows(idx_fail[i][j])).t()) - nar*S1;
          info += nar*(join_cols(join_rows(S2_11, S2_12),
                            join_rows(S2_12.t(), S2_22)) - S1*S1.t());
        }
      } else {
        for (unsigned int j = 0; j < idx_fail[i].size(); ++j) {
          vec B_sp_tmp = B_sp.row(j).t();
          unsigned int arstart = idx_fail[i][j](0), arend = n_Z_tv_theta-1,
            nar = idx_fail[i][j].n_elem;
          vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp);
          double S0 = accu(exp_lincomb);
          mat Z_tv_exp =
            Z_tv_strata.rows(arstart,arend).each_col() % exp_lincomb;
          vec S1_tv = sum(Z_tv_exp).t();
          mat S2 = Z_tv_strata.rows(arstart,arend).t() * Z_tv_exp;
          grad += kron(sum(Z_tv_strata.rows(idx_fail[i][j])).t()-nar*S1_tv/S0, B_sp_tmp);
          info += nar*
            (kron(S2/S0-S1_tv*S1_tv.t()/pow(S0,2),B_sp_tmp*B_sp_tmp.t()));
        }
      }
    }
  }
  vec step; // Newton step
  if (method=="ProxN") {
    info.diag() += Z_tv.n_rows / lambda;
    step = solve(info, grad, solve_opts::fast+solve_opts::likely_sympd);
    info.diag() -= Z_tv.n_rows / lambda;
  } else if (method=="Newton") {
    step = solve(info, grad, solve_opts::fast+solve_opts::likely_sympd);
  }
  double inc = dot(grad, step) / Z_tv.n_rows; // square of Newton increment
  if (ti) {
    return List::create(_["step_tv"]=reshape(step.head(theta.n_elem),
                                      size(theta.t())).t(),
                                      _["step_ti"]=step.tail(beta_ti.n_elem),
                                      _["grad"]=grad, _["info"]=info, _["inc"]=inc);
  } else {
    return List::create(_["step"]=reshape(step, size(theta.t())).t(),
                        _["grad"]=grad, _["info"]=info, _["inc"]=inc);
  }
}

List stepinc_ranstra(const vec &event, const mat &Z_tv, const mat &B_spline, 
                     const mat &theta, const mat &Z_ti, const vec &beta_ti, 
                     const bool &ti, const unsigned int &n_stra, 
                     vector<uvec> &idx_stra, vector<uvec> &idx_fail,
                     const string &method="Newton", const double &lambda=1e8,
                     const bool &parallel=false, const unsigned int &threads=1) {
  
  vec grad; mat info; // gradient and info matrix
  if (ti) {
    grad = zeros<vec>(theta.n_elem+beta_ti.n_elem);
    info = zeros<mat>(theta.n_elem+beta_ti.n_elem, theta.n_elem+beta_ti.n_elem);
  } else {
    grad = zeros<vec>(theta.n_elem);
    info = zeros<mat>(theta.n_elem, theta.n_elem);
  }
  if (parallel) {
    uvec quant = conv_to<uvec>::from(
      floor(quantile(regspace<uvec>(0,n_stra),regspace(0.0,1.0/threads,1.0))));
    omp_set_num_threads(threads);
    #pragma omp parallel
    {
      unsigned int id = omp_get_thread_num();
      vec grad_tmp(size(grad), fill::zeros);
      mat info_tmp(size(info), fill::zeros);
      for (unsigned int i = quant[id]; i < quant[id+1]; ++i) {
        uvec idx_stra_tmp = idx_stra[i];
        uvec idx_fail_tmp = idx_fail[i];
        mat Z_tv_stra = Z_tv.rows(idx_stra_tmp);
        mat Z_ti_stra; vec Z_ti_beta_ti;
        if (ti) {
          Z_ti_stra = Z_ti.rows(idx_stra_tmp);
          Z_ti_beta_ti = Z_ti_stra * beta_ti;
        }
        mat Z_tv_theta = Z_tv_stra * theta;
        mat B_sp = B_spline.rows(idx_stra_tmp.elem(idx_fail_tmp));
        if (ti) {
          for (unsigned int j = 0; j < idx_fail_tmp.n_elem; ++j) {
            vec B_sp_tmp = B_sp.row(j).t();
            unsigned int arstart = idx_fail_tmp(j), arend = Z_tv_theta.n_rows-1;
            vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp +
              Z_ti_beta_ti.subvec(arstart,arend));
            double S0 = accu(exp_lincomb);
            mat Z_tv_exp =
              Z_tv_stra.rows(arstart,arend).each_col() % exp_lincomb;
            mat Z_ti_exp =
              Z_ti_stra.rows(arstart,arend).each_col() % exp_lincomb;
            vec S1_tv = sum(Z_tv_exp).t(), S1_ti = sum(Z_ti_exp).t();
            vec S1 = join_cols(kron(S1_tv,B_sp_tmp), S1_ti)/S0;
            mat S2_11 = kron(Z_tv_stra.rows(arstart,arend).t()*Z_tv_exp,
                             B_sp_tmp*B_sp_tmp.t())/S0,
                S2_12 = kron(Z_tv_stra.rows(arstart,arend).t()*Z_ti_exp,
                             B_sp_tmp)/S0,
                S2_22 = Z_ti_stra.rows(arstart,arend).t()*Z_ti_exp/S0;
            grad_tmp += join_cols(kron(Z_tv_stra.row(arstart).t(), B_sp_tmp),
                                  Z_ti_stra.row(arstart).t()) - S1;
            info_tmp += join_cols(join_rows(S2_11, S2_12),
                                  join_rows(S2_12.t(), S2_22)) - S1*S1.t();
          }
        } else {
          for (unsigned int j = 0; j < idx_fail_tmp.n_elem; ++j) {
            vec B_sp_tmp = B_sp.row(j).t();
            unsigned int arstart = idx_fail_tmp(j), arend = Z_tv_theta.n_rows-1;
            vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp);
            double S0 = accu(exp_lincomb);
            mat Z_tv_exp =
              Z_tv_stra.rows(arstart,arend).each_col() % exp_lincomb;
            vec S1_tv = sum(Z_tv_exp).t();
            mat S2 = Z_tv_stra.rows(arstart,arend).t() * Z_tv_exp;
            grad_tmp += kron(Z_tv_stra.row(arstart).t()-S1_tv/S0, B_sp_tmp);
            info_tmp += kron(S2/S0-S1_tv*S1_tv.t()/pow(S0,2),B_sp_tmp*B_sp_tmp.t());
          }
        }
      }
      #pragma omp critical (gradient)
        grad += grad_tmp;
      #pragma omp critical (information)
        info += info_tmp;
    }
  } else {
    for (unsigned int i = 0; i < n_stra; ++i) {
      uvec idx_stra_tmp = idx_stra[i];
      uvec idx_fail_tmp = idx_fail[i];
      mat Z_tv_stra = Z_tv.rows(idx_stra_tmp);
      mat Z_ti_stra; vec Z_ti_beta_ti;
      if (ti) {
        Z_ti_stra = Z_ti.rows(idx_stra_tmp);
        Z_ti_beta_ti = Z_ti_stra * beta_ti;
      }
      mat Z_tv_theta = Z_tv_stra * theta;
      mat B_sp = B_spline.rows(idx_stra_tmp.elem(idx_fail_tmp));
      if (ti) {
        for (unsigned int j = 0; j < idx_fail_tmp.n_elem; ++j) {
          vec B_sp_tmp = B_sp.row(j).t();
          unsigned int arstart = idx_fail_tmp(j), arend = Z_tv_theta.n_rows-1;
          vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp +
            Z_ti_beta_ti.subvec(arstart,arend));
          double S0 = accu(exp_lincomb);
          mat Z_tv_exp =
            Z_tv_stra.rows(arstart,arend).each_col() % exp_lincomb;
          mat Z_ti_exp =
            Z_ti_stra.rows(arstart,arend).each_col() % exp_lincomb;
          vec S1_tv = sum(Z_tv_exp).t(), S1_ti = sum(Z_ti_exp).t();
          vec S1 = join_cols(kron(S1_tv,B_sp_tmp), S1_ti)/S0;
          mat S2_11 = kron(Z_tv_stra.rows(arstart,arend).t()*Z_tv_exp,
                           B_sp_tmp*B_sp_tmp.t())/S0,
              S2_12 = kron(Z_tv_stra.rows(arstart,arend).t()*Z_ti_exp,
                           B_sp_tmp)/S0,
              S2_22 = Z_ti_stra.rows(arstart,arend).t()*Z_ti_exp/S0;
          grad += join_cols(kron(Z_tv_stra.row(arstart).t(), B_sp_tmp),
                            Z_ti_stra.row(arstart).t()) - S1;
          info += join_cols(join_rows(S2_11, S2_12),
                            join_rows(S2_12.t(), S2_22)) - S1*S1.t();
        }
      } else {
        for (unsigned int j = 0; j < idx_fail_tmp.n_elem; ++j) {
          vec B_sp_tmp = B_sp.row(j).t();
          unsigned int arstart = idx_fail_tmp(j), arend = Z_tv_theta.n_rows-1;
          vec exp_lincomb = exp(Z_tv_theta.rows(arstart,arend) * B_sp_tmp);
          double S0 = accu(exp_lincomb);
          mat Z_tv_exp =
            Z_tv_stra.rows(arstart,arend).each_col() % exp_lincomb;
          vec S1_tv = sum(Z_tv_exp).t();
          mat S2 = Z_tv_stra.rows(arstart,arend).t() * Z_tv_exp;
          grad += kron(Z_tv_stra.row(arstart).t()-S1_tv/S0, B_sp_tmp);
          info += kron(S2/S0-S1_tv*S1_tv.t()/pow(S0,2),B_sp_tmp*B_sp_tmp.t());
        }
      }
    }
  }
  vec step; // Newton step
  if (method=="ProxN") {
    info.diag() += Z_tv.n_rows / lambda;
    step = solve(info, grad, solve_opts::fast+solve_opts::likely_sympd);
    info.diag() -= Z_tv.n_rows / lambda;
  } else if (method=="Newton") {
    step = solve(info, grad, solve_opts::fast+solve_opts::likely_sympd);
  }
  double inc = dot(grad, step) / Z_tv.n_rows; // square of Newton increment
  if (ti) {
    return List::create(_["step_tv"]=reshape(step.head(theta.n_elem),
                                      size(theta.t())).t(),
                                      _["step_ti"]=step.tail(beta_ti.n_elem),
                                      _["grad"]=grad, _["info"]=info, _["inc"]=inc);
  } else {
    return List::create(_["step"]=reshape(step, size(theta.t())).t(),
                        _["grad"]=grad, _["info"]=info, _["inc"]=inc);
  }
}

// [[Rcpp::export]]
List surtiver_fixtra_fit(const vec &event, const IntegerVector &count_strata,
                         const mat &Z_tv, const mat &B_spline, const mat &theta_init,
                         const mat &Z_ti, const vec &beta_ti_init,
                         const string &method="Newton",
                         const double lambda=1e8, const double &factor=1.0,
                         const bool &parallel=false, const unsigned int &threads=1,
                         const double &tol=1e-10, const unsigned int &iter_max=20,
                         const double &s=1e-2, const double &t=0.6,
                         const string &btr="dynamic",
                         const string &stop="incre") {

  bool ti = norm(Z_ti, "inf") > sqrt(datum::eps);
  IntegerVector cumsum_strata = cumsum(count_strata);
  unsigned int n_strata = cumsum_strata.length();
  cumsum_strata.push_front(0);

  vector<uvec> idx_fail, idx_B_sp;
  vector<vector<unsigned int>> idx_Z_strata;
  // each element of idx_Z_strata contains start/end row indices of Z_strata
  vector<unsigned int> n_fail, n_Z_strata;
  for (unsigned int i = 0; i < n_strata; ++i) {
    uvec idx_fail_tmp =
      find(event.rows(cumsum_strata[i], cumsum_strata[i+1]-1)==1);
    n_fail.push_back(idx_fail_tmp.n_elem);
    vector<unsigned int> idx_Z_strata_tmp;
    idx_Z_strata_tmp.push_back(cumsum_strata[i]+idx_fail_tmp(0));
    idx_Z_strata_tmp.push_back(cumsum_strata[i+1]-1);
    idx_Z_strata.push_back(idx_Z_strata_tmp);
    n_Z_strata.push_back(cumsum_strata[i+1]-cumsum_strata[i]-
      idx_fail_tmp(0));
    idx_B_sp.push_back(cumsum_strata[i]+idx_fail_tmp);
    idx_fail_tmp -= idx_fail_tmp(0);
    idx_fail.push_back(idx_fail_tmp);
  }

  // istart and iend for each thread when parallel=true
  vector<vec> cumsum_ar;
  vector<vector<unsigned int>> istart, iend;
  if (parallel) {
    for (unsigned int i = 0; i < n_strata; ++i) {
      double scale_fac = as_scalar(idx_fail[i].tail(1));
      cumsum_ar.push_back(
        (double)n_Z_strata[i] / scale_fac * regspace(1,n_fail[i]) -
          cumsum(conv_to<vec>::from(idx_fail[i])/scale_fac));
      vector<unsigned int> istart_tmp, iend_tmp;
      for (unsigned int id = 0; id < threads; ++id) {
        istart_tmp.push_back(as_scalar(find(cumsum_ar[i] >=
          cumsum_ar[i](n_fail[i]-1)/(double)threads*id, 1)));
        iend_tmp.push_back(as_scalar(find(cumsum_ar[i] >=
          cumsum_ar[i](n_fail[i]-1)/(double)threads*(id+1), 1)));
        if (id == threads-1) {
          iend_tmp.pop_back();
          iend_tmp.push_back(n_fail[i]);
        }
      }
      istart.push_back(istart_tmp);
      iend.push_back(iend_tmp);
    }
  }
  // iterative algorithm with backtracking line search
  unsigned int iter = 0, btr_max = 1000, btr_ct = 0;
  mat theta = theta_init; vec beta_ti = beta_ti_init;
  List theta_list = List::create(theta), beta_ti_list = List::create(beta_ti);
  double crit = 1.0, v = 1.0, logplkd_init = 0, logplkd, diff_logplkd, 
    inc, rhs_btr = 0;
  List objfun_list, update_list;
  NumericVector logplkd_vec;
  objfun_list = objfun_fixtra(Z_tv, B_spline, theta, Z_ti, beta_ti, ti, n_strata,
                              idx_B_sp, idx_fail, n_Z_strata, idx_Z_strata,
                              istart, iend, parallel, threads);
  logplkd = objfun_list["logplkd"];
  logplkd_vec.push_back(logplkd);
  while (iter < iter_max && crit > tol) {
    ++iter;
    update_list = stepinc_fixtra(Z_tv, B_spline, theta, Z_ti, beta_ti, ti, n_strata,
                          idx_B_sp, idx_fail, idx_Z_strata, istart, iend,
                          method, lambda*pow(factor,iter-1), parallel, threads);
    v = 1.0; // reset step size
    if (ti) {
      mat step_tv = update_list["step_tv"];
      vec step_ti = update_list["step_ti"];
      inc = update_list["inc"];
      if (btr=="none") {
        theta += step_tv;
        beta_ti += step_ti;
        theta_list.push_back(theta); beta_ti_list.push_back(beta_ti);
        crit = inc / 2;
        objfun_list = objfun_fixtra(Z_tv, B_spline, theta, Z_ti, beta_ti, ti, n_strata, 
                                    idx_B_sp, idx_fail, n_Z_strata, idx_Z_strata,
                                    istart, iend, parallel, threads);
        logplkd = objfun_list["logplkd"];
      } else {
        mat theta_tmp = theta + step_tv;
        vec beta_ti_tmp = beta_ti + step_ti;
        objfun_list = objfun_fixtra(Z_tv, B_spline, theta_tmp, Z_ti, beta_ti_tmp,
                                ti, n_strata, idx_B_sp, idx_fail, n_Z_strata,
                                idx_Z_strata, istart, iend,
                                parallel, threads);
        double logplkd_tmp = objfun_list["logplkd"];
        diff_logplkd = logplkd_tmp - logplkd;
        if (btr=="dynamic")      rhs_btr = inc;
        else if (btr=="static")  rhs_btr = 1.0;
        while (diff_logplkd < s * v * rhs_btr && btr_ct < btr_max) {
          ++btr_ct;
          v *= t;
          theta_tmp = theta + v * step_tv;
          beta_ti_tmp = beta_ti + step_ti;
          objfun_list = objfun_fixtra(Z_tv, B_spline, theta_tmp, Z_ti, beta_ti_tmp,
                                  ti, n_strata, idx_B_sp, idx_fail,
                                  n_Z_strata, idx_Z_strata, istart, iend,
                                  parallel, threads);
          double logplkd_tmp = objfun_list["logplkd"];
          diff_logplkd = logplkd_tmp - logplkd;
        }
        theta = theta_tmp; beta_ti = beta_ti_tmp;
        theta_list.push_back(theta); beta_ti_list.push_back(beta_ti);
        if (iter==1) logplkd_init = logplkd;
        if (stop=="incre")
          crit = inc / 2;
        else if (stop=="relch")
          crit = abs(diff_logplkd/(diff_logplkd+logplkd));
        else if (stop=="ratch")
          crit = abs(diff_logplkd/(diff_logplkd+logplkd-logplkd_init));
        logplkd += diff_logplkd;
      }
    } else {
      mat step = update_list["step"];
      inc = update_list["inc"];
      if (btr=="none") {
        theta += step;
        crit = inc / 2;
        objfun_list = objfun_fixtra(Z_tv, B_spline, theta, Z_ti, beta_ti, ti,
                                    n_strata, idx_B_sp, idx_fail, n_Z_strata,
                                    idx_Z_strata, istart, iend, parallel, threads);
        logplkd = objfun_list["logplkd"];
      } else {
        mat theta_tmp = theta + step;
        objfun_list = objfun_fixtra(Z_tv, B_spline, theta_tmp, Z_ti, beta_ti, ti,
                                n_strata, idx_B_sp, idx_fail, n_Z_strata,
                                idx_Z_strata, istart, iend, parallel,
                                threads);
        double logplkd_tmp = objfun_list["logplkd"];
        diff_logplkd = logplkd_tmp - logplkd;
        if (btr=="dynamic")      rhs_btr = inc;
        else if (btr=="static")  rhs_btr = 1.0;
        while (diff_logplkd < s * v * rhs_btr && btr_ct < btr_max) {
          ++btr_ct;
          v *= t;
          theta_tmp = theta + v * step;
          objfun_list = objfun_fixtra(Z_tv, B_spline, theta_tmp, Z_ti, beta_ti, ti,
                                      n_strata, idx_B_sp, idx_fail, n_Z_strata,
                                      idx_Z_strata, istart, iend, parallel,
                                      threads);
          double logplkd_tmp = objfun_list["logplkd"];
          diff_logplkd = logplkd_tmp - logplkd;
        }
        theta = theta_tmp;
        theta_list.push_back(theta);
        if (iter==1) logplkd_init = logplkd;
        if (stop=="incre")
          crit = inc / 2;
        else if (stop=="relch")
          crit = abs(diff_logplkd/(diff_logplkd+logplkd));
        else if (stop=="ratch")
          crit = abs(diff_logplkd/(diff_logplkd+logplkd-logplkd_init));
        logplkd += diff_logplkd;
      }
    }
    cout << "Iter " << iter << ": Obj fun = " << setprecision(7) << fixed << 
      logplkd << "; Stopping crit = " << setprecision(7) << scientific << 
        crit << ";" << endl;
    logplkd_vec.push_back(logplkd);
  }
  cout << "Algorithm converged after " << iter << " iterations!" << endl;
  CharacterVector names_strata = count_strata.names();
  vector<vec> hazard = objfun_list["hazard"];
  List hazard_list;
  for (unsigned int i = 0; i < n_strata; ++i) {
    hazard_list.push_back(hazard[i], as<string>(names_strata[i]));
  }
  if (ti) {
    return List::create(_["ctrl.pts"]=theta,
                        _["ctrl.pts.hist"]=theta_list,
                        _["tief"]=beta_ti,
                        _["tief.hist"]=beta_ti_list,
                        _["info"]=update_list["info"], 
                        _["logplkd"]=logplkd*event.n_elem,
                        _["logplkd.hist"]=logplkd_vec*event.n_elem,
                        _["btr.ct"]=btr_ct,
                        _["hazard"]=hazard_list);
  } else {
    return List::create(_["ctrl.pts"]=theta, _["ctrl.pts.hist"]=theta_list,
                        _["info"]=update_list["info"],
                        _["logplkd"]=logplkd*event.n_elem,
                        _["logplkd.hist"]=logplkd_vec*event.n_elem,
                        _["btr.ct"]=btr_ct,
                        _["hazard"]=hazard_list);
  }
}

// [[Rcpp::export]]
List surtiver_fixtra_bresties_fit(const vec &event, const vec &time, 
                                  const IntegerVector &count_strata,
                                  const mat &Z_tv, const mat &B_spline, 
                                  const mat &theta_init,
                                  const mat &Z_ti, const vec &beta_ti_init,
                                  const string &method="Newton",
                                  const double lambda=1e8, 
                                  const double &factor=1.0,
                                  const bool &parallel=false, 
                                  const unsigned int &threads=1,
                                  const double &tol=1e-10, 
                                  const unsigned int &iter_max=20,
                                  const double &s=1e-2, const double &t=0.6,
                                  const string &btr="dynamic",
                                  const string &stop="incre") {
  // B_spline only includes distinct failure times sorted by strata
  bool ti = norm(Z_ti, "inf") > sqrt(datum::eps);
  IntegerVector cumsum_strata = cumsum(count_strata);
  unsigned int n_strata = cumsum_strata.length();
  cumsum_strata.push_front(0);
  vector<vector<uvec>> idx_fail;
  vector<vector<unsigned int>> idx_fail_1st, idx_Z_strata;
  vector<unsigned int> n_fail_time, n_Z_strata;
  // each element of idx_Z_strata contains start/end row indices of Z_strata
  for (unsigned int i = 0; i < n_strata; ++i) {
    vec event_tmp = event.rows(cumsum_strata[i], cumsum_strata[i+1]-1),
      time_tmp = time.rows(cumsum_strata[i], cumsum_strata[i+1]-1);
    uvec idx_fail_tmp = find(event_tmp==1);
    vector<unsigned int> idx_Z_strata_tmp;
    idx_Z_strata_tmp.push_back(cumsum_strata[i]+idx_fail_tmp(0));
    idx_Z_strata_tmp.push_back(cumsum_strata[i+1]-1);
    idx_Z_strata.push_back(idx_Z_strata_tmp);
    n_Z_strata.push_back(cumsum_strata[i+1]-cumsum_strata[i]-
      idx_fail_tmp(0));
    vec time_fail_tmp = time_tmp.elem(idx_fail_tmp);
    idx_fail_tmp -= idx_fail_tmp(0);
    vec uniq_t = unique(time_fail_tmp);
    n_fail_time.push_back(uniq_t.n_elem);
    vector<uvec> idx_fail_tmp_tmp;
    vector<unsigned int> idx_fail_1st_tmp;
    for (vec::iterator j = uniq_t.begin(); j < uniq_t.end(); ++j) {
      uvec tmp = idx_fail_tmp.elem(find(time_fail_tmp==*j));
      idx_fail_tmp_tmp.push_back(tmp);
      idx_fail_1st_tmp.push_back(tmp[0]);
    }
    idx_fail.push_back(idx_fail_tmp_tmp);
    idx_fail_1st.push_back(idx_fail_1st_tmp);
  }
  IntegerVector n_failtime = wrap(n_fail_time);
  IntegerVector cumsum_failtime = cumsum(n_failtime);
  cumsum_failtime.push_front(0);
  vector<uvec> idx_B_sp;
  for (unsigned int i = 0; i < n_strata; ++i) {
    idx_B_sp.push_back(regspace<uvec>(cumsum_failtime[i],
                                      cumsum_failtime[i+1]-1));
  }
  // istart and iend for each thread when parallel=true
  vector<vec> cumsum_ar;
  vector<vector<unsigned int>> istart, iend;
  if (parallel) {
    for (unsigned int i = 0; i < n_strata; ++i) {
      double scale_fac = as_scalar(idx_fail[i].back().tail(1));
      cumsum_ar.push_back(
        (double)n_Z_strata[i] / scale_fac * regspace(1, idx_fail[i].size()) -
          cumsum(conv_to<vec>::from(idx_fail_1st[i])/scale_fac));
      vector<unsigned int> istart_tmp, iend_tmp;
      for (unsigned int id = 0; id < threads; ++id) {
        istart_tmp.push_back(as_scalar(find(cumsum_ar[i] >=
          cumsum_ar[i](idx_fail[i].size()-1)/(double)threads*id, 1)));
        iend_tmp.push_back(as_scalar(find(cumsum_ar[i] >=
          cumsum_ar[i](idx_fail[i].size()-1)/(double)threads*(id+1), 1)));
        if (id == threads-1) {
          iend_tmp.pop_back();
          iend_tmp.push_back(idx_fail[i].size());
        }
      }
      istart.push_back(istart_tmp);
      iend.push_back(iend_tmp);
    }
  }
  // iterative algorithm with backtracking line search
  unsigned int iter = 0, btr_max = 1000, btr_ct = 0;
  mat theta = theta_init; vec beta_ti = beta_ti_init;
  List theta_list = List::create(theta), beta_ti_list = List::create(beta_ti);
  double crit = 1.0, v = 1.0, logplkd_init = 0, logplkd, diff_logplkd,
    inc, rhs_btr = 0;
  List objfun_list, update_list;
  NumericVector logplkd_vec;
  objfun_list = obj_fixtra_bresties(Z_tv, B_spline, theta, Z_ti, beta_ti,
                                    ti, n_strata, idx_B_sp, idx_fail,
                                    n_Z_strata, idx_Z_strata,
                                    istart, iend, parallel, threads);
  logplkd = objfun_list["logplkd"];
  logplkd_vec.push_back(logplkd);
  while (iter < iter_max && crit > tol) {
    ++iter;
    update_list = stepinc_fixtra_bresties(Z_tv, B_spline, theta, Z_ti, beta_ti,
                                          ti, n_strata, idx_B_sp, idx_fail,
                                          idx_Z_strata, istart, iend, method,
                                          lambda*pow(factor,iter-1),
                                          parallel, threads);
    v = 1.0; // reset step size
    if (ti) {
      mat step_tv = update_list["step_tv"];
      vec step_ti = update_list["step_ti"];
      inc = update_list["inc"];
      if (btr=="none") {
        theta += step_tv;
        beta_ti += step_ti;
        theta_list.push_back(theta); beta_ti_list.push_back(beta_ti);
        crit = inc / 2;
        objfun_list = obj_fixtra_bresties(Z_tv, B_spline, theta, Z_ti, beta_ti,
                                          ti, n_strata, idx_B_sp, idx_fail,
                                          n_Z_strata, idx_Z_strata,
                                          istart, iend, parallel, threads);
        logplkd = objfun_list["logplkd"];
      } else {
        mat theta_tmp = theta + step_tv;
        vec beta_ti_tmp = beta_ti + step_ti;
        objfun_list = obj_fixtra_bresties(Z_tv, B_spline, theta_tmp, Z_ti,
                                          beta_ti_tmp, ti, n_strata, idx_B_sp,
                                          idx_fail, n_Z_strata, idx_Z_strata,
                                          istart, iend, parallel, threads);
        double logplkd_tmp = objfun_list["logplkd"];
        diff_logplkd = logplkd_tmp - logplkd;
        if (btr=="dynamic")      rhs_btr = inc;
        else if (btr=="static")  rhs_btr = 1.0;
        while (diff_logplkd < s * v * rhs_btr && btr_ct < btr_max) {
          ++btr_ct;
          v *= t;
          theta_tmp = theta + v * step_tv;
          beta_ti_tmp = beta_ti + step_ti;
          objfun_list = obj_fixtra_bresties(Z_tv, B_spline, theta_tmp, Z_ti,
                                            beta_ti_tmp, ti, n_strata, idx_B_sp,
                                            idx_fail, n_Z_strata, idx_Z_strata,
                                            istart, iend, parallel, threads);
          double logplkd_tmp = objfun_list["logplkd"];
          diff_logplkd = logplkd_tmp - logplkd;
        }
        theta = theta_tmp; beta_ti = beta_ti_tmp;
        theta_list.push_back(theta); beta_ti_list.push_back(beta_ti);
        if (iter==1) logplkd_init = logplkd;
        if (stop=="incre")
          crit = inc / 2;
        else if (stop=="relch")
          crit = abs(diff_logplkd/(diff_logplkd+logplkd));
        else if (stop=="ratch")
          crit = abs(diff_logplkd/(diff_logplkd+logplkd-logplkd_init));
        logplkd += diff_logplkd;
      }
    } else {
      mat step = update_list["step"];
      inc = update_list["inc"];
      if (btr=="none") {
        theta += step;
        theta_list.push_back(theta);
        crit = inc / 2;
        objfun_list = obj_fixtra_bresties(Z_tv, B_spline, theta, Z_ti, beta_ti,
                                          ti, n_strata, idx_B_sp, idx_fail,
                                          n_Z_strata, idx_Z_strata,
                                          istart, iend, parallel, threads);
        logplkd = objfun_list["logplkd"];
      } else {
        mat theta_tmp = theta + step;
        objfun_list = obj_fixtra_bresties(Z_tv, B_spline, theta_tmp, Z_ti,
                                          beta_ti, ti, n_strata, idx_B_sp,
                                          idx_fail, n_Z_strata, idx_Z_strata,
                                          istart, iend, parallel, threads);
        double logplkd_tmp = objfun_list["logplkd"];
        diff_logplkd = logplkd_tmp - logplkd;
        if (btr=="dynamic")      rhs_btr = inc;
        else if (btr=="static")  rhs_btr = 1.0;
        while (diff_logplkd < s * v * rhs_btr && btr_ct < btr_max) {
          ++btr_ct;
          v *= t;
          theta_tmp = theta + v * step;
          objfun_list = obj_fixtra_bresties(Z_tv, B_spline, theta_tmp, Z_ti,
                                            beta_ti, ti, n_strata, idx_B_sp,
                                            idx_fail, n_Z_strata, idx_Z_strata,
                                            istart, iend, parallel, threads);
          double logplkd_tmp = objfun_list["logplkd"];
          diff_logplkd = logplkd_tmp - logplkd;
        }
        theta = theta_tmp;
        theta_list.push_back(theta);
        if (iter==1) logplkd_init = logplkd;
        if (stop=="incre")
          crit = inc / 2;
        else if (stop=="relch")
          crit = abs(diff_logplkd/(diff_logplkd+logplkd));
        else if (stop=="ratch")
          crit = abs(diff_logplkd/(diff_logplkd+logplkd-logplkd_init));
        logplkd += diff_logplkd;
      }
    }
    cout << "Iter " << iter << ": Obj fun = " << setprecision(7) << fixed <<
      logplkd << "; Stopping crit = " << setprecision(7) << scientific <<
        crit << ";" << endl;
    logplkd_vec.push_back(logplkd);
  }
  cout << "Algorithm converged after " << iter << " iterations!" << endl;
  CharacterVector names_strata = count_strata.names();
  vector<vec> hazard = objfun_list["hazard"];
  List hazard_list;
  for (unsigned int i = 0; i < n_strata; ++i) {
    hazard_list.push_back(hazard[i], as<string>(names_strata[i]));
  }
  if (ti) {
    return List::create(_["ctrl.pts"]=theta, _["ctrl.pts.hist"]=theta_list,
                        _["tief"]=beta_ti,
                        _["tief.hist"]=beta_ti_list,
                        _["info"]=update_list["info"],
                        _["logplkd"]=logplkd*event.n_elem,
                        _["logplkd.hist"]=logplkd_vec*event.n_elem,
                        _["btr.ct"]=btr_ct,
                        _["hazard"]=hazard_list);
  } else {
    return List::create(_["ctrl.pts"]=theta, _["ctrl.pts.hist"]=theta_list,
                        _["info"]=update_list["info"],
                        _["logplkd"]=logplkd*event.n_elem,
                        _["logplkd.hist"]=logplkd_vec*event.n_elem,
                        _["btr.ct"]=btr_ct,
                        _["hazard"]=hazard_list);
  }
}

// [[Rcpp::export]]
List surtiver_ranstra_fit(const vec &event, const unsigned int &n_stra,
                          const mat &Z_tv, const mat &B_spline, const mat &theta_init,
                          const mat &Z_ti, const vec &beta_ti_init,
                          const string &method="Newton",
                          const double lambda=1e8, const double &factor=1.0,
                          const bool &parallel=false, const unsigned int &threads=1,
                          const double &tol=1e-10, const unsigned int &iter_max=20,
                          const double &s=1e-2, const double &t=0.6,
                          const string &btr="dynamic",
                          const string &stop="incre") {
  
  bool ti = norm(Z_ti, "inf") > sqrt(datum::eps);
  // iterative algorithm with backtracking line search
  unsigned int iter = 0, btr_max = 1000, btr_ct = 0;
  mat theta = theta_init; vec beta_ti = beta_ti_init;
  double crit = 1.0, v = 1.0, logplkd_init = 0, logplkd, diff_logplkd, 
    inc, rhs_btr = 0;
  List update_list;
  auto tuple_idx = ranstra(event, n_stra);
  vector<uvec> idx_stra, idx_fail;
  tie(idx_stra, idx_fail) = tuple_idx;
  logplkd = objfun_ranstra(Z_tv, B_spline, theta, Z_ti, beta_ti, ti, n_stra,
                           idx_stra, idx_fail, parallel, threads);
  while (iter < iter_max && crit > tol) {
    ++iter;
    auto tuple_idx = ranstra(event, n_stra);
    vector<uvec> idx_stra, idx_fail;
    tie(idx_stra, idx_fail) = tuple_idx;
    update_list = stepinc_ranstra(event, Z_tv, B_spline, theta, Z_ti, beta_ti, 
                                  ti, n_stra, idx_stra, idx_fail, method, 
                                  lambda*pow(factor,iter-1), parallel, threads);
    v = 1.0; // reset step size
    if (ti) {
      mat step_tv = update_list["step_tv"];
      vec step_ti = update_list["step_ti"];
      inc = update_list["inc"];
      if (btr=="none") {
        theta += step_tv;
        beta_ti += step_ti;
        crit = inc / 2;
        logplkd = objfun_ranstra(Z_tv, B_spline, theta, Z_ti, beta_ti, ti, 
                                 n_stra, idx_stra, idx_fail, parallel, threads);
      } else {
        mat theta_tmp = theta + step_tv;
        vec beta_ti_tmp = beta_ti + step_ti;
        diff_logplkd = objfun_ranstra(Z_tv, B_spline, theta_tmp, Z_ti, 
                                      beta_ti_tmp, ti, n_stra, idx_stra, 
                                      idx_fail, parallel, threads) - logplkd;
        if (btr=="dynamic")      rhs_btr = inc;
        else if (btr=="static")  rhs_btr = 1.0;
        while (diff_logplkd < s * v * rhs_btr && btr_ct < btr_max) {
          ++btr_ct;
          v *= t;
          theta_tmp = theta + v * step_tv;
          beta_ti_tmp = beta_ti + step_ti;
          diff_logplkd = objfun_ranstra(Z_tv, B_spline, theta_tmp, Z_ti, 
                                        beta_ti_tmp, ti, n_stra, idx_stra,
                                        idx_fail, parallel, threads) - logplkd;
        }
        theta = theta_tmp; beta_ti = beta_ti_tmp;
        if (iter==1) logplkd_init = logplkd;
        if (stop=="incre")
          crit = inc / 2;
        else if (stop=="relch")
          crit = abs(diff_logplkd/(diff_logplkd+logplkd));
        else if (stop=="ratch")
          crit = abs(diff_logplkd/(diff_logplkd+logplkd-logplkd_init));
        logplkd += diff_logplkd;
      }
    } else {
      mat step = update_list["step"];
      inc = update_list["inc"];
      if (btr=="none") {
        theta += step;
        crit = inc / 2;
        logplkd = objfun_ranstra(Z_tv, B_spline, theta, Z_ti, beta_ti, ti,
                                 n_stra, idx_stra, idx_fail, parallel, threads);
      } else {
        mat theta_tmp = theta + step;
        diff_logplkd = objfun_ranstra(Z_tv, B_spline, theta_tmp, Z_ti, beta_ti, 
                                      ti, n_stra, idx_stra, idx_fail, parallel,
                                      threads) - logplkd;
        if (btr=="dynamic")      rhs_btr = inc;
        else if (btr=="static")  rhs_btr = 1.0;
        unsigned int btr_ct_iter = 0, btr_max_iter = 50;
        while (diff_logplkd < s * v * rhs_btr && btr_ct < btr_max && 
               btr_ct_iter < btr_max_iter) {
          ++btr_ct;
          ++btr_ct_iter;
          v *= t;
          theta_tmp = theta + v * step;
          diff_logplkd = objfun_ranstra(Z_tv, B_spline, theta_tmp, Z_ti, 
                                        beta_ti, ti, n_stra, idx_stra, idx_fail,
                                        parallel, threads) - logplkd;
        }
        theta = theta_tmp;
        if (iter==1) logplkd_init = logplkd;
        if (stop=="incre")
          crit = inc / 2;
        else if (stop=="relch")
          crit = abs(diff_logplkd/(diff_logplkd+logplkd));
        else if (stop=="ratch")
          crit = abs(diff_logplkd/(diff_logplkd+logplkd-logplkd_init));
        logplkd += diff_logplkd;
      }
    }
    cout << "Iter " << iter << ": Obj fun = " << setprecision(7) << fixed <<
      logplkd << "; Stopping crit = " <<
      setprecision(7) << scientific << crit << ";" << endl;
  }
  cout << "Algorithm converged after " << iter << " iterations!" << endl;
  if (ti) {
    return List::create(_["ctrl.pts"]=theta, _["tief"]=beta_ti,
                        _["info"]=update_list["info"], 
                        _["logplkd"]=logplkd*event.n_elem,
                        _["btr.ct"]=btr_ct);
  } else {
    return List::create(_["ctrl.pts"]=theta, _["info"]=update_list["info"],
                        _["logplkd"]=logplkd*event.n_elem, _["btr.ct"]=btr_ct);
  }
}