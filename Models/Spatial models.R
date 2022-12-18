## 1. Baseline: no spatial effect
AS_baseline <- "data {
  int<lower=1> N; // Number of data rows
  int<lower=1> M; // Number of features
  matrix[N, M] X; //parameters 
  int<lower =0, upper = 1> y[N];
}

transformed data {
  real m0 = 10;           // Expected number of large slopes
  real slab_scale = 3;    // Scale for large slopes
  real slab_scale2 = square(slab_scale);
  real slab_df = 20;      // Effective degrees of freedom for large slopes
  real half_slab_df = 0.5 * slab_df;
}

parameters {
  vector[M] beta_tilde;
  vector<lower=0>[M] lambda;
  real<lower=0> c2_tilde;
  real tau_tilde;
  real alpha;
}

transformed parameters {
  vector[M] beta;
  real tau0 = (m0 / (M - m0)) * (2 / sqrt(1.0 * N));
  real tau = tau0 * tau_tilde;

  real c2 = slab_scale2 * c2_tilde;

  vector[M] lambda_tilde;
  lambda_tilde = sqrt( c2 * square(lambda) ./ (c2 + square(tau) * square(lambda)) );
  
  beta = tau * lambda_tilde .* beta_tilde;
}

model {
  beta_tilde ~ normal(0, 1);
  lambda ~ cauchy(0, 1);
  tau_tilde ~ cauchy(0, 1);
  c2_tilde ~ inv_gamma(half_slab_df, half_slab_df);
  alpha ~ normal(0, 2);
  
  for (n in 1:N) {
  y[n] ~ bernoulli_logit(X[n,] * beta + alpha);
  } }
"

## 2. Random spatial effect 
AS_RHorseshoe <- "data {
  int<lower=1> N; // Number of data rows
  int<lower=1> M; // Number of features
  matrix[N, M] X; //parameters 
  int<lower=1> N_area;
  int S[N];   //area subject i belongs to
  int<lower =0, upper = 1> y[N];
}

transformed data {
  real m0 = 10;           // Expected number of large slopes
  real slab_scale = 3;    // Scale for large slopes
  real slab_scale2 = square(slab_scale);
  real slab_df = 20;      // Effective degrees of freedom for large slopes
  real half_slab_df = 0.5 * slab_df;
}

parameters {
  vector[M] beta_tilde;
  vector<lower=0>[M] lambda;
  real<lower=0> c2_tilde;
  real tau_tilde;
  real alpha;
  vector[N_area] phi;
}

transformed parameters {
  vector[M] beta;
  real tau0 = (m0 / (M - m0)) * (2 / sqrt(1.0 * N));
  real tau = tau0 * tau_tilde;

  real c2 = slab_scale2 * c2_tilde;

  vector[M] lambda_tilde;
  lambda_tilde = sqrt( c2 * square(lambda) ./ (c2 + square(tau) * square(lambda)) );
  
  beta = tau * lambda_tilde .* beta_tilde;
}

model {
  beta_tilde ~ normal(0, 1);
  lambda ~ cauchy(0, 1);
  tau_tilde ~ cauchy(0, 1);
  c2_tilde ~ inv_gamma(half_slab_df, half_slab_df);
  alpha ~ normal(0, 2);
  phi ~ normal(0, 2);
  
  for (n in 1:N) {
  y[n] ~ bernoulli_logit(X[n,] * beta + alpha + phi[S[n]]);
  } }

generated quantities {
  vector[N] yhat_val;
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = bernoulli_logit_lpmf(y[n] | X[n,] * beta + alpha + phi[S[n]]);
    yhat_val[n] = inv_logit(X[n,] * beta + alpha + phi[S[n]]);
  }}"

## 3. With autoregressive data structure (Regularized Horseshoe prior selected)
AS_spatial <- "data {
  int<lower=1> N; // Number of observations in the data
  int<lower=1> M; // Number of features 
  matrix[N, M] X; //the matrix of parameters values
  int<lower =0, upper = 1> y[N]; // outcome vector
  int<lower=1> N_area;  //the number of regions in Quebec
  int<lower=0> N_edges;    // number of edges between regions: if region A, B attached
  int<lower=1, upper=N> node1[N_edges];   // the adjacency indicator between node1[i] (region) and node2[i] (region)
  int<lower=1, upper=N> node2[N_edges];   // and node1[i] < node2[i]
  int S[N];   //the region patient i belongs to
}
transformed data {
  real tau0 = 0.001;     
  real slab_scale = 4;  
  real slab_scale2 = square(slab_scale);
  real slab_df = 20;     
  real half_slab_df = 0.5 * slab_df;
}
parameters {
  vector[M] beta_tilde;
  vector<lower=0>[M] lambda;
  real<lower=0> c2_tilde;
  real tau_tilde;
  
  real alpha;   //the intercept in the logistic model  
  
  real<lower=0> sigma;      // the global spatial effect
  vector[N_area] zeta;      // region-specific spatial effects, with sigma*zeta equals to regional random effects
}
transformed parameters {
  vector[M] beta;
  real tau = tau0 * tau_tilde;
  
  real c2 = slab_scale2 * c2_tilde;
  
  vector[M] lambda_tilde;
  lambda_tilde = sqrt( c2 * square(lambda) ./ (c2 + square(tau) *
                                                 square(lambda)) );
  
  beta = tau * lambda_tilde .* beta_tilde;
}
model {
  beta_tilde ~ normal(0, 1);
  lambda ~ cauchy(0, 1);
  tau_tilde ~ cauchy(0, 1);
  c2_tilde ~ inv_gamma(half_slab_df, half_slab_df);
  alpha ~ normal(0, 2);
  sigma ~ gamma(1,1);

  target += -0.5 * dot_self(zeta[node1] - zeta[node2]);
  sum(zeta) ~ normal(0, 0.001 * N);  
  
  for (n in 1:N) {
    y[n] ~ bernoulli_logit(X[n,] * beta + alpha + sigma*zeta[S[n]]);
  } 
}
generated quantities {
  vector[N] yhat_val;
  for (n in 1:N) {
    yhat_val[n] = inv_logit(X[n,] * beta + alpha + sigma*zeta[S[n]]);
  }
}"

#function to generate adjacency relationship used in the spatial model
mungeCARdata4stan = function(adjBUGS,numBUGS) {
  N = length(numBUGS);
  nn = numBUGS;
  N_edges = length(adjBUGS) / 2;
  node1 = vector(mode="numeric", length=N_edges);
  node2 = vector(mode="numeric", length=N_edges);
  iAdj = 0;
  iEdge = 0;
  for (i in 1:N) {
    for (j in 1:nn[i]) {
      iAdj = iAdj + 1;
      if (i < adjBUGS[iAdj]) {
        iEdge = iEdge + 1;
        node1[iEdge] = i;
        node2[iEdge] = adjBUGS[iAdj];
      }
    }
  }
  return (list("N"=N,"N_edges"=N_edges,"node1"=node1,"node2"=node2));
}