## 1. baseline model with simple random effects of region
AS_random <- "data {
  int<lower=1> N; // Number of data rows
  int<lower=1> M; // Number of features
  matrix[N, M] X; //parameters 
  int<lower=1> N_area;
  int S[N];   //area subject i belongs to
  int<lower =0, upper = 1> y[N];
}


parameters {
  vector[M] beta;
  vector[N_area] phi;
  real alpha; 
}

model {

  phi ~ normal(0, 2);
  beta ~ normal(0, 2);
  alpha ~ normal(0, 2);
  
  for (n in 1:N) {
  y[n] ~ bernoulli_logit(X[n,] * beta + alpha + phi[S[n]]);
  } }

generated quantities {
  vector[N] yhat_val;
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = bernoulli_logit_lpmf(y[n] | X[n,] * beta + alpha + phi[S[n]]);
    yhat_val[n] = inv_logit(X[n,] * beta + alpha + phi[S[n]]);
  }}
"

## 2. with random effects and 3 different shrinkage priors  
### (1) Laplace prior  
AS_Laplace <- "data {
  int<lower=1> N; // Number of data rows
  int<lower=1> M; // Number of features
  matrix[N, M] X; //parameters 
  int<lower=1> N_area;
  int S[N];   //area subject i belongs to
  int<lower =0, upper = 1> y[N];
}

parameters {
  
  vector[M] beta;
  vector[N_area] phi;
  real alpha;
}

model {

  beta ~ double_exponential(0, 1);
  alpha ~ normal(0, 2);
  phi ~ normal(0, 2); 

  for (n in 1:N) {
  y[n] ~ bernoulli_logit(X[n,] * beta + alpha + phi[S[n]]);
  } 
}

generated quantities {
  vector[N] yhat_val;
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = bernoulli_logit_lpmf(y[n] | X[n,] * beta + alpha + phi[S[n]]);
    yhat_val[n] = inv_logit(X[n,] * beta + alpha + phi[S[n]]);
  }}"

### (2). Horseshoe prior  
AS_Horseshoe <- "data {
  int<lower=1> N; // Number of data rows
  int<lower=1> M; // Number of features
  matrix[N, M] X; //parameters 
  int<lower=1> N_area;
  int S[N];   //area subject i belongs to
  int<lower =0, upper = 1> y[N];
}

parameters {
  
  vector[M] beta_tilde;
  vector<lower=0>[M] lambda;
  real<lower=0> tau_tilde;
  vector[N_area] phi;
  real alpha;
}

transformed parameters {

  vector[M] beta;
  
  beta = beta_tilde .* lambda * .25 * tau_tilde;
}
model {

  beta_tilde ~ normal(0, 1);
  lambda ~ cauchy(0, 1);
  tau_tilde ~ cauchy(0, 1);
  alpha ~ normal(0, 2);
  phi ~ normal(0, 2); 

  for (n in 1:N) {
  y[n] ~ bernoulli_logit(X[n,] * beta + alpha + phi[S[n]]);
  } 
}

generated quantities {
  vector[N] yhat_val;
  vector[N] log_lik;
  for (n in 1:N) {
    log_lik[n] = bernoulli_logit_lpmf(y[n] | X[n,] * beta + alpha + phi[S[n]]);
    yhat_val[n] = inv_logit(X[n,] * beta + alpha + phi[S[n]]);
  }}"

### (3). Regularized Horseshoe Prior  
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

## Compile Rstan models
library(rstan)
library(dplyr)

random_model <- stan_model(model_code = AS_random)
laplace_model <- stan_model(model_code = AS_Laplace)
horseshoe_model <- stan_model(model_code = AS_Horseshoe)
rhorseshoe_model <- stan_model(model_code = AS_RHorseshoe)

## Simulate Data 
###generate the predictors: normally distributed continuous predictors (like age), and bernoulli distributed predictors (like sex)
X <- matrix(nrow=100,ncol=50)
for(j in 1:50){
  temp <- sample(c(1,2,3),1);
  if(temp == 1){
    X[,j] = rnorm(100);
  }
  if(temp == 2){
    X[,j] = rnorm(100,mean=0,sd=3);
  }
  if(temp == 3){
    X[,j] = rbinom(100,1,0.5);
  }
}

area <- round(runif(100, min = 0.5, max = 5.5))
spatial_effect <- c()
for(i in 1:100){
  if(area[i]==1){
    spatial_effect[i] = 1
  }
  if(area[i]==2){
    spatial_effect[i] = 0.5
  }
  if(area[i]==3){
    spatial_effect[i] = -0.5
  }
  if(area[i]==4){
    spatial_effect[i] = 0
  }
  if(area[i]==5){
    spatial_effect[i] = -1
  }
}

beta = c(runif(3,min=-2,max=(-0.8)), runif(3,max=2,min=0.8), rnorm(44,mean=0,sd=0.1))
y = rbinom(100,1,exp(X%*%beta+spatial_effect)/(1+exp(X%*%beta+spatial_effect)))

## Run models and record evaluation matrics
data_random = list(N=100,M=50,X=X,y=y,N_area=5,S=area)

fit_random <- sampling(
  random_model
  , data = data_random
  , iter = 3000
  , cores = 4
  , chains = 4
  , verbose = F,control = list(adapt_delta = 0.95))

###Convergence check: Rhat
sum_beta <- summary(fit_random,pars = c("beta"))$summary
rhat_beta <- sum_beta[,10]
rhat_beta

### Predictive Performance
library(pROC)
# 1. AUC 
ext_fit <- extract(fit_spatial)
treatment_hat = apply(ext_fit$yhat_val,2,mean)
ds1<-data.frame(cbind(treatment,treatment_hat))
colnames(ds1) <- c("true","pred")
dsroc1<-roc(data=ds1, response = true,predictor = pred)
auc(dsroc1)

# 2. Brier Score 
mean((treatment-treatment_hat)^2) 

# 3. Goodness of fit (WAIC)
library(loo)
log_lik1 <- extract_log_lik(fit_random)
waic1 <- waic(log_lik1)

## Repeat the model evaluation process and compare AUC, Brier Score and WAIC for three shrinkage priors
## Repeat the whole process for 100 times to reduce randomness