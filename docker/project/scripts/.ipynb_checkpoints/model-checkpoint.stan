data {
  int N;
  int<lower=0, upper=1>  Curr[N];
  int<lower=0, upper=1>  Prev[N];
  int<lower=0, upper=1> Y[N];
}

parameters {
  real c[2];
  real b[3];
}

transformed parameters {
  real q_C[N];
  real q_Y[N];
  for (n in 1:N){
    q_C[n] = inv_logit(c[1] + c[2]*Prev[n]);
    q_Y[n] = inv_logit(b[1] + b[2]*Prev[n] + b[3]*Curr[n]);
  }
}

model {
  for (n in 1:N) {
    Curr[n] ~ bernoulli(q_C[n]);
    Y[n] ~ bernoulli(q_Y[n]);
  }
}