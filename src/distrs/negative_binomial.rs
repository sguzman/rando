use crate::distrs::{FittedDistribution, DistributionFit};
use anyhow::Result;
use statrs::distribution::{NegativeBinomial, Discrete, DiscreteCDF};
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;

pub struct FittedNegativeBinomial {
    pub r: f64,
    pub p: f64,
}

impl FittedDistribution for FittedNegativeBinomial {
    fn name(&self) -> &'static str { "NegativeBinomial" }
    fn params(&self) -> Vec<f64> { vec![self.r, self.p] }
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 || x != x.floor() { return 0.0; }
        let d = NegativeBinomial::new(self.r, self.p).unwrap();
        d.pmf(x as u64)
    }
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 { return 0.0; }
        let d = NegativeBinomial::new(self.r, self.p).unwrap();
        d.cdf(x as u64)
    }
    fn inv_cdf(&self, p: f64) -> f64 {
        // Simple search for inverse CDF on discrete distribution
        let d = NegativeBinomial::new(self.r, self.p).unwrap();
        let mut prob = 0.0;
        let mut k = 0;
        while prob < p && k < 10000 {
            prob = d.cdf(k);
            if prob >= p { break; }
            k += 1;
        }
        k as f64
    }
}

struct NegativeBinomialMLECost<'a> {
    data: &'a [f64],
}

impl<'a> CostFunction for NegativeBinomialMLECost<'a> {
    type Param = Vec<f64>; // [r, p]
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, Error> {
        let r = params[0];
        let p = params[1];
        if r <= 0.0 || p <= 0.0 || p >= 1.0 { return Ok(f64::INFINITY); }
        
        let d = NegativeBinomial::new(r, p).unwrap();
        let mut log_likelihood = 0.0;
        for &x in self.data {
            log_likelihood += d.ln_pmf(x as u64);
        }
        Ok(-log_likelihood)
    }
}

pub struct NegativeBinomialFit;

impl DistributionFit for NegativeBinomialFit {
    type Fitted = FittedNegativeBinomial;

    fn fit(data: &[f64]) -> Result<Self::Fitted> {
        let mean = crate::stats::mean(data);
        let var = crate::stats::variance(data);
        
        // Initial guess via Method of Moments
        // mean = r(1-p)/p
        // var = r(1-p)/p^2
        // => p = mean / var
        // => r = mean * p / (1-p)
        let mut p_init = mean / var;
        if p_init >= 1.0 || p_init <= 0.0 { p_init = 0.5; }
        let mut r_init = (mean * p_init) / (1.0 - p_init);
        if r_init <= 0.0 { r_init = 1.0; }

        let cost = NegativeBinomialMLECost { data };
        let solver = NelderMead::new(vec![
            vec![r_init, p_init],
            vec![r_init * 1.1, p_init],
            vec![r_init, p_init * 0.9],
        ]);

        let res = Executor::new(cost, solver)
            .configure(|state| state.max_iters(1000))
            .run()?;

        let best = res.state().get_best_param().unwrap();
        Ok(FittedNegativeBinomial { r: best[0], p: best[1] })
    }
}
