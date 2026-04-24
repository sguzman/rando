use super::{FittedDistribution, DistributionFit};
use anyhow::Result;
use std::f64::consts::PI;
use crate::stats::{mean, std_dev};
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;

pub struct FittedLogistic {
    pub mu: f64,
    pub s: f64,
}

impl FittedDistribution for FittedLogistic {
    fn name(&self) -> &'static str {
        "LogisticDistribution"
    }

    fn params(&self) -> Vec<f64> {
        vec![self.mu, self.s]
    }

    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.s;
        let ez = (-z).exp();
        ez / (self.s * (1.0 + ez).powi(2))
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.s;
        -self.s.ln() - z - 2.0 * (1.0 + (-z).exp()).ln()
    }

    fn cdf(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-(x - self.mu) / self.s).exp())
    }

    fn inv_cdf(&self, p: f64) -> f64 {
        self.mu + self.s * (p / (1.0 - p)).ln()
    }
}

struct LogisticNLL<'a> {
    data: &'a [f64],
}

impl<'a> CostFunction for LogisticNLL<'a> {
    type Param = Vec<f64>; // [mu, s]
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> std::result::Result<Self::Output, Error> {
        let mu = p[0];
        let s = p[1];
        if s <= 0.0 {
            return Ok(f64::INFINITY);
        }

        let mut nll = 0.0;
        for &x in self.data {
            let z = (x - mu) / s;
            nll += s.ln() + z + 2.0 * (1.0 + (-z).exp()).ln();
        }

        Ok(nll)
    }
}

pub struct LogisticFit;

impl DistributionFit for LogisticFit {
    type Fitted = FittedLogistic;

    fn fit(data: &[f64]) -> Result<Self::Fitted> {
        if data.is_empty() {
            return Err(anyhow::anyhow!("Empty data"));
        }

        let mu_init = mean(data);
        let sigma = std_dev(data);
        let s_init = sigma * 3.0f64.sqrt() / PI;
        let s_init = if s_init <= 0.0 { 1e-6 } else { s_init };

        let cost = LogisticNLL { data };
        
        let simplex = vec![
            vec![mu_init, s_init],
            vec![mu_init + 0.1, s_init],
            vec![mu_init, s_init * 1.1],
        ];

        let solver = NelderMead::new(simplex)
            .with_sd_tolerance(1e-6)?;

        let res = Executor::new(cost, solver)
            .configure(|state| state.max_iters(100))
            .run()?;

        let best_param = res.state().get_best_param().ok_or_else(|| anyhow::anyhow!("Optimization failed"))?.clone();
        let mu = best_param[0];
        let s = best_param[1];

        Ok(FittedLogistic { mu, s })
    }
}
