use super::{FittedDistribution, DistributionFit};
use anyhow::Result;
use statrs::distribution::{StudentsT, Continuous, ContinuousCDF};
use crate::stats::{median, quantile};
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;

pub struct FittedStudentT {
    pub nu: f64,
    pub mu: f64,
    pub sigma: f64,
    inner: StudentsT,
}

impl FittedDistribution for FittedStudentT {
    fn name(&self) -> &'static str {
        "StudentTDistribution"
    }

    fn params(&self) -> Vec<f64> {
        vec![self.nu, self.mu, self.sigma]
    }

    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        self.inner.pdf(z) / self.sigma
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        self.inner.ln_pdf(z) - self.sigma.ln()
    }

    fn cdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        self.inner.cdf(z)
    }

    fn inv_cdf(&self, p: f64) -> f64 {
        self.mu + self.sigma * self.inner.inverse_cdf(p)
    }
}

struct StudentTNLL<'a> {
    data: &'a [f64],
}

impl<'a> CostFunction for StudentTNLL<'a> {
    type Param = Vec<f64>; // [nu, mu, sigma]
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> std::result::Result<Self::Output, Error> {
        let nu = p[0];
        let mu = p[1];
        let sigma = p[2];

        if nu <= 0.0 || sigma <= 0.0 {
            return Ok(f64::INFINITY);
        }

        let dist = match StudentsT::new(0.0, 1.0, nu) {
            Ok(d) => d,
            Err(_) => return Ok(f64::INFINITY),
        };

        let mut nll = 0.0;
        for &x in self.data {
            let z = (x - mu) / sigma;
            nll -= dist.ln_pdf(z) - sigma.ln();
        }

        Ok(nll)
    }
}

pub struct StudentTFit;

impl DistributionFit for StudentTFit {
    type Fitted = FittedStudentT;

    fn fit(data: &[f64]) -> Result<Self::Fitted> {
        if data.is_empty() {
            return Err(anyhow::anyhow!("Empty data"));
        }

        let mu_init = median(data);
        let q1 = quantile(data, 0.25);
        let q3 = quantile(data, 0.75);
        let sigma_init = (q3 - q1) / 2.0;
        let sigma_init = if sigma_init <= 0.0 { 1e-6 } else { sigma_init };
        let nu_init = 3.0;

        let cost = StudentTNLL { data };
        
        let simplex = vec![
            vec![nu_init, mu_init, sigma_init],
            vec![nu_init * 1.1, mu_init, sigma_init],
            vec![nu_init, mu_init + 0.1, sigma_init],
            vec![nu_init, mu_init, sigma_init * 1.1],
        ];

        let solver = NelderMead::new(simplex)
            .with_sd_tolerance(1e-6)?;

        let res = Executor::new(cost, solver)
            .configure(|state| state.max_iters(200))
            .run()?;

        let best_param = res.state().get_best_param().ok_or_else(|| anyhow::anyhow!("Optimization failed"))?.clone();
        let nu = best_param[0];
        let mu = best_param[1];
        let sigma = best_param[2];

        let inner = StudentsT::new(0.0, 1.0, nu).map_err(|e| anyhow::anyhow!(e))?;
        Ok(FittedStudentT { nu, mu, sigma, inner })
    }
}
