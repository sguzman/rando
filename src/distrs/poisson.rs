use super::{FittedDistribution, DistributionFit};
use anyhow::Result;
use statrs::distribution::{Poisson, Discrete, DiscreteCDF};

pub struct FittedPoisson {
    pub lambda: f64,
    inner: Poisson,
}

impl FittedDistribution for FittedPoisson {
    fn name(&self) -> &'static str {
        "PoissonDistribution"
    }

    fn params(&self) -> Vec<f64> {
        vec![self.lambda]
    }

    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 || x.fract() != 0.0 { return 0.0; }
        self.inner.pmf(x as u64)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 { return 0.0; }
        self.inner.cdf(x.floor() as u64)
    }

    fn inv_cdf(&self, p: f64) -> f64 {
        // Simple search for discrete distribution
        let mut x = 0;
        while self.inner.cdf(x as u64) < p {
            x += 1;
            if x > 10000 { break; } // Safety
        }
        x as f64
    }
}

pub struct PoissonFit;

impl DistributionFit for PoissonFit {
    type Fitted = FittedPoisson;

    fn fit(data: &[f64]) -> Result<Self::Fitted> {
        let n = data.len() as f64;
        let lambda = data.iter().sum::<f64>() / n;
        Ok(FittedPoisson {
            lambda,
            inner: Poisson::new(lambda)?,
        })
    }
}
