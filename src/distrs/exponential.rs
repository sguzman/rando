use crate::distrs::{FittedDistribution, DistributionFit};
use anyhow::Result;
use statrs::distribution::{Exp, Continuous, ContinuousCDF};

pub struct FittedExponential {
    pub lambda: f64,
    inner: Exp,
}

impl FittedDistribution for FittedExponential {
    fn name(&self) -> &'static str { "ExponentialDistribution" }
    fn params(&self) -> Vec<f64> { vec![self.lambda] }
    fn pdf(&self, x: f64) -> f64 { self.inner.pdf(x) }
    fn cdf(&self, x: f64) -> f64 { self.inner.cdf(x) }
    fn inv_cdf(&self, p: f64) -> f64 { self.inner.inverse_cdf(p) }
    fn ln_pdf(&self, x: f64) -> f64 { self.inner.ln_pdf(x) }
}

pub struct ExponentialFit;

impl DistributionFit for ExponentialFit {
    type Fitted = FittedExponential;

    fn fit(data: &[f64]) -> Result<Self::Fitted> {
        if data.is_empty() { return Err(anyhow::anyhow!("Empty data")); }
        let n = data.len() as f64;
        let sum: f64 = data.iter().sum();
        if sum <= 0.0 { return Err(anyhow::anyhow!("Data must be positive for Exponential fit")); }
        let lambda = n / sum;
        let inner = Exp::new(lambda).map_err(|e| anyhow::anyhow!(e))?;
        Ok(FittedExponential { lambda, inner })
    }
}
