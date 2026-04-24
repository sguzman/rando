use super::{FittedDistribution, DistributionFit};
use anyhow::Result;
use statrs::distribution::{Normal, Continuous, ContinuousCDF};
use crate::stats::{mean, std_dev};

pub struct FittedNormal {
    pub mu: f64,
    pub sigma: f64,
    inner: Normal,
}

impl FittedDistribution for FittedNormal {
    fn name(&self) -> &'static str {
        "NormalDistribution"
    }

    fn params(&self) -> Vec<f64> {
        vec![self.mu, self.sigma]
    }

    fn pdf(&self, x: f64) -> f64 {
        self.inner.pdf(x)
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        self.inner.ln_pdf(x)
    }

    fn cdf(&self, x: f64) -> f64 {
        self.inner.cdf(x)
    }

    fn inv_cdf(&self, p: f64) -> f64 {
        self.inner.inverse_cdf(p)
    }
}

pub struct NormalFit;

impl DistributionFit for NormalFit {
    type Fitted = FittedNormal;

    fn fit(data: &[f64]) -> Result<Self::Fitted> {
        if data.is_empty() { return Err(anyhow::anyhow!("Empty data")); }
        let mu = mean(data);
        let n = data.len() as f64;
        let var_mle = data.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / n;
        let sigma = var_mle.sqrt();
        let inner = Normal::new(mu, sigma).map_err(|e| anyhow::anyhow!(e))?;
        Ok(FittedNormal { mu, sigma, inner })
    }
}
