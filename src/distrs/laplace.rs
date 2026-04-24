use super::{FittedDistribution, DistributionFit};
use anyhow::Result;
use statrs::distribution::{Laplace, Continuous, ContinuousCDF};
use crate::stats::median;

pub struct FittedLaplace {
    pub mu: f64,
    pub b: f64,
    inner: Laplace,
}

impl FittedDistribution for FittedLaplace {
    fn name(&self) -> &'static str {
        "LaplaceDistribution"
    }

    fn params(&self) -> Vec<f64> {
        vec![self.mu, self.b]
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

pub struct LaplaceFit;

impl DistributionFit for LaplaceFit {
    type Fitted = FittedLaplace;

    fn fit(data: &[f64]) -> Result<Self::Fitted> {
        if data.is_empty() {
            return Err(anyhow::anyhow!("Empty data"));
        }
        let mu = median(data);
        let b = data.iter().map(|&x| (x - mu).abs()).sum::<f64>() / data.len() as f64;
        let inner = Laplace::new(mu, b).map_err(|e| anyhow::anyhow!(e))?;
        Ok(FittedLaplace { mu, b, inner })
    }
}
