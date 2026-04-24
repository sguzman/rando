use crate::distrs::{FittedDistribution, DistributionFit};
use anyhow::Result;
use statrs::distribution::{LogNormal, Continuous, ContinuousCDF};

pub struct FittedLogNormal {
    pub mu: f64,
    pub sigma: f64,
}

impl FittedDistribution for FittedLogNormal {
    fn name(&self) -> &'static str { "LogNormalDistribution" }
    fn params(&self) -> Vec<f64> { vec![self.mu, self.sigma] }
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 { return 0.0; }
        let d = LogNormal::new(self.mu, self.sigma).unwrap();
        d.pdf(x)
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 { return 0.0; }
        let d = LogNormal::new(self.mu, self.sigma).unwrap();
        d.cdf(x)
    }
    fn inv_cdf(&self, p: f64) -> f64 {
        let d = LogNormal::new(self.mu, self.sigma).unwrap();
        d.inverse_cdf(p)
    }
}

pub struct LogNormalFit;

impl DistributionFit for LogNormalFit {
    type Fitted = FittedLogNormal;

    fn fit(data: &[f64]) -> Result<Self::Fitted> {
        if data.is_empty() { return Err(anyhow::anyhow!("Empty data")); }
        let log_data: Vec<f64> = data.iter().filter(|&&x| x > 0.0).map(|&x| x.ln()).collect();
        if log_data.len() < data.len() {
            return Err(anyhow::anyhow!("Data must be positive for Log-Normal fit"));
        }
        
        let mu = crate::stats::mean(&log_data);
        let n = log_data.len() as f64;
        let var_mle = log_data.iter().map(|&x| (x - mu).powi(2)).sum::<f64>() / n;
        let sigma = var_mle.sqrt();
        
        Ok(FittedLogNormal { mu, sigma })
    }
}
