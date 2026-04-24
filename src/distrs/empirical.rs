use crate::distrs::{FittedDistribution, DistributionFit};
use anyhow::Result;

pub struct FittedEmpirical {
    pub data: Vec<f64>,
}

impl FittedDistribution for FittedEmpirical {
    fn name(&self) -> &'static str { "EmpiricalDistribution" }
    fn params(&self) -> Vec<f64> { self.data.clone() }
    
    fn pdf(&self, _x: f64) -> f64 {
        // PDF of empirical is a sum of Dirac deltas, 
        // but often we use a binned version or 1/n at data points.
        // For AIC, we need a consistent way. Let's return a small constant or 0 if not at point.
        0.0 
    }
    
    fn cdf(&self, x: f64) -> f64 {
        let count = self.data.iter().filter(|&&v| v <= x).count();
        count as f64 / self.data.len() as f64
    }
    
    fn inv_cdf(&self, p: f64) -> f64 {
        let mut sorted = self.data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = (p * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx]
    }
}

pub struct EmpiricalFit;

impl DistributionFit for EmpiricalFit {
    type Fitted = FittedEmpirical;

    fn fit(data: &[f64]) -> Result<Self::Fitted> {
        if data.is_empty() { return Err(anyhow::anyhow!("Empty data")); }
        Ok(FittedEmpirical { data: data.to_vec() })
    }
}
