use crate::distrs::{FittedDistribution, DistributionFit};
use anyhow::Result;
use statrs::distribution::{Normal, Continuous, ContinuousCDF};

pub struct FittedKDE {
    pub data: Vec<f64>,
    pub bandwidth: f64,
}

impl FittedDistribution for FittedKDE {
    fn name(&self) -> &'static str { "SmoothKernelDistribution" }
    fn params(&self) -> Vec<f64> { 
        let mut p = vec![self.bandwidth];
        p.extend_from_slice(&self.data);
        p
    }
    
    fn pdf(&self, x: f64) -> f64 {
        let n = self.data.len() as f64;
        let mut sum = 0.0;
        let normal = Normal::new(0.0, 1.0).unwrap();
        for &xi in &self.data {
            sum += normal.pdf((x - xi) / self.bandwidth);
        }
        sum / (n * self.bandwidth)
    }
    
    fn cdf(&self, x: f64) -> f64 {
        let n = self.data.len() as f64;
        let mut sum = 0.0;
        let normal = Normal::new(0.0, 1.0).unwrap();
        for &xi in &self.data {
            sum += normal.cdf((x - xi) / self.bandwidth);
        }
        sum / n
    }
    
    fn inv_cdf(&self, _p: f64) -> f64 {
        // Numerical inversion required
        0.0 
    }
}

pub struct KDEFit;

impl DistributionFit for KDEFit {
    type Fitted = FittedKDE;

    fn fit(data: &[f64]) -> Result<Self::Fitted> {
        if data.is_empty() { return Err(anyhow::anyhow!("Empty data")); }
        let n = data.len() as f64;
        let sigma = crate::stats::std_dev(data);
        // Silverman's Rule of Thumb
        let bandwidth = 1.06 * sigma * n.powf(-0.2);
        let bandwidth = if bandwidth <= 0.0 { 1.0 } else { bandwidth };
        
        Ok(FittedKDE {
            data: data.to_vec(),
            bandwidth,
        })
    }
}
