use crate::distrs::{FittedDistribution, DistributionFit};
use anyhow::Result;
use statrs::distribution::{Normal, Continuous};
use statrs::statistics::Distribution;

pub struct FittedMixture {
    pub components: Vec<(f64, Normal)>, // weight, distribution
}

impl FittedDistribution for FittedMixture {
    fn name(&self) -> &'static str { "MixtureDistribution" }
    fn params(&self) -> Vec<f64> {
        let mut p = Vec::new();
        for (w, d) in &self.components {
            p.push(*w);
            p.push(d.mean().unwrap());
            p.push(d.std_dev().unwrap());
        }
        p
    }
    
    fn pdf(&self, x: f64) -> f64 {
        self.components.iter().map(|(w, d)| w * d.pdf(x)).sum()
    }
    
    fn cdf(&self, x: f64) -> f64 {
        use statrs::distribution::ContinuousCDF;
        self.components.iter().map(|(w, d)| w * d.cdf(x)).sum()
    }
    
    fn inv_cdf(&self, _p: f64) -> f64 {
        // No simple analytic inverse for mixtures.
        // Could implement numerical solver (root finding on CDF - p).
        0.0 
    }
}

pub struct GaussianMixtureFit {
    pub k: usize,
}

impl DistributionFit for GaussianMixtureFit {
    type Fitted = FittedMixture;

    fn fit(data: &[f64]) -> Result<Self::Fitted> {
        if data.is_empty() { return Err(anyhow::anyhow!("Empty data")); }
        // Simple heuristic for 2 components if k=2
        // In reality, this should use EM algorithm.
        // For now, let's implement a very basic 2-component split.
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = sorted.len() / 2;
        let left = &sorted[..mid];
        let right = &sorted[mid..];
        
        let m1 = crate::stats::mean(left);
        let s1 = crate::stats::std_dev(left).max(0.001);
        let m2 = crate::stats::mean(right);
        let s2 = crate::stats::std_dev(right).max(0.001);
        
        let w1 = 0.5;
        let w2 = 0.5;
        
        Ok(FittedMixture {
            components: vec![
                (w1, Normal::new(m1, s1).unwrap()),
                (w2, Normal::new(m2, s2).unwrap()),
            ],
        })
    }
}
