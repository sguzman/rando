use crate::distrs::{FittedDistribution, DistributionFit};
use anyhow::Result;
use statrs::distribution::{ChiSquared, Continuous, ContinuousCDF};

pub struct FittedChiSquare {
    pub dof: f64,
}

impl FittedDistribution for FittedChiSquare {
    fn name(&self) -> &'static str { "ChiSquareDistribution" }
    fn params(&self) -> Vec<f64> { vec![self.dof] }
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 { return 0.0; }
        let d = ChiSquared::new(self.dof).unwrap();
        d.pdf(x)
    }
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 { return 0.0; }
        let d = ChiSquared::new(self.dof).unwrap();
        d.cdf(x)
    }
    fn inv_cdf(&self, p: f64) -> f64 {
        let d = ChiSquared::new(self.dof).unwrap();
        d.inverse_cdf(p)
    }
}

pub struct ChiSquareFit;

impl DistributionFit for ChiSquareFit {
    type Fitted = FittedChiSquare;

    fn fit(data: &[f64]) -> Result<Self::Fitted> {
        if data.is_empty() { return Err(anyhow::anyhow!("Empty data")); }
        let m = crate::stats::mean(data);
        if m <= 0.0 { return Err(anyhow::anyhow!("Data mean must be positive for Chi-Square fit")); }
        // For Chi-Square, Mean = dof
        let dof = m;
        Ok(FittedChiSquare { dof })
    }
}
