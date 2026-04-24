use anyhow::Result;

/// A trait for distributions that can be fitted to data.
pub trait FittedDistribution {
    /// Returns the name of the distribution family.
    fn name(&self) -> &'static str;
    
    /// Returns the fitted parameters.
    fn params(&self) -> Vec<f64>;
    
    /// Probability Density Function (PDF).
    fn pdf(&self, x: f64) -> f64;
    
    /// Log-PDF.
    fn ln_pdf(&self, x: f64) -> f64 {
        self.pdf(x).ln()
    }
    
    /// Cumulative Distribution Function (CDF).
    fn cdf(&self, x: f64) -> f64;
    
    /// Inverse CDF (Quantile function).
    fn inv_cdf(&self, p: f64) -> f64;

    /// Log-likelihood of the data given this distribution.
    fn log_likelihood(&self, xs: &[f64]) -> f64 {
        xs.iter().map(|&x| self.ln_pdf(x)).sum()
    }

    /// Akaike Information Criterion (AIC).
    fn aic(&self, xs: &[f64]) -> f64 {
        let k = self.params().len() as f64;
        2.0 * k - 2.0 * self.log_likelihood(xs)
    }

    /// Bayesian Information Criterion (BIC).
    fn bic(&self, xs: &[f64]) -> f64 {
        let k = self.params().len() as f64;
        let n = xs.len() as f64;
        k * n.ln() - 2.0 * self.log_likelihood(xs)
    }
}

/// A trait for fitting a distribution family to data.
pub trait DistributionFit {
    type Fitted: FittedDistribution;
    
    /// Fit the distribution to the given data.
    fn fit(data: &[f64]) -> Result<Self::Fitted>;
}

pub mod normal;
pub mod student_t;
pub mod cauchy;
pub mod laplace;
pub mod logistic;

pub use normal::*;
pub use student_t::*;
pub use cauchy::*;
pub use laplace::*;
pub use logistic::*;
