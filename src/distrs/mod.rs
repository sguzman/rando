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
    
    /// Sample a single value from the distribution.
    fn sample(&self) -> f64 {
        self.inv_cdf(rand::random::<f64>())
    }

    /// Sample multiple values from the distribution.
    fn sample_many(&self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.sample()).collect()
    }

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

    /// Survival function S(x) = 1 - CDF(x).
    fn survival_function(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }

    /// Hazard function H(x) = PDF(x) / S(x).
    fn hazard_function(&self, x: f64) -> f64 {
        let s = self.survival_function(x);
        if s <= 0.0 { return f64::INFINITY; }
        self.pdf(x) / s
    }

    /// n-th raw moment E[X^n]. Placeholder for numerical integration.
    fn moment(&self, _n: u32) -> f64 {
        0.0
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
pub mod poisson;
pub mod gamma;
pub mod negative_binomial;
pub mod exponential;
pub mod weibull;
pub mod log_normal;
pub mod chi_square;
pub mod empirical;
pub mod mixture;
pub mod smooth_kernel;

pub use normal::*;
pub use student_t::*;
pub use cauchy::*;
pub use laplace::*;
pub use logistic::*;
pub use poisson::*;
pub use gamma::*;
pub use negative_binomial::*;
pub use exponential::*;
pub use weibull::*;
pub use log_normal::*;
pub use chi_square::*;
pub use empirical::*;
pub use mixture::*;
pub use smooth_kernel::*;

/// A container for any fitted distribution
pub enum FittedDistributionBox {
    Normal(FittedNormal),
    StudentT(FittedStudentT),
    Cauchy(FittedCauchy),
    Laplace(FittedLaplace),
    Logistic(FittedLogistic),
    Poisson(FittedPoisson),
    Gamma(FittedGamma),
    NegativeBinomial(FittedNegativeBinomial),
    Exponential(FittedExponential),
    Weibull(FittedWeibull),
    LogNormal(FittedLogNormal),
    ChiSquare(FittedChiSquare),
    Empirical(FittedEmpirical),
    Mixture(FittedMixture),
    KDE(FittedKDE),
}

impl FittedDistribution for FittedDistributionBox {
    fn name(&self) -> &'static str {
        match self {
            Self::Normal(d) => d.name(),
            Self::StudentT(d) => d.name(),
            Self::Cauchy(d) => d.name(),
            Self::Laplace(d) => d.name(),
            Self::Logistic(d) => d.name(),
            Self::Poisson(d) => d.name(),
            Self::Gamma(d) => d.name(),
            Self::NegativeBinomial(d) => d.name(),
            Self::Exponential(d) => d.name(),
            Self::Weibull(d) => d.name(),
            Self::LogNormal(d) => d.name(),
            Self::ChiSquare(d) => d.name(),
            Self::Empirical(d) => d.name(),
            Self::Mixture(d) => d.name(),
            Self::KDE(d) => d.name(),
        }
    }
    fn params(&self) -> Vec<f64> {
        match self {
            Self::Normal(d) => d.params(),
            Self::StudentT(d) => d.params(),
            Self::Cauchy(d) => d.params(),
            Self::Laplace(d) => d.params(),
            Self::Logistic(d) => d.params(),
            Self::Poisson(d) => d.params(),
            Self::Gamma(d) => d.params(),
            Self::NegativeBinomial(d) => d.params(),
            Self::Exponential(d) => d.params(),
            Self::Weibull(d) => d.params(),
            Self::LogNormal(d) => d.params(),
            Self::ChiSquare(d) => d.params(),
            Self::Empirical(d) => d.params(),
            Self::Mixture(d) => d.params(),
            Self::KDE(d) => d.params(),
        }
    }
    fn pdf(&self, x: f64) -> f64 {
        match self {
            Self::Normal(d) => d.pdf(x),
            Self::StudentT(d) => d.pdf(x),
            Self::Cauchy(d) => d.pdf(x),
            Self::Laplace(d) => d.pdf(x),
            Self::Logistic(d) => d.pdf(x),
            Self::Poisson(d) => d.pdf(x),
            Self::Gamma(d) => d.pdf(x),
            Self::NegativeBinomial(d) => d.pdf(x),
            Self::Exponential(d) => d.pdf(x),
            Self::Weibull(d) => d.pdf(x),
            Self::LogNormal(d) => d.pdf(x),
            Self::ChiSquare(d) => d.pdf(x),
            Self::Empirical(d) => d.pdf(x),
            Self::Mixture(d) => d.pdf(x),
            Self::KDE(d) => d.pdf(x),
        }
    }
    fn cdf(&self, x: f64) -> f64 {
        match self {
            Self::Normal(d) => d.cdf(x),
            Self::StudentT(d) => d.cdf(x),
            Self::Cauchy(d) => d.cdf(x),
            Self::Laplace(d) => d.cdf(x),
            Self::Logistic(d) => d.cdf(x),
            Self::Poisson(d) => d.cdf(x),
            Self::Gamma(d) => d.cdf(x),
            Self::NegativeBinomial(d) => d.cdf(x),
            Self::Exponential(d) => d.cdf(x),
            Self::Weibull(d) => d.cdf(x),
            Self::LogNormal(d) => d.cdf(x),
            Self::ChiSquare(d) => d.cdf(x),
            Self::Empirical(d) => d.cdf(x),
            Self::Mixture(d) => d.cdf(x),
            Self::KDE(d) => d.cdf(x),
        }
    }
    fn inv_cdf(&self, p: f64) -> f64 {
        match self {
            Self::Normal(d) => d.inv_cdf(p),
            Self::StudentT(d) => d.inv_cdf(p),
            Self::Cauchy(d) => d.inv_cdf(p),
            Self::Laplace(d) => d.inv_cdf(p),
            Self::Logistic(d) => d.inv_cdf(p),
            Self::Poisson(d) => d.inv_cdf(p),
            Self::Gamma(d) => d.inv_cdf(p),
            Self::NegativeBinomial(d) => d.inv_cdf(p),
            Self::Exponential(d) => d.inv_cdf(p),
            Self::Weibull(d) => d.inv_cdf(p),
            Self::LogNormal(d) => d.inv_cdf(p),
            Self::ChiSquare(d) => d.inv_cdf(p),
            Self::Empirical(d) => d.inv_cdf(p),
            Self::Mixture(d) => d.inv_cdf(p),
            Self::KDE(d) => d.inv_cdf(p),
        }
    }
}

/// Finds the distribution that best fits the data based on AIC.
pub fn find_distribution(data: &[f64]) -> Result<FittedDistributionBox> {
    let mut best_aic = f64::INFINITY;
    let mut best_fit: Option<FittedDistributionBox> = None;

    macro_rules! try_fit {
        ($fit_type:ident, $variant:ident) => {
            if let Ok(fit) = $fit_type::fit(data) {
                let aic = fit.aic(data);
                if aic < best_aic {
                    best_aic = aic;
                    best_fit = Some(FittedDistributionBox::$variant(fit));
                }
            }
        };
    }

    // Try discrete distributions if data is non-negative integers
    let is_discrete = data.iter().all(|&x| x >= 0.0 && x.floor() == x);
    if is_discrete {
        try_fit!(PoissonFit, Poisson);
        try_fit!(NegativeBinomialFit, NegativeBinomial);
        
        // If we found a good discrete fit, we might want to return it immediately
        // or at least prioritize it. For now, let's just return the best discrete fit if one exists.
        if let Some(fit) = best_fit {
            return Ok(fit);
        }
    }

    // Try continuous distributions
    try_fit!(NormalFit, Normal);
    try_fit!(StudentTFit, StudentT);
    try_fit!(CauchyFit, Cauchy);
    try_fit!(LaplaceFit, Laplace);
    try_fit!(LogisticFit, Logistic);
    try_fit!(GammaFit, Gamma);
    try_fit!(ExponentialFit, Exponential);
    try_fit!(WeibullFit, Weibull);
    try_fit!(LogNormalFit, LogNormal);
    try_fit!(ChiSquareFit, ChiSquare);
    
    let _ = best_aic;

    best_fit.ok_or_else(|| anyhow::anyhow!("Could not fit any distribution to the data"))
}
