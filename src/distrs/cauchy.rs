use super::{FittedDistribution, DistributionFit};
use anyhow::Result;
use statrs::distribution::{Cauchy, Continuous, ContinuousCDF};
use crate::stats::{median, quantile};
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;

pub struct FittedCauchy {
    pub location: f64,
    pub scale: f64,
    inner: Cauchy,
}

impl FittedDistribution for FittedCauchy {
    fn name(&self) -> &'static str {
        "CauchyDistribution"
    }

    fn params(&self) -> Vec<f64> {
        vec![self.location, self.scale]
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

struct CauchyNLL<'a> {
    data: &'a [f64],
}

impl<'a> CostFunction for CauchyNLL<'a> {
    type Param = Vec<f64>; // [location, scale]
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> std::result::Result<Self::Output, Error> {
        let location = p[0];
        let scale = p[1];
        if scale <= 0.0 {
            return Ok(f64::INFINITY);
        }

        let n = self.data.len() as f64;
        let mut sum_ln = 0.0;
        for &x in self.data {
            let z = (x - location) / scale;
            sum_ln += (1.0 + z * z).ln();
        }

        Ok(n * scale.ln() + sum_ln)
    }
}

pub struct CauchyFit;

impl DistributionFit for CauchyFit {
    type Fitted = FittedCauchy;

    fn fit(data: &[f64]) -> Result<Self::Fitted> {
        if data.is_empty() {
            return Err(anyhow::anyhow!("Empty data"));
        }

        let loc_init = median(data);
        let q1 = quantile(data, 0.25);
        let q3 = quantile(data, 0.75);
        let scale_init = (q3 - q1) / 2.0;
        let scale_init = if scale_init <= 0.0 { 1e-6 } else { scale_init };

        let cost = CauchyNLL { data };
        
        let simplex = vec![
            vec![loc_init, scale_init],
            vec![loc_init + 0.1, scale_init],
            vec![loc_init, scale_init * 1.1],
        ];

        let solver = NelderMead::new(simplex)
            .with_sd_tolerance(1e-6)?;

        let res = Executor::new(cost, solver)
            .configure(|state| state.max_iters(100))
            .run()?;

        let best_param = res.state().get_best_param().ok_or_else(|| anyhow::anyhow!("Optimization failed"))?.clone();
        let location = best_param[0];
        let scale = best_param[1];
        let scale = if scale <= 0.0 { 1e-6 } else { scale };

        let inner = Cauchy::new(location, scale).map_err(|e| anyhow::anyhow!(e))?;
        Ok(FittedCauchy { location, scale, inner })
    }
}
