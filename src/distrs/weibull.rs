use crate::distrs::{FittedDistribution, DistributionFit};
use anyhow::Result;
use statrs::distribution::{Weibull, Continuous, ContinuousCDF};
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;

pub struct FittedWeibull {
    pub shape: f64,
    pub scale: f64,
}

impl FittedDistribution for FittedWeibull {
    fn name(&self) -> &'static str { "WeibullDistribution" }
    fn params(&self) -> Vec<f64> { vec![self.shape, self.scale] }
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 { return 0.0; }
        let d = Weibull::new(self.shape, self.scale).unwrap();
        d.pdf(x)
    }
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 { return 0.0; }
        let d = Weibull::new(self.shape, self.scale).unwrap();
        d.cdf(x)
    }
    fn inv_cdf(&self, p: f64) -> f64 {
        let d = Weibull::new(self.shape, self.scale).unwrap();
        d.inverse_cdf(p)
    }
}

struct WeibullMLECost<'a> {
    data: &'a [f64],
}

impl<'a> CostFunction for WeibullMLECost<'a> {
    type Param = Vec<f64>; // [shape, scale]
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let shape = p[0];
        let scale = p[1];
        if shape <= 0.0 || scale <= 0.0 { return Ok(f64::INFINITY); }
        
        let d = match Weibull::new(shape, scale) {
            Ok(d) => d,
            Err(_) => return Ok(f64::INFINITY),
        };

        let mut ll = 0.0;
        for &x in self.data {
            if x < 0.0 { return Ok(f64::INFINITY); }
            ll += d.ln_pdf(x);
        }
        Ok(-ll)
    }
}

pub struct WeibullFit;

impl DistributionFit for WeibullFit {
    type Fitted = FittedWeibull;

    fn fit(data: &[f64]) -> Result<Self::Fitted> {
        if data.is_empty() { return Err(anyhow::anyhow!("Empty data")); }
        
        // Initial guess using method of moments or simple heuristics
        let m = crate::stats::mean(data);
        let initial_scale = m;
        let initial_shape = 1.2; // Common starting point
        
        let cost = WeibullMLECost { data };
        let solver = NelderMead::new(vec![
            vec![initial_shape, initial_scale],
            vec![initial_shape * 1.1, initial_scale],
            vec![initial_shape, initial_scale * 1.1],
        ]);

        let res = Executor::new(cost, solver)
            .configure(|state| state.max_iters(500))
            .run()?;

        let best = res.state().get_best_param().unwrap();
        Ok(FittedWeibull { shape: best[0], scale: best[1] })
    }
}
