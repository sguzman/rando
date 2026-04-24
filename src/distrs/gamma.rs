use crate::distrs::{FittedDistribution, DistributionFit};
use anyhow::Result;
use statrs::distribution::{Gamma, Continuous, ContinuousCDF};
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;

pub struct FittedGamma {
    pub shape: f64,
    pub scale: f64,
}

impl FittedDistribution for FittedGamma {
    fn name(&self) -> &'static str { "Gamma" }
    fn params(&self) -> Vec<f64> { vec![self.shape, self.scale] }
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 { return 0.0; }
        let g = Gamma::new(self.shape, 1.0 / self.scale).unwrap();
        g.pdf(x)
    }
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 { return 0.0; }
        let g = Gamma::new(self.shape, 1.0 / self.scale).unwrap();
        g.cdf(x)
    }
    fn inv_cdf(&self, p: f64) -> f64 {
        let g = Gamma::new(self.shape, 1.0 / self.scale).unwrap();
        g.inverse_cdf(p)
    }
}

struct GammaMLECost<'a> {
    data: &'a [f64],
}

impl<'a> CostFunction for GammaMLECost<'a> {
    type Param = Vec<f64>; // [shape, scale]
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let shape = p[0];
        let scale = p[1];
        if shape <= 0.0 || scale <= 0.0 { return Ok(f64::INFINITY); }
        
        // theta = scale, so rate = 1/scale
        let g = match Gamma::new(shape, 1.0 / scale) {
            Ok(g) => g,
            Err(_) => return Ok(f64::INFINITY),
        };

        let mut log_likelihood = 0.0;
        for &x in self.data {
            log_likelihood += g.ln_pdf(x);
        }
        Ok(-log_likelihood)
    }
}

pub struct GammaFit;

impl DistributionFit for GammaFit {
    type Fitted = FittedGamma;

    fn fit(data: &[f64]) -> Result<Self::Fitted> {
        let _ = data.len(); 
        let m = crate::stats::mean(data);
        let v = crate::stats::variance(data);
        
        // Initial guess using method of moments:
        // m = k*theta, v = k*theta^2 => theta = v/m, k = m^2/v
        let initial_scale = v / m;
        let initial_shape = m * m / v;
        
        let cost = GammaMLECost { data };
        let solver = NelderMead::new(vec![
            vec![initial_shape, initial_scale],
            vec![initial_shape * 1.1, initial_scale],
            vec![initial_shape, initial_scale * 1.1],
        ]);

        let res = Executor::new(cost, solver)
            .configure(|state| state.max_iters(1000).target_cost(0.0))
            .run()?;

        let best_params = res.state().get_best_param().unwrap();
        Ok(FittedGamma { shape: best_params[0], scale: best_params[1] })
    }
}
