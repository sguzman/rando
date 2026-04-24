use crate::stats::mean;
use anyhow::Result;
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;

pub trait FittedModel {
    fn predict(&self, x: &[f64]) -> f64;
    fn coefficients(&self) -> &[f64];
    fn aic(&self) -> f64;
    fn bic(&self) -> f64;
}

pub struct LinearModelFit {
    pub coefficients: Vec<f64>, // [intercept, beta1, beta2, ...]
    pub r_squared: f64,
    pub aic: f64,
    pub bic: f64,
}

impl FittedModel for LinearModelFit {
    fn predict(&self, x: &[f64]) -> f64 {
        let mut pred = self.coefficients[0];
        for i in 0..x.len() {
            pred += self.coefficients[i + 1] * x[i];
        }
        pred
    }

    fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    fn aic(&self) -> f64 {
        self.aic
    }

    fn bic(&self) -> f64 {
        self.bic
    }
}

impl LinearModelFit {
    pub fn predict(&self, x: &[f64]) -> f64 {
        let mut pred = self.coefficients[0];
        for i in 0..x.len() {
            pred += self.coefficients[i + 1] * x[i];
        }
        pred
    }
}

struct LinearRegressionCost<'a> {
    x: &'a Vec<Vec<f64>>,
    y: &'a Vec<f64>,
}

impl<'a> CostFunction for LinearRegressionCost<'a> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let mut rss = 0.0;
        for i in 0..self.y.len() {
            let mut pred = p[0];
            for j in 0..self.x[i].len() {
                pred += p[j + 1] * self.x[i][j];
            }
            rss += (self.y[i] - pred).powi(2);
        }
        Ok(rss)
    }
}

pub fn linear_model_fit(x: &Vec<Vec<f64>>, y: &Vec<f64>) -> Result<LinearModelFit> {
    let n = y.len();
    let num_vars = x[0].len();
    let cost = LinearRegressionCost { x, y };
    
    // Initial guess: all zeros
    let mut params = vec![0.0; num_vars + 1];
    // Simple heuristic for intercept
    params[0] = mean(y);

    let solver = NelderMead::new(vec![
        params.clone(),
        {
            let mut p = params.clone();
            p[0] += 1.0;
            p
        },
        {
            let mut p = params.clone();
            if num_vars > 0 { p[1] += 1.0; }
            p
        }
    ]);

    let res = Executor::new(cost, solver)
        .configure(|state| state.max_iters(1000).target_cost(1e-10))
        .run()?;

    let best_params = res.state().get_best_param().unwrap().clone();
    
    // Calculate R-squared
    let y_mean = mean(y);
    let tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let rss = res.state().get_best_cost();
    let r_squared = 1.0 - (rss / tss);
    
    // AIC = n * ln(RSS/n) + 2k + const
    // But for simplicity and parity with basic definitions:
    // k includes intercept, slopes, and variance/dispersion
    let k = (num_vars + 2) as f64; 
    let aic = (n as f64) * ((rss / n as f64).ln() + 1.0 + (2.0 * std::f64::consts::PI).ln()) + 2.0 * k;
    let bic = (n as f64) * ((rss / n as f64).ln() + 1.0 + (2.0 * std::f64::consts::PI).ln()) + k * (n as f64).ln();

    Ok(LinearModelFit {
        coefficients: best_params,
        r_squared,
        aic,
        bic,
    })
}

pub struct LogitModelFit {
    pub coefficients: Vec<f64>,
    pub aic: f64,
    pub bic: f64,
}

impl FittedModel for LogitModelFit {
    fn predict(&self, x: &[f64]) -> f64 {
        let mut z = self.coefficients[0];
        for i in 0..x.len() {
            z += self.coefficients[i + 1] * x[i];
        }
        1.0 / (1.0 + (-z).exp())
    }

    fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    fn aic(&self) -> f64 {
        self.aic
    }

    fn bic(&self) -> f64 {
        self.bic
    }
}

struct LogitRegressionCost<'a> {
    x: &'a Vec<Vec<f64>>,
    y: &'a Vec<f64>,
}

impl<'a> CostFunction for LogitRegressionCost<'a> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let mut nll = 0.0;
        for i in 0..self.y.len() {
            let mut z = p[0];
            for j in 0..self.x[i].len() {
                z += p[j + 1] * self.x[i][j];
            }
            let prob = 1.0 / (1.0 + (-z).exp());
            let prob = prob.clamp(1e-15, 1.0 - 1e-15);
            nll -= self.y[i] * prob.ln() + (1.0 - self.y[i]) * (1.0 - prob).ln();
        }
        Ok(nll)
    }
}

pub fn logit_model_fit(x: &Vec<Vec<f64>>, y: &Vec<f64>) -> Result<LogitModelFit> {
    let n = y.len();
    let num_vars = x[0].len();
    let cost = LogitRegressionCost { x, y };
    
    let params = vec![0.0; num_vars + 1];
    let solver = NelderMead::new(vec![
        params.clone(),
        {
            let mut p = params.clone();
            p[0] += 0.1;
            p
        },
        {
            let mut p = params.clone();
            if num_vars > 0 { p[1] += 0.1; }
            p
        }
    ]);

    let res = Executor::new(cost, solver)
        .configure(|state| state.max_iters(2000).target_cost(1e-10))
        .run()?;

    let best_params = res.state().get_best_param().unwrap().clone();
    let nll = res.state().get_best_cost();
    
    // Mathematica counts intercept + variables + dispersion
    let k = (num_vars + 2) as f64;
    let aic = 2.0 * nll + 2.0 * k;
    let bic = 2.0 * nll + k * (n as f64).ln();

    Ok(LogitModelFit {
        coefficients: best_params,
        aic,
        bic,
    })
}

pub struct NonlinearModelFit {
    pub coefficients: Vec<f64>,
    pub aic: f64,
    pub bic: f64,
    pub model_fn: Box<dyn Fn(&[f64], &[f64]) -> f64 + Send + Sync>,
}

impl NonlinearModelFit {
    pub fn predict(&self, x: &[f64]) -> f64 {
        (self.model_fn)(&self.coefficients, x)
    }
}

struct NonlinearRegressionCost<'a> {
    x: &'a Vec<Vec<f64>>,
    y: &'a Vec<f64>,
    model_fn: &'a (dyn Fn(&[f64], &[f64]) -> f64 + Send + Sync),
}

impl<'a> CostFunction for NonlinearRegressionCost<'a> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let mut rss = 0.0;
        for i in 0..self.y.len() {
            let pred = (self.model_fn)(p, &self.x[i]);
            rss += (self.y[i] - pred).powi(2);
        }
        Ok(rss)
    }
}

pub fn nonlinear_model_fit<F>(
    x: &Vec<Vec<f64>>,
    y: &Vec<f64>,
    model_fn: F,
    initial_params: Vec<f64>
) -> Result<NonlinearModelFit> 
where F: Fn(&[f64], &[f64]) -> f64 + Send + Sync + 'static
{
    let n = y.len();
    let num_params = initial_params.len();
    let cost = NonlinearRegressionCost { x, y, model_fn: &model_fn };
    
    // Simplistic Nelder-Mead initialization
    let mut simplex = vec![initial_params.clone()];
    for i in 0..num_params {
        let mut p = initial_params.clone();
        if p[i] == 0.0 { p[i] = 0.1; } else { p[i] *= 1.1; }
        simplex.push(p);
    }

    let solver = NelderMead::new(simplex);

    let res = Executor::new(cost, solver)
        .configure(|state| state.max_iters(5000).target_cost(1e-10))
        .run()?;

    let best_params = res.state().get_best_param().unwrap().clone();
    let rss = res.state().get_best_cost();
    
    let k = (num_params + 1) as f64; // +1 for variance
    let aic = (n as f64) * ((rss / n as f64).ln() + 1.0 + (2.0 * std::f64::consts::PI).ln()) + 2.0 * k;
    let bic = (n as f64) * ((rss / n as f64).ln() + 1.0 + (2.0 * std::f64::consts::PI).ln()) + k * (n as f64).ln();

    Ok(NonlinearModelFit {
        coefficients: best_params,
        aic,
        bic,
        model_fn: Box::new(model_fn),
    })
}
