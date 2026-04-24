use crate::distrs::FittedDistribution;
use crate::hypo_tests::kolmogorov_smirnov_test;
use anyhow::Result;
use chrono::NaiveDate;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug)]
pub struct RollingResult {
    pub window_size: usize,
    pub scores: HashMap<String, Vec<(NaiveDate, f64)>>, // Family -> [(Date, PValue)]
}

pub fn get_return_series(dates: &[NaiveDate], values: &[f64]) -> (Vec<NaiveDate>, Vec<f64>) {
    let mut rets = Vec::with_capacity(values.len() - 1);
    let mut out_dates = Vec::with_capacity(dates.len() - 1);
    
    for i in 1..values.len() {
        let ret = (values[i] / values[i-1]).ln();
        rets.push(ret);
        out_dates.push(dates[i]);
    }
    
    (out_dates, rets)
}

pub fn make_rolling_windows(data: &[f64], window_size: usize) -> Vec<&[f64]> {
    let mut windows = Vec::new();
    if data.len() < window_size {
        return windows;
    }
    
    for i in 0..=(data.len() - window_size) {
        windows.push(&data[i..i + window_size]);
    }
    
    windows
}

pub fn rolling_distribution_p_values<F>(
    dates: &[NaiveDate],
    returns: &[f64],
    window_size: usize,
    fit_fn: F,
) -> Result<Vec<(NaiveDate, f64)>>
where
    F: Fn(&[f64]) -> Result<Box<dyn FittedDistribution>>,
{
    if returns.len() < window_size {
        return Err(anyhow::anyhow!("Data shorter than window size"));
    }

    let mut results = Vec::new();
    
    for i in 0..=(returns.len() - window_size) {
        let window_data = &returns[i..i + window_size];
        let end_date = dates[i + window_size - 1];
        
        match fit_fn(window_data) {
            Ok(fit) => {
                if let Ok(test_res) = kolmogorov_smirnov_test(window_data, fit.as_ref()) {
                    results.push((end_date, test_res.p_value));
                }
            }
            Err(_) => {
                // Skip if fit fails
            }
        }
    }
    
    Ok(results)
}
