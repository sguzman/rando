use statrs::distribution::{Normal, ContinuousCDF};

pub struct ZTestResult {
    pub statistic: f64,
    pub p_value: f64,
}

pub fn z_test_one_sample(data: &[f64], mu0: f64, sigma: f64) -> ZTestResult {
    let n = data.len() as f64;
    let m = crate::stats::mean(data);
    let statistic = (m - mu0) / (sigma / n.sqrt());
    let normal = Normal::new(0.0, 1.0).unwrap();
    let p_value = 2.0 * (1.0 - normal.cdf(statistic.abs()));
    ZTestResult { statistic, p_value }
}
