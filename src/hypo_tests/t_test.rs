use statrs::distribution::{StudentsT, ContinuousCDF};

pub struct TTestResult {
    pub statistic: f64,
    pub p_value: f64,
    pub df: f64,
}

pub fn t_test_one_sample(data: &[f64], mu0: f64) -> TTestResult {
    let n = data.len() as f64;
    let m = crate::stats::mean(data);
    let s = crate::stats::std_dev(data);
    let statistic = (m - mu0) / (s / n.sqrt());
    let df = n - 1.0;
    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
    let p_value = 2.0 * (1.0 - t_dist.cdf(statistic.abs()));
    TTestResult { statistic, p_value, df }
}

pub fn t_test_two_sample(data1: &[f64], data2: &[f64]) -> TTestResult {
    let n1 = data1.len() as f64;
    let n2 = data2.len() as f64;
    let m1 = crate::stats::mean(data1);
    let m2 = crate::stats::mean(data2);
    let v1 = crate::stats::variance(data1);
    let v2 = crate::stats::variance(data2);
    
    // Welch's t-test (unequal variances)
    let statistic = (m1 - m2) / (v1/n1 + v2/n2).sqrt();
    let df = (v1/n1 + v2/n2).powi(2) / ((v1/n1).powi(2)/(n1-1.0) + (v2/n2).powi(2)/(n2-1.0));
    
    let t_dist = StudentsT::new(0.0, 1.0, df).unwrap();
    let p_value = 2.0 * (1.0 - t_dist.cdf(statistic.abs()));
    TTestResult { statistic, p_value, df }
}
