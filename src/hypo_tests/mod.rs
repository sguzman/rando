pub mod t_test;
pub mod z_test;
pub mod normality;
pub mod correlation;

use crate::distrs::FittedDistribution;
use anyhow::Result;
use statrs::distribution::{ChiSquared, ContinuousCDF};

pub struct TestResult {
    pub statistic: f64,
    pub p_value: f64,
}

pub fn t_test(data: &[f64], mu0: f64) -> Result<TestResult> {
    let res = t_test::t_test_one_sample(data, mu0);
    Ok(TestResult { statistic: res.statistic, p_value: res.p_value })
}

pub fn z_test(data: &[f64], mu0: f64, sigma: f64) -> Result<TestResult> {
    let res = z_test::z_test_one_sample(data, mu0, sigma);
    Ok(TestResult { statistic: res.statistic, p_value: res.p_value })
}

pub struct KSTestResult {
    pub statistic: f64,
    pub p_value: f64,
}

pub struct ADTestResult {
    pub statistic: f64,
    pub p_value: f64,
}

pub fn jarque_bera_test(data: &[f64]) -> TestResult {
    let n = data.len() as f64;
    let s = crate::stats::skewness(data);
    let k = crate::stats::kurtosis(data);
    let statistic = (n / 6.0) * (s.powi(2) + (k - 3.0).powi(2) / 4.0);
    let chi2 = ChiSquared::new(2.0).unwrap();
    let p_value = 1.0 - chi2.cdf(statistic);
    TestResult { statistic, p_value }
}

pub fn pearson_chi_square_test(observed: &[f64], expected: &[f64]) -> TestResult {
    let mut statistic = 0.0;
    for i in 0..observed.len() {
        statistic += (observed[i] - expected[i]).powi(2) / expected[i];
    }
    let df = (observed.len() - 1) as f64;
    let chi2 = ChiSquared::new(df).unwrap();
    let p_value = 1.0 - chi2.cdf(statistic);
    TestResult { statistic, p_value }
}

pub fn variance_test(data: &[f64], sigma0_sq: f64) -> TestResult {
    let n = data.len() as f64;
    if n < 2.0 { return TestResult { statistic: 0.0, p_value: 1.0 }; }
    let s_sq = crate::stats::variance(data);
    let statistic = (n - 1.0) * s_sq / sigma0_sq;
    let chi2 = ChiSquared::new(n - 1.0).unwrap();
    // Two-sided test
    let p1 = chi2.cdf(statistic);
    let p2 = 1.0 - p1;
    let p_value = 2.0 * p1.min(p2);
    TestResult { statistic, p_value }
}

pub fn levene_test(groups: &[&[f64]]) -> TestResult {
    // Levene's test using absolute deviations from the median (Brown-Forsythe)
    let k = groups.len() as f64;
    let mut n_total = 0.0;
    let mut z_groups = Vec::new();
    for g in groups {
        let med = crate::stats::median(g);
        let zg: Vec<f64> = g.iter().map(|&x| (x - med).abs()).collect();
        n_total += g.len() as f64;
        z_groups.push(zg);
    }
    
    let z_flat: Vec<f64> = z_groups.iter().flatten().cloned().collect();
    let z_grand_mean = crate::stats::mean(&z_flat);
    
    let mut numer = 0.0;
    for zg in &z_groups {
        let group_mean = crate::stats::mean(zg);
        numer += zg.len() as f64 * (group_mean - z_grand_mean).powi(2);
    }
    numer *= n_total - k;
    
    let mut denom = 0.0;
    for zg in &z_groups {
        let group_mean = crate::stats::mean(zg);
        for &z in zg {
            denom += (z - group_mean).powi(2);
        }
    }
    denom *= k - 1.0;
    
    let statistic = if denom == 0.0 { 0.0 } else { numer / denom };
    
    use statrs::distribution::FisherSnedecor;
    let f_dist = FisherSnedecor::new(k - 1.0, n_total - k).unwrap();
    let p_value = 1.0 - f_dist.cdf(statistic);
    
    TestResult { statistic, p_value }
}

pub fn kolmogorov_smirnov_test(data: &[f64], dist: &dyn FittedDistribution) -> Result<KSTestResult> {
    if data.is_empty() {
        return Err(anyhow::anyhow!("Empty data"));
    }

    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = data.len() as f64;
    let mut d_max = 0.0f64;

    for (i, &x) in sorted_data.iter().enumerate() {
        let theoretical_cdf = dist.cdf(x);
        let empirical_cdf_low = i as f64 / n;
        let empirical_cdf_high = (i + 1) as f64 / n;
        
        let d1 = (empirical_cdf_high - theoretical_cdf).abs();
        let d2 = (theoretical_cdf - empirical_cdf_low).abs();
        
        d_max = d_max.max(d1).max(d2);
    }

    let stat = (n.sqrt() + 0.12 + 0.11 / n.sqrt()) * d_max;
    let p_value = compute_kolmogorov_p_value(stat);

    Ok(KSTestResult {
        statistic: d_max,
        p_value,
    })
}

pub fn anderson_darling_test(data: &[f64], dist: &dyn FittedDistribution) -> Result<ADTestResult> {
    if data.is_empty() {
        return Err(anyhow::anyhow!("Empty data"));
    }

    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = data.len() as f64;
    let mut sum_terms = 0.0;

    for (i, &x) in sorted_data.iter().enumerate() {
        let f_x = dist.cdf(x).clamp(1e-10, 1.0 - 1e-10);
        let f_rev_x = dist.cdf(sorted_data[n as usize - 1 - i]).clamp(1e-10, 1.0 - 1e-10);
        
        let k = (2 * i + 1) as f64;
        sum_terms += k * (f_x.ln() + (1.0 - f_rev_x).ln());
    }

    let a2 = -n - (sum_terms / n);
    let p_value = compute_ad_p_value(a2);

    Ok(ADTestResult {
        statistic: a2,
        p_value,
    })
}

fn compute_kolmogorov_p_value(stat: f64) -> f64 {
    if stat < 0.2 {
        return 1.0;
    }
    
    let mut sum = 0.0;
    for k in 1..=100 {
        let term = (-1.0f64).powi(k - 1) * (-2.0 * (k as f64).powi(2) * stat.powi(2)).exp();
        sum += term;
        if term.abs() < 1e-10 {
            break;
        }
    }
    (2.0 * sum).clamp(0.0, 1.0)
}

fn compute_ad_p_value(a2: f64) -> f64 {
    if a2 <= 0.0 { return 1.0; }
    
    if a2 >= 0.6 {
        1.2937 * (-2.2568 * a2).exp() + 0.011
    } else if a2 >= 0.34 {
        1.2937 * (-2.2568 * a2).exp() + 0.011 
    } else {
        1.0 - (1.0 - (0.01 / a2)).exp() 
    }.clamp(0.0, 1.0)
}
