use crate::distrs::FittedDistribution;
use anyhow::Result;

pub struct KSTestResult {
    pub statistic: f64,
    pub p_value: f64,
}

pub struct ADTestResult {
    pub statistic: f64,
    pub p_value: f64,
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
    
    // P-value approximation for AD (general approach)
    // Note: True p-value depends on the distribution. This is a heuristic.
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
    // Heuristic p-value for Anderson-Darling
    // Based on asymptotic distribution
    if a2 <= 0.0 { return 1.0; }
    
    if a2 >= 0.6 {
        1.2937 * (-2.2568 * a2).exp() + 0.011
    } else if a2 >= 0.34 {
        1.2937 * (-2.2568 * a2).exp() + 0.011 // Simple placeholder
    } else {
        1.0 - (1.0 - (0.01 / a2)).exp() // Placeholder
    }.clamp(0.0, 1.0)
}
