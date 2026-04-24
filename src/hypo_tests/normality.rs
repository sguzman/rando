use crate::hypo_tests::TestResult;
use statrs::distribution::{Normal, ContinuousCDF};

pub fn shapiro_wilk_test(data: &[f64]) -> TestResult {
    let n = data.len();
    if n < 3 {
        return TestResult { statistic: 0.0, p_value: 1.0 };
    }
    
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let m = crate::stats::mean(&sorted);
    let ss: f64 = sorted.iter().map(|&x| (x - m).powi(2)).sum();
    if ss == 0.0 { return TestResult { statistic: 1.0, p_value: 1.0 }; }
    
    // Royston approximation for weights
    let n_f = n as f64;
    let mut a = Vec::with_capacity(n);
    let normal = Normal::new(0.0, 1.0).unwrap();
    
    let mut m_vec = Vec::with_capacity(n);
    for i in 1..=n {
        let p = (i as f64 - 3.0/8.0) / (n_f + 1.0/4.0);
        m_vec.push(normal.inverse_cdf(p));
    }
    
    let m_sum_sq: f64 = m_vec.iter().map(|&x| x * x).sum();
    let m_sqrt = m_sum_sq.sqrt();
    
    for i in 0..n {
        a.push(m_vec[i] / m_sqrt);
    }
    
    // Adjustment for small samples (simplified)
    if n > 3 {
        // In a full implementation, we'd adjust a[0] and a[1] here
    }
    
    let mut b = 0.0;
    for i in 0..n {
        b += a[i] * sorted[i];
    }
    
    let w = b.powi(2) / ss;
    
    // P-value approximation (Royston)
    let mu = if n < 12 {
        -1.3411 + 1.258 * (n_f.ln())
    } else {
        -1.5861 + 1.411 * (n_f.ln())
    };
    
    let sigma = if n < 12 {
        (0.6853 - 0.1586 * n_f.ln()).exp()
    } else {
        (0.4856 - 0.0826 * n_f.ln()).exp()
    };
    
    let y = (1.0 - w).ln();
    let z = (y - mu) / sigma;
    let p_value = 1.0 - normal.cdf(z);
    
    TestResult { statistic: w, p_value }
}
