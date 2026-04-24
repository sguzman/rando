use crate::hypo_tests::TestResult;
use statrs::distribution::{StudentsT, ContinuousCDF};

pub fn correlation_test(x: &[f64], y: &[f64]) -> TestResult {
    if x.len() != y.len() || x.len() < 3 {
        return TestResult { statistic: 0.0, p_value: 1.0 };
    }
    
    let n = x.len() as f64;
    let mx = crate::stats::mean(x);
    let my = crate::stats::mean(y);
    
    let mut numer = 0.0;
    let mut denx = 0.0;
    let mut deny = 0.0;
    
    for i in 0..x.len() {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        numer += dx * dy;
        denx += dx * dx;
        deny += dy * dy;
    }
    
    let r = numer / (denx * deny).sqrt();
    if r.abs() >= 1.0 {
        return TestResult { statistic: r, p_value: 0.0 };
    }
    
    let t_stat = r * ((n - 2.0) / (1.0 - r * r)).sqrt();
    let t_dist = StudentsT::new(0.0, 1.0, n - 2.0).unwrap();
    let p_value = 2.0 * (1.0 - t_dist.cdf(t_stat.abs()));
    
    TestResult { statistic: r, p_value }
}
