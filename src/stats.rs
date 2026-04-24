pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

pub fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let m = mean(data);
    let var = data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    var.sqrt()
}

pub fn median(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

pub fn quantile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let index = p * (n - 1) as f64;
    let i = index.floor() as usize;
    let frac = index - i as f64;
    if i + 1 < n {
        sorted[i] * (1.0 - frac) + sorted[i + 1] * frac
    } else {
        sorted[i]
    }
}

pub fn harmonic_mean(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let sum_recip: f64 = data.iter().map(|&x| 1.0 / x).sum();
    n / sum_recip
}

pub fn geometric_mean(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let sum_log: f64 = data.iter().map(|&x| x.ln()).sum();
    (sum_log / n).exp()
}

pub fn trimmed_mean(data: &[f64], alpha: f64) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = data.len();
    let k = (n as f64 * alpha).floor() as usize;
    let trimmed = &sorted[k..n-k];
    if trimmed.is_empty() { return 0.0; }
    trimmed.iter().sum::<f64>() / trimmed.len() as f64
}

pub fn winsorized_mean(data: &[f64], alpha: f64) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = data.len();
    let k = (n as f64 * alpha).floor() as usize;
    if n == 0 { return 0.0; }
    
    let low_val = sorted[k];
    let high_val = sorted[n - k - 1];
    
    for i in 0..k {
        sorted[i] = low_val;
    }
    for i in (n - k)..n {
        sorted[i] = high_val;
    }
    
    sorted.iter().sum::<f64>() / n as f64
}
