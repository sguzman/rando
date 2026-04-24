pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

pub fn median(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

pub fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let m = mean(data);
    data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}

pub fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

pub fn quantile(data: &[f64], q: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (q * (sorted.len() - 1) as f64) as usize;
    sorted[idx]
}

pub fn skewness(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 3.0 { return 0.0; }
    let m = mean(data);
    let mut m3 = 0.0;
    let mut m2 = 0.0;
    for &x in data {
        let diff = x - m;
        m3 += diff.powi(3);
        m2 += diff.powi(2);
    }
    m3 /= n;
    m2 /= n;
    let sigma = m2.sqrt();
    if sigma == 0.0 { return 0.0; }
    m3 / sigma.powi(3)
}

pub fn kurtosis(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 4.0 { return 0.0; }
    let m = mean(data);
    let mut m4 = 0.0;
    let mut m2 = 0.0;
    for &x in data {
        let diff = x - m;
        let diff2 = diff * diff;
        m4 += diff2 * diff2;
        m2 += diff2;
    }
    m4 /= n;
    m2 /= n;
    if m2 == 0.0 { return 0.0; }
    m4 / (m2 * m2)
}

pub fn trimmed_mean(data: &[f64], fraction: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let k = (n as f64 * fraction).round() as usize;
    if 2 * k >= n { return median(data); }
    let trimmed = &sorted[k..(n - k)];
    mean(trimmed)
}

pub fn winsorized_mean(data: &[f64], fraction: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let k = (n as f64 * fraction).round() as usize;
    if 2 * k >= n { return median(data); }
    
    let low_val = sorted[k];
    let high_val = sorted[n - k - 1];
    
    for i in 0..k {
        sorted[i] = low_val;
    }
    for i in (n - k)..n {
        sorted[i] = high_val;
    }
    
    mean(&sorted)
}

pub fn mad(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let m = median(data);
    let abs_diffs: Vec<f64> = data.iter().map(|&x| (x - m).abs()).collect();
    median(&abs_diffs)
}

pub fn moving_average(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || window > data.len() { return vec![]; }
    let mut result = Vec::with_capacity(data.len() - window + 1);
    for i in 0..=(data.len() - window) {
        let sum: f64 = data[i..i + window].iter().sum();
        result.push(sum / window as f64);
    }
    result
}

pub fn moving_median(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || window > data.len() { return vec![]; }
    let mut result = Vec::with_capacity(data.len() - window + 1);
    for i in 0..=(data.len() - window) {
        let mut slice = data[i..i + window].to_vec();
        slice.sort_by(|a, b| a.partial_cmp(b).unwrap());
        result.push(median(&slice));
    }
    result
}
