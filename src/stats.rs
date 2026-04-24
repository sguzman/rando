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

pub fn moment(data: &[f64], n: u32) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().map(|&x| x.powi(n as i32)).sum::<f64>() / data.len() as f64
}

pub fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

pub fn quantile(data: &[f64], q: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = q * (sorted.len() - 1) as f64;
    let i = pos.floor() as usize;
    let f = pos - i as f64;
    if i + 1 < sorted.len() {
        (1.0 - f) * sorted[i] + f * sorted[i+1]
    } else {
        sorted[i]
    }
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

pub fn harmonic_mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let sum_inv: f64 = data.iter().map(|&x| 1.0 / x).sum();
    if sum_inv == 0.0 { return 0.0; }
    data.len() as f64 / sum_inv
}

pub fn geometric_mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let sum_ln: f64 = data.iter().map(|&x| x.ln()).sum();
    (sum_ln / data.len() as f64).exp()
}

pub fn contraharmonic_mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let sum_sq: f64 = data.iter().map(|&x| x * x).sum();
    let sum: f64 = data.iter().sum();
    if sum == 0.0 { return 0.0; }
    sum_sq / sum
}

pub fn commonest(data: &[f64]) -> Vec<f64> {
    if data.is_empty() { return vec![]; }
    use std::collections::HashMap;
    let mut counts = HashMap::new();
    for &x in data {
        let bits = x.to_bits();
        *counts.entry(bits).or_insert(0) += 1;
    }
    let max_count = *counts.values().max().unwrap();
    counts.into_iter()
        .filter(|&(_, count)| count == max_count)
        .map(|(bits, _)| f64::from_bits(bits))
        .collect()
}

pub fn biweight_location(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let med = median(data);
    let m = mad(data);
    if m == 0.0 { return med; }
    let c = 6.0; 
    let mut numer = 0.0;
    let mut denom = 0.0;
    for &x in data {
        let u = (x - med) / (c * m);
        if u.abs() < 1.0 {
            let w = (1.0 - u * u).powi(2);
            numer += w * x;
            denom += w;
        }
    }
    if denom == 0.0 { return med; }
    numer / denom
}

pub fn interquartile_range(data: &[f64]) -> f64 {
    let q3 = quantile(data, 0.75);
    let q1 = quantile(data, 0.25);
    q3 - q1
}

pub fn quartile_deviation(data: &[f64]) -> f64 {
    interquartile_range(data) / 2.0
}

pub fn quartiles(data: &[f64]) -> [f64; 3] {
    [
        quantile(data, 0.25),
        quantile(data, 0.50),
        quantile(data, 0.75),
    ]
}

pub fn quartile_skewness(data: &[f64]) -> f64 {
    let q = quartiles(data);
    if q[2] == q[0] { return 0.0; }
    (q[2] + q[0] - 2.0 * q[1]) / (q[2] - q[0])
}

pub fn entropy(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    use std::collections::HashMap;
    let mut counts = HashMap::new();
    for &x in data {
        let bits = x.to_bits();
        *counts.entry(bits).or_insert(0) += 1;
    }
    let n = data.len() as f64;
    counts.values().map(|&count| {
        let p = count as f64 / n;
        -p * p.ln()
    }).sum()
}

pub fn mean_deviation(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let m = mean(data);
    data.iter().map(|&x| (x - m).abs()).sum::<f64>() / data.len() as f64
}

pub fn qn_dispersion(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 { return 0.0; }
    let mut diffs = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            diffs.push((data[i] - data[j]).abs());
        }
    }
    diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = (n * (n - 1) / 2) / 4; 
    diffs[k] * 2.21914 
}

pub fn sn_dispersion(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 { return 0.0; }
    let mut medians_i = Vec::with_capacity(n);
    for i in 0..n {
        let mut diffs_j = Vec::with_capacity(n - 1);
        for j in 0..n {
            if i == j { continue; }
            diffs_j.push((data[i] - data[j]).abs());
        }
        medians_i.push(median(&diffs_j));
    }
    median(&medians_i) * 1.1926 
}

pub fn trimmed_variance(data: &[f64], fraction: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let k = (n as f64 * fraction).round() as usize;
    if 2 * k >= n { return 0.0; }
    let trimmed = &sorted[k..(n - k)];
    variance(trimmed)
}

pub fn winsorized_variance(data: &[f64], fraction: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let k = (n as f64 * fraction).round() as usize;
    if 2 * k >= n { return 0.0; }
    
    let low_val = sorted[k];
    let high_val = sorted[n - k - 1];
    
    for i in 0..k {
        sorted[i] = low_val;
    }
    for i in (n - k)..n {
        sorted[i] = high_val;
    }
    
    variance(&sorted)
}

pub fn spatial_median(data: &[Vec<f64>], tol: f64, max_iter: usize) -> Vec<f64> {
    if data.is_empty() { return vec![]; }
    let dims = data[0].len();
    if data.iter().any(|v| v.len() != dims) { return vec![]; }
    
    // Initial guess: geometric mean of points
    let mut current_median = vec![0.0; dims];
    for v in data {
        for i in 0..dims {
            current_median[i] += v[i];
        }
    }
    for i in 0..dims {
        current_median[i] /= data.len() as f64;
    }
    
    for _ in 0..max_iter {
        let mut numer = vec![0.0; dims];
        let mut denom = 0.0;
        let mut all_at_median = true;
        
        for v in data {
            let dist = v.iter().zip(current_median.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            
            if dist > tol {
                let weight = 1.0 / dist;
                for i in 0..dims {
                    numer[i] += v[i] * weight;
                }
                denom += weight;
                all_at_median = false;
            }
        }
        
        if all_at_median || denom == 0.0 { break; }
        
        let mut next_median = vec![0.0; dims];
        let mut max_diff = 0.0f64;
        for i in 0..dims {
            next_median[i] = numer[i] / denom;
            max_diff = max_diff.max((next_median[i] - current_median[i]).abs());
        }
        
        current_median = next_median;
        if max_diff < tol { break; }
    }
    
    current_median
}
