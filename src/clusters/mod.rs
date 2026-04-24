use anyhow::Result;

pub fn find_clusters(data: &[f64], k: usize) -> Result<Vec<Vec<f64>>> {
    if data.is_empty() || k == 0 { return Ok(vec![]); }
    if k >= data.len() { return Ok(data.iter().map(|&x| vec![x]).collect()); }

    let mut centroids: Vec<f64> = (0..k).map(|_| {
        let idx = rand::random_range(0..data.len());
        data[idx]
    }).collect();

    let mut clusters = vec![vec![]; k];
    for _ in 0..100 { // Max iterations
        clusters = vec![vec![]; k];
        for &x in data {
            let mut min_dist = f64::INFINITY;
            let mut best_k = 0;
            for (i, &c) in centroids.iter().enumerate() {
                let d = (x - c).abs();
                if d < min_dist {
                    min_dist = d;
                    best_k = i;
                }
            }
            clusters[best_k].push(x);
        }

        // Update centroids
        let mut changed = false;
        for i in 0..k {
            if clusters[i].is_empty() { continue; }
            let new_c = crate::stats::mean(&clusters[i]);
            if (new_c - centroids[i]).abs() > 1e-6 {
                centroids[i] = new_c;
                changed = true;
            }
        }
        if !changed { break; }
    }

    Ok(clusters)
}
