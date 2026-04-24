use rando::*;
use chrono::NaiveDate;
use anyhow::Result;

fn main() -> Result<()> {
    // 1. Generate synthetic price data (Random Walk)
    let n = 300;
    let start_date = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
    let mut dates = Vec::new();
    let mut prices = Vec::new();
    let mut current_price = 100.0;
    
    for i in 0..n {
        dates.push(start_date + chrono::Duration::days(i as i64));
        prices.push(current_price);
        current_price *= 1.0 + (rand::random::<f64>() - 0.5) * 0.02;
    }

    println!("Generated {} price points.", n);

    // 2. Compute returns
    let (ret_dates, returns) = get_return_series(&dates, &prices);
    println!("Computed {} return points.", returns.len());

    // 3. Run rolling distribution p-values for Cauchy distribution (using MLE)
    println!("Running rolling Cauchy MLE fitting...");
    let window_size = 60;
    let results = rolling_distribution_p_values(&ret_dates, &returns, window_size, |data| {
        let fit = CauchyFit::fit(data)?;
        Ok(Box::new(fit) as Box<dyn FittedDistribution>)
    })?;

    println!("Rolling Cauchy p-values (first 5):");
    for (date, p) in results.iter().take(5) {
        println!("{}: {:.4}", date, p);
    }

    // 4. Plot results
    println!("Generating plot: cauchy_p_values.png");
    plot_rolling_p_values("cauchy_p_values.png", "Rolling Cauchy P-Values", &results)?;

    // 5. Test Anderson-Darling on all returns
    let final_fit = NormalFit::fit(&returns)?;
    let ad_res = anderson_darling_test(&returns, &final_fit)?;
    println!("\nAnderson-Darling Test (Normal Fit):");
    println!("Statistic: {:.4}", ad_res.statistic);
    println!("P-Value: {:.4}", ad_res.p_value);

    Ok(())
}
