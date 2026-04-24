use plotters::prelude::*;
use chrono::NaiveDate;
use anyhow::Result;

pub fn plot_rolling_p_values(
    path: &str,
    title: &str,
    series: &[(NaiveDate, f64)],
) -> Result<()> {
    let root = BitMapBackend::new(path, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_date = series.iter().map(|(d, _)| *d).min().unwrap();
    let max_date = series.iter().map(|(d, _)| *d).max().unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min_date..max_date, 0.0..1.0)?;

    chart.configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .disable_mesh()
        .x_label_formatter(&|d| d.format("%Y-%m-%d").to_string())
        .draw()?;

    // Draw significance line at 0.05
    chart.draw_series(LineSeries::new(
        vec![(min_date, 0.05), (max_date, 0.05)],
        &RED.mix(0.5),
    ))?
    .label("Significance (0.05)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.mix(0.5)));

    chart.draw_series(LineSeries::new(
        series.iter().map(|(d, p)| (*d, *p)),
        &BLUE,
    ))?
    .label("P-Value")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}
