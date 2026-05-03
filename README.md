# Rando

Rando is a Rust statistics and simulation toolkit oriented around fitted distributions, rolling tests, and time-series style experimentation.

## Intent

Package recurring statistical workflows such as distribution fitting, goodness-of-fit testing, and rolling analysis into reusable Rust code instead of notebook-only experiments.

## Ambition

The current library modules and example `main.rs` suggest a broader quantitative-analysis helper library that could support simulation, verification, plotting, and applied research workflows.

## Current Status

The repository already contains reusable modules, plotting support, verification tests, and a demo-style binary that generates synthetic data and runs statistical checks.

## Core Capabilities Or Focus Areas

- Synthetic data generation for experiments.
- Distribution fitting and statistical testing.
- Rolling-window p-value analysis.
- Plot generation for results.
- Reusable library modules alongside a demo binary.

## Project Layout

- `src/`: Rust source for the main crate or application entrypoint.
- `Cargo.toml`: crate or workspace manifest and the first place to check for package structure.

## Setup And Requirements

- Rust toolchain.
- Input data or willingness to use synthetic/demo workflows.
- A local environment that can write plot outputs such as PNG files.

## Build / Run / Test Commands

```bash
cargo build
cargo test
cargo run
```

## Notes, Limitations, Or Known Gaps

- The current binary behaves more like a worked example than a production CLI.
- Some outputs are generated as files in the project root, which is useful for experimentation but worth documenting for repeatability.

## Next Steps Or Roadmap Hints

- Separate library APIs from demo binaries more explicitly if this becomes a shared dependency.
- Add clearer CLI/config surfaces if non-developer users will run statistical workflows directly.
