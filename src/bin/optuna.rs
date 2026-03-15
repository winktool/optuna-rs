//! Optuna CLI 二进制入口点。
//!
//! ```bash
//! cargo run --features cli -- create-study --storage journal.log --study-name test --direction minimize
//! ```

#[cfg(feature = "cli")]
fn main() {
    use clap::Parser;
    use optuna_rs::cli::commands::{Cli, run};

    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "cli"))]
fn main() {
    eprintln!("CLI feature is not enabled. Rebuild with --features cli");
    std::process::exit(1);
}
