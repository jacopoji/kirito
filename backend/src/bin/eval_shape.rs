//! Headless runner for `evaluate_transparent_shape` - drives every ground-truth
//! fixture through the shipped and candidate `TransparentShapeSolver` in one
//! command, instead of clicking "Evaluate against ground truth" through the
//! debug UI's fixture picker once per fixture. Prints the same
//! `frame_evaluation` summary lines that would otherwise only show up in
//! `log.txt`. Still links `platforms`, so it inherits the same
//! `requireAdministrator` manifest as the rest of this workspace - run it
//! from an elevated terminal.
//!
//! Usage: `cargo run --bin eval_shape [fixture]`, e.g.
//! `cargo run --bin eval_shape failure_4`. With no argument, runs all six.

use backend::{TransparentShapeDifficulty, evaluate_transparent_shape_direct};

struct StdoutLogger;

impl log::Log for StdoutLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::LevelFilter::Debug
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        // Only the lines this tool cares about - `evaluate_transparent_shape`
        // also logs a full per-frame candidate/track dump at Debug level
        // (thousands of lines per fixture) meant for `log.txt` grepping, not
        // a terminal.
        let message = record.args().to_string();
        if message.starts_with("frame_evaluation")
            || message.starts_with("starting transparent shape evaluation")
            || message.starts_with("evaluating fixture")
        {
            println!("{message}");
        }
    }

    fn flush(&self) {}
}

fn main() {
    log::set_logger(&StdoutLogger).expect("logger already set");
    log::set_max_level(log::LevelFilter::Debug);

    let requested = std::env::args().nth(1);
    let all = [
        ("normal", TransparentShapeDifficulty::Normal),
        ("hard", TransparentShapeDifficulty::Hard),
        ("failure", TransparentShapeDifficulty::Failure),
        ("failure_2", TransparentShapeDifficulty::Failure2),
        ("failure_3", TransparentShapeDifficulty::Failure3),
        ("failure_4", TransparentShapeDifficulty::Failure4),
    ];

    let selected: Vec<_> = match &requested {
        Some(name) => all
            .into_iter()
            .filter(|&(fixture_name, _)| fixture_name == name)
            .collect(),
        None => all.to_vec(),
    };

    if selected.is_empty() {
        eprintln!(
            "no fixture named {requested:?} - expected one of: normal, hard, failure, failure_2, failure_3, failure_4"
        );
        std::process::exit(1);
    }

    for (_, difficulty) in selected {
        evaluate_transparent_shape_direct(difficulty);
    }
}
