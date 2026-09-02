//! Public `binder-llm` CLI contract (Phase 1).
//!
//! The executable is the public surface for binary distribution.
//! The library API remains the primary embed path (Kellnr / private source).

use std::env;
use std::process::ExitCode;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const GIT_SHA: &str = env!("BINDER_LLM_GIT_SHA");
const BUILD_DATE: &str = env!("BINDER_LLM_BUILD_DATE");

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(code) => code,
    }
}

fn run(args: &[String]) -> Result<(), ExitCode> {
    if args.is_empty() {
        print_help();
        return Ok(());
    }

    match args[0].as_str() {
        "-h" | "--help" | "help" => {
            print_help();
            Ok(())
        }
        "-V" | "--version" | "version" => {
            print_version();
            Ok(())
        }
        other => {
            eprintln!("error: unknown argument '{other}'");
            eprintln!();
            print_help_to_stderr();
            Err(ExitCode::from(2))
        }
    }
}

fn print_version() {
    println!("binder-llm {VERSION}");
    println!("commit: {GIT_SHA}");
    println!("built: {BUILD_DATE}");
}

fn print_help() {
    println!(
        "\
binder-llm {VERSION}

binder-llm is a provider-agnostic LLM transport library. This binary is the
public distribution surface for install/verify; applications should embed the
Rust crate, not shell out to this CLI for inference.

USAGE:
    binder-llm --version
    binder-llm --help

OPTIONS:
    -h, --help       Print help
    -V, --version    Print version and build provenance

Source is privately maintained. See binder-llm-releases for installation and
license.
"
    );
}

fn print_help_to_stderr() {
    eprintln!(
        "\
USAGE:
    binder-llm --version
    binder-llm --help
"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provenance_constants_are_populated() {
        assert!(!VERSION.is_empty());
        assert_ne!(GIT_SHA, "unknown", "build.rs failed to inject git SHA");
        assert_ne!(
            BUILD_DATE, "unknown",
            "build.rs failed to inject build date"
        );
    }
}
