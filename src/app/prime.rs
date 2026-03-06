use std::fs;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, anyhow};
use serde_json::Value;

pub(super) fn fetch_prime_metrics(run_id: &str) -> Result<Value> {
    run_prime_json(&["rl", "metrics", run_id])
        .with_context(|| format!("`prime rl metrics {run_id}` failed"))
}

pub(super) fn fetch_prime_distributions(run_id: &str, step: Option<i64>) -> Result<Value> {
    match step {
        Some(s) => {
            let s_arg = s.to_string();
            run_prime_json(&["rl", "distributions", run_id, "-s", s_arg.as_str()])
                .with_context(|| format!("`prime rl distributions {run_id} --step {s}` failed"))
        }
        None => run_prime_json(&["rl", "distributions", run_id])
            .with_context(|| format!("`prime rl distributions {run_id}` failed")),
    }
}

pub(super) fn fetch_prime_progress(run_id: &str) -> Result<Value> {
    run_prime_json(&["rl", "progress", run_id])
        .with_context(|| format!("`prime rl progress {run_id}` failed"))
}

pub(super) fn fetch_prime_checkpoints(run_id: &str) -> Result<Value> {
    run_prime_json(&["rl", "checkpoints", run_id, "-o", "json"])
        .with_context(|| format!("`prime rl checkpoints {run_id} -o json` failed"))
}

pub(super) fn fetch_prime_logs(run_id: &str, tail: usize) -> Result<String> {
    let tail_str = tail.to_string();
    run_prime_text(&["rl", "logs", run_id, "--tail", tail_str.as_str()])
        .with_context(|| format!("`prime rl logs {run_id}` failed"))
}

pub(super) fn run_prime_json(args: &[&str]) -> Result<Value> {
    let text = run_prime_text(args)?;
    serde_json::from_str(&text).with_context(|| format!("prime output was not valid JSON: {text}"))
}

pub(super) fn run_prime_text(args: &[&str]) -> Result<String> {
    let output = Command::new("prime")
        .args(args)
        .output()
        .with_context(|| format!("failed to run `prime {}`", args.join(" ")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(anyhow!("prime command failed: {stderr}"));
    }

    String::from_utf8(output.stdout).context("prime output was not UTF-8")
}

pub(super) fn read_json_file(path: &PathBuf) -> Result<Value> {
    let text = fs::read_to_string(path)?;
    serde_json::from_str(&text).with_context(|| format!("invalid JSON at {}", path.display()))
}
