use std::cmp::Ordering;

use anyhow::{Result, bail};
use serde_json::{Map, Value};

use super::model::*;

const MAX_HISTOGRAM_DEPTH: usize = 64;
const MAX_HISTOGRAM_NODES: usize = 4096;

pub(super) fn parse_metric_points(root: &Value) -> Result<(Vec<MetricPoint>, String)> {
    let metrics = if let Some(arr) = root.get("metrics").and_then(Value::as_array) {
        arr
    } else if let Some(arr) = root.as_array() {
        arr
    } else {
        bail!("metrics JSON must be either array or object with `metrics` array");
    };

    if metrics.is_empty() {
        return Ok((Vec::new(), "reward".to_string()));
    }

    let reward_key = detect_reward_key(metrics).unwrap_or_else(|| "reward".to_string());
    let points = metrics
        .iter()
        .enumerate()
        .filter_map(|(idx, row)| {
            let step = detect_step(row).unwrap_or(idx as i64);
            let reward = row.get(&reward_key).and_then(to_f64)?;
            Some(MetricPoint { step, reward })
        })
        .collect::<Vec<_>>();

    Ok((points, reward_key))
}

fn detect_reward_key(metrics: &[Value]) -> Option<String> {
    const PREFERRED_KEYS: &[&str] = &[
        "reward_mean",
        "reward",
        "mean_reward",
        "avg_reward",
        "train_reward",
        "reward_total",
        "score",
    ];

    let first_obj = metrics.first()?.as_object()?;
    for key in PREFERRED_KEYS {
        if first_obj.contains_key(*key) {
            return Some((*key).to_string());
        }
    }

    first_obj
        .iter()
        .find(|(k, v)| {
            let key = k.to_ascii_lowercase();
            (key.contains("reward") || key.contains("score")) && to_f64(v).is_some()
        })
        .map(|(k, _)| k.clone())
}

pub(super) fn detect_step(row: &Value) -> Option<i64> {
    const STEP_KEYS: &[&str] = &[
        "step",
        "global_step",
        "train_step",
        "iteration",
        "checkpoint_step",
    ];

    for key in STEP_KEYS {
        if let Some(v) = row.get(*key)
            && let Some(i) = to_i64(v)
        {
            return Some(i);
        }
    }

    None
}

pub(super) fn parse_distribution_steps(root: &Value) -> Vec<i64> {
    let mut steps = root
        .get("steps_with_distributions")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(to_i64).collect::<Vec<_>>())
        .unwrap_or_default();
    steps.sort_unstable();
    steps.dedup();
    steps
}

pub(super) fn merge_distribution_steps(a: &[i64], b: &[i64]) -> Vec<i64> {
    let mut left = a.to_vec();
    let mut right = b.to_vec();
    left.sort_unstable();
    left.dedup();
    right.sort_unstable();
    right.dedup();

    if !left.is_empty() && !right.is_empty() {
        let mut intersection = Vec::with_capacity(left.len().min(right.len()));
        let mut i = 0usize;
        let mut j = 0usize;
        while i < left.len() && j < right.len() {
            if left[i] == right[j] {
                intersection.push(left[i]);
                i += 1;
                j += 1;
            } else if left[i] < right[j] {
                i += 1;
            } else {
                j += 1;
            }
        }
        if !intersection.is_empty() {
            return intersection;
        }
    }

    let mut merged = left;
    merged.extend(right);
    merged.sort_unstable();
    merged.dedup();
    merged
}

fn to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

fn to_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|x| x as i64)),
        Value::String(s) => s.parse::<i64>().ok(),
        _ => None,
    }
}

pub(super) fn summarize_points(points: &[MetricPoint], window: usize, thresholds: &[f64]) -> Summary {
    let steps = points.len();
    let (best_step, best_reward) = points
        .iter()
        .max_by(|a, b| a.reward.partial_cmp(&b.reward).unwrap_or(Ordering::Equal))
        .map(|p| (p.step, p.reward))
        .unwrap_or((0, 0.0));

    let final_reward = points.last().map(|p| p.reward).unwrap_or(0.0);
    let auc_reward = auc(points);

    let tail_len = points.len().min(window.max(1));
    let tail = &points[points.len() - tail_len..];
    let last_mean = mean(tail.iter().map(|p| p.reward));
    let last_std = stddev(tail.iter().map(|p| p.reward), last_mean);
    let last_slope = linear_slope(tail);

    let early = segment_summary(points, 0.0, 1.0 / 3.0);
    let mid = segment_summary(points, 1.0 / 3.0, 2.0 / 3.0);
    let late = segment_summary(points, 2.0 / 3.0, 1.0);

    let time_to_threshold = thresholds
        .iter()
        .copied()
        .map(|t| {
            let step = points.iter().find(|p| p.reward >= t).map(|p| p.step);
            (t, step)
        })
        .collect::<Vec<_>>();

    Summary {
        steps,
        final_reward,
        best_reward,
        best_step,
        auc_reward,
        last_mean,
        last_std,
        last_slope,
        early,
        mid,
        late,
        time_to_threshold,
    }
}

fn segment_summary(points: &[MetricPoint], start_frac: f64, end_frac: f64) -> SegmentSummary {
    if points.is_empty() {
        return SegmentSummary {
            mean: 0.0,
            slope: 0.0,
        };
    }

    let len = points.len();
    let mut start = ((len as f64) * start_frac).floor() as usize;
    let mut end = ((len as f64) * end_frac).ceil() as usize;
    start = start.min(len.saturating_sub(1));
    end = end.max(start + 1).min(len);

    let slice = &points[start..end];
    SegmentSummary {
        mean: mean(slice.iter().map(|p| p.reward)),
        slope: linear_slope(slice),
    }
}

fn auc(points: &[MetricPoint]) -> f64 {
    if points.len() < 2 {
        return points.first().map(|p| p.reward).unwrap_or(0.0);
    }

    let mut total = 0.0;
    for pair in points.windows(2) {
        let p0 = &pair[0];
        let p1 = &pair[1];
        let dx = (p1.step - p0.step).max(1) as f64;
        total += 0.5 * (p0.reward + p1.reward) * dx;
    }
    total
}

fn mean<I>(it: I) -> f64
where
    I: Iterator<Item = f64>,
{
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in it {
        sum += value;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn stddev<I>(it: I, mu: f64) -> f64
where
    I: Iterator<Item = f64>,
{
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in it {
        sum += (value - mu).powi(2);
        count += 1;
    }
    if count < 2 {
        0.0
    } else {
        (sum / count as f64).sqrt()
    }
}

fn linear_slope(points: &[MetricPoint]) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }

    let n = points.len() as f64;
    let sx = points.iter().map(|p| p.step as f64).sum::<f64>();
    let sy = points.iter().map(|p| p.reward).sum::<f64>();
    let sxx = points
        .iter()
        .map(|p| (p.step as f64) * (p.step as f64))
        .sum::<f64>();
    let sxy = points
        .iter()
        .map(|p| (p.step as f64) * p.reward)
        .sum::<f64>();

    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-9 {
        0.0
    } else {
        (n * sxy - sx * sy) / denom
    }
}

pub(super) fn compare_runs(a: &RunData, b: &RunData, thresholds: &[f64]) -> Comparison {
    let sa = &a.summary;
    let sb = &b.summary;

    let mut rows = vec![
        metric_row("steps", sa.steps as f64, sb.steps as f64, false, |v| {
            format!("{}", v as usize)
        }),
        metric_row("final_reward", sa.final_reward, sb.final_reward, true, |v| {
            format!("{v:.4}")
        }),
        metric_row("best_reward", sa.best_reward, sb.best_reward, true, |v| {
            format!("{v:.4}")
        }),
        metric_row(
            "best_step (lower better)",
            sa.best_step as f64,
            sb.best_step as f64,
            false,
            |v| format!("{}", v as i64),
        ),
        metric_row("auc_reward", sa.auc_reward, sb.auc_reward, true, |v| {
            format!("{v:.2}")
        }),
        metric_row("last_window_mean", sa.last_mean, sb.last_mean, true, |v| {
            format!("{v:.4}")
        }),
        metric_row(
            "last_window_std (lower better)",
            sa.last_std,
            sb.last_std,
            false,
            |v| format!("{v:.4}"),
        ),
        metric_row("last_window_slope", sa.last_slope, sb.last_slope, true, |v| {
            format!("{v:.6}")
        }),
        metric_row("early_mean", sa.early.mean, sb.early.mean, true, |v| {
            format!("{v:.4}")
        }),
        metric_row("mid_mean", sa.mid.mean, sb.mid.mean, true, |v| {
            format!("{v:.4}")
        }),
        metric_row("late_mean", sa.late.mean, sb.late.mean, true, |v| {
            format!("{v:.4}")
        }),
        metric_row("early_slope", sa.early.slope, sb.early.slope, true, |v| {
            format!("{v:.6}")
        }),
        metric_row("mid_slope", sa.mid.slope, sb.mid.slope, true, |v| {
            format!("{v:.6}")
        }),
        metric_row("late_slope", sa.late.slope, sb.late.slope, true, |v| {
            format!("{v:.6}")
        }),
    ];

    for threshold in thresholds {
        let a_step = sa
            .time_to_threshold
            .iter()
            .find(|(th, _)| (*th - *threshold).abs() < f64::EPSILON)
            .and_then(|(_, s)| *s);
        let b_step = sb
            .time_to_threshold
            .iter()
            .find(|(th, _)| (*th - *threshold).abs() < f64::EPSILON)
            .and_then(|(_, s)| *s);

        let (a_str, b_str, delta, kind) = match (a_step, b_step) {
            (Some(x), Some(y)) => {
                let diff = y - x;
                let kind = if diff < 0 {
                    DeltaKind::Better
                } else if diff > 0 {
                    DeltaKind::Worse
                } else {
                    DeltaKind::Neutral
                };
                (x.to_string(), y.to_string(), format_signed_i64(diff), kind)
            }
            (None, Some(y)) => (
                "n/a".to_string(),
                y.to_string(),
                "B reached, A did not".to_string(),
                DeltaKind::Better,
            ),
            (Some(x), None) => (
                x.to_string(),
                "n/a".to_string(),
                "A reached, B did not".to_string(),
                DeltaKind::Worse,
            ),
            (None, None) => (
                "n/a".to_string(),
                "n/a".to_string(),
                "both not reached".to_string(),
                DeltaKind::Neutral,
            ),
        };

        rows.push(CompareRowData {
            metric: format!("time_to_{threshold:.2}"),
            a: a_str,
            b: b_str,
            delta,
            kind,
        });
    }

    let mut findings = Vec::new();
    let final_delta = sb.final_reward - sa.final_reward;
    if final_delta.abs() >= 0.02 {
        findings.push(format!(
            "Final reward changed by {:+.4} (B {} A).",
            final_delta,
            if final_delta > 0.0 { "above" } else { "below" }
        ));
    }

    let auc_delta = sb.auc_reward - sa.auc_reward;
    if auc_delta.abs() >= 0.02 {
        findings.push(format!(
            "Reward AUC delta is {:+.2}; B has {} integrated reward signal.",
            auc_delta,
            if auc_delta > 0.0 { "higher" } else { "lower" }
        ));
    }

    let late_mean_delta = sb.late.mean - sa.late.mean;
    if late_mean_delta.abs() >= 0.02 {
        findings.push(format!(
            "Late-phase mean reward changed by {:+.4}.",
            late_mean_delta
        ));
    }

    if sb.late.slope.abs() < 1e-4 {
        findings.push("Run B appears plateaued in the late segment (slope near zero).".to_string());
    }

    if findings.is_empty() {
        findings.push(
            "No large deltas passed default summary heuristics; inspect segment rows and chart."
                .to_string(),
        );
    }

    Comparison { rows, findings }
}

pub(super) fn build_distribution_comparison(
    a: &DistributionBundle,
    b: &DistributionBundle,
) -> DistributionComparison {
    let mut rows = vec![
        optional_metric_row(
            "distribution_step",
            a.step.map(|x| x as f64),
            b.step.map(|x| x as f64),
            false,
            |v| format!("{v:.0}"),
        ),
        optional_metric_row(
            "reward_count",
            a.reward.as_ref().map(|s| s.total_count),
            b.reward.as_ref().map(|s| s.total_count),
            true,
            |v| format!("{v:.0}"),
        ),
        optional_metric_row(
            "reward_mean",
            a.reward.as_ref().map(|s| s.mean),
            b.reward.as_ref().map(|s| s.mean),
            true,
            |v| format!("{v:.4}"),
        ),
        optional_metric_row(
            "reward_p10",
            a.reward.as_ref().map(|s| s.p10),
            b.reward.as_ref().map(|s| s.p10),
            true,
            |v| format!("{v:.4}"),
        ),
        optional_metric_row(
            "reward_p50",
            a.reward.as_ref().map(|s| s.p50),
            b.reward.as_ref().map(|s| s.p50),
            true,
            |v| format!("{v:.4}"),
        ),
        optional_metric_row(
            "reward_p90",
            a.reward.as_ref().map(|s| s.p90),
            b.reward.as_ref().map(|s| s.p90),
            true,
            |v| format!("{v:.4}"),
        ),
        optional_metric_row(
            "reward_iqr",
            a.reward.as_ref().map(|s| s.p75 - s.p25),
            b.reward.as_ref().map(|s| s.p75 - s.p25),
            false,
            |v| format!("{v:.4}"),
        ),
        optional_metric_row(
            "reward_entropy",
            a.reward.as_ref().map(|s| s.entropy),
            b.reward.as_ref().map(|s| s.entropy),
            false,
            |v| format!("{v:.4}"),
        ),
        optional_metric_row(
            "adv_p50",
            a.advantage.as_ref().map(|s| s.p50),
            b.advantage.as_ref().map(|s| s.p50),
            true,
            |v| format!("{v:.4}"),
        ),
        optional_metric_row(
            "adv_mean",
            a.advantage.as_ref().map(|s| s.mean),
            b.advantage.as_ref().map(|s| s.mean),
            true,
            |v| format!("{v:.4}"),
        ),
        optional_metric_row(
            "adv_iqr",
            a.advantage.as_ref().map(|s| s.p75 - s.p25),
            b.advantage.as_ref().map(|s| s.p75 - s.p25),
            false,
            |v| format!("{v:.4}"),
        ),
        optional_metric_row(
            "adv_std",
            a.advantage.as_ref().map(|s| s.std),
            b.advantage.as_ref().map(|s| s.std),
            false,
            |v| format!("{v:.4}"),
        ),
    ];

    let reward_js = match (&a.reward, &b.reward) {
        (Some(x), Some(y)) => Some(js_divergence(&x.bins, &y.bins)),
        _ => None,
    };
    let adv_js = match (&a.advantage, &b.advantage) {
        (Some(x), Some(y)) => Some(js_divergence(&x.bins, &y.bins)),
        _ => None,
    };

    rows.push(optional_metric_row(
        "reward_js_divergence (lower better)",
        Some(0.0),
        reward_js,
        false,
        |v| format!("{v:.4}"),
    ));
    rows.push(optional_metric_row(
        "adv_js_divergence (lower better)",
        Some(0.0),
        adv_js,
        false,
        |v| format!("{v:.4}"),
    ));

    let mut findings = Vec::new();
    if let Some(js) = reward_js {
        if js > 0.10 {
            findings.push(format!(
                "Reward distribution drift is high (JS={js:.4}); policy behavior likely shifted meaningfully."
            ));
        } else {
            findings.push(format!("Reward distribution drift is moderate/low (JS={js:.4})."));
        }
    } else {
        findings.push("Reward distribution unavailable for one or both runs.".to_string());
    }

    if let Some(js) = adv_js {
        findings.push(format!("Advantage distribution JS divergence: {js:.4}."));
    } else {
        findings.push("Advantage distribution unavailable for one or both runs.".to_string());
    }

    DistributionComparison {
        rows,
        findings,
        reward_bars_a: a
            .reward
            .as_ref()
            .map(|x| bins_to_bar_data(&x.bins))
            .unwrap_or_default(),
        reward_bars_b: b
            .reward
            .as_ref()
            .map(|x| bins_to_bar_data(&x.bins))
            .unwrap_or_default(),
    }
}

pub(super) fn build_health_comparison(a: &HealthData, b: &HealthData) -> HealthComparison {
    let mut rows = Vec::new();
    rows.push(optional_metric_row(
        "log_lines",
        a.log.as_ref().map(|x| x.lines as f64),
        b.log.as_ref().map(|x| x.lines as f64),
        true,
        |v| format!("{v:.0}"),
    ));
    rows.push(optional_metric_row(
        "warnings_per_1k_lines (lower better)",
        a.log.as_ref().map(|x| per_1k(x.warnings as f64, x.lines as f64)),
        b.log.as_ref().map(|x| per_1k(x.warnings as f64, x.lines as f64)),
        false,
        |v| format!("{v:.2}"),
    ));
    rows.push(optional_metric_row(
        "errors_per_1k_lines (lower better)",
        a.log.as_ref().map(|x| per_1k(x.errors as f64, x.lines as f64)),
        b.log.as_ref().map(|x| per_1k(x.errors as f64, x.lines as f64)),
        false,
        |v| format!("{v:.2}"),
    ));
    rows.push(optional_metric_row(
        "lag_events_per_1k_lines (lower better)",
        a.log
            .as_ref()
            .map(|x| per_1k(x.lag_events as f64, x.lines as f64)),
        b.log
            .as_ref()
            .map(|x| per_1k(x.lag_events as f64, x.lines as f64)),
        false,
        |v| format!("{v:.2}"),
    ));
    rows.push(optional_metric_row(
        "checkpoints_total",
        a.checkpoints.as_ref().map(|x| x.total as f64),
        b.checkpoints.as_ref().map(|x| x.total as f64),
        true,
        |v| format!("{v:.0}"),
    ));
    rows.push(optional_metric_row(
        "checkpoint_ready_rate",
        a.checkpoints.as_ref().map(checkpoint_ready_rate),
        b.checkpoints.as_ref().map(checkpoint_ready_rate),
        true,
        |v| format!("{v:.3}"),
    ));
    rows.push(optional_metric_row(
        "checkpoint_failed_count (lower better)",
        a.checkpoints.as_ref().map(|x| x.failed as f64),
        b.checkpoints.as_ref().map(|x| x.failed as f64),
        false,
        |v| format!("{v:.0}"),
    ));

    let mut findings = Vec::new();
    if let (Some(al), Some(bl)) = (&a.log, &b.log) {
        let err_delta =
            per_1k(bl.errors as f64, bl.lines as f64) - per_1k(al.errors as f64, al.lines as f64);
        if err_delta.abs() >= 0.5 {
            findings.push(format!(
                "Error density changed by {:+.2} per 1k log lines.",
                err_delta
            ));
        }

        let lag_delta = per_1k(bl.lag_events as f64, bl.lines as f64)
            - per_1k(al.lag_events as f64, al.lines as f64);
        if lag_delta.abs() >= 0.5 {
            findings.push(format!(
                "Event-loop lag density changed by {:+.2} per 1k lines.",
                lag_delta
            ));
        }
    }

    if let (Some(ac), Some(bc)) = (&a.checkpoints, &b.checkpoints) {
        let ready_delta = checkpoint_ready_rate(bc) - checkpoint_ready_rate(ac);
        if ready_delta.abs() >= 0.05 {
            findings.push(format!(
                "Checkpoint readiness rate changed by {:+.3}.",
                ready_delta
            ));
        }
    }

    if findings.is_empty() {
        findings.push("No strong health deltas under default heuristics.".to_string());
    }

    HealthComparison { rows, findings }
}

fn per_1k(count: f64, denom: f64) -> f64 {
    if denom <= 0.0 {
        0.0
    } else {
        1000.0 * count / denom
    }
}

fn checkpoint_ready_rate(c: &CheckpointHealth) -> f64 {
    if c.total == 0 {
        0.0
    } else {
        c.ready as f64 / c.total as f64
    }
}

pub(super) fn parse_log_health(text: &str) -> LogHealth {
    let mut lines = 0usize;
    let mut warnings = 0usize;
    let mut errors = 0usize;
    let mut lag_events = 0usize;

    for raw in text.lines() {
        if raw.trim().is_empty() {
            continue;
        }
        lines += 1;

        let line = raw.to_ascii_lowercase();
        if line.contains("warning") || line.contains(" warn ") || line.starts_with("warn") {
            warnings += 1;
        }
        if line.contains("error")
            || line.contains("exception")
            || line.contains("traceback")
            || line.contains("failed")
        {
            errors += 1;
        }
        if line.contains("busy event loop")
            || line.contains("event loop lag")
            || line.contains("lag over the last")
        {
            lag_events += 1;
        }
    }

    LogHealth {
        lines,
        warnings,
        errors,
        lag_events,
    }
}

pub(super) fn parse_checkpoint_health(root: &Value) -> Option<CheckpointHealth> {
    let items = if let Some(arr) = root.get("checkpoints").and_then(Value::as_array) {
        arr
    } else if let Some(arr) = root.as_array() {
        arr
    } else {
        return None;
    };

    let mut checkpoints = CheckpointHealth {
        total: items.len(),
        ready: 0,
        failed: 0,
        pending: 0,
        uploading: 0,
        other: 0,
    };

    for item in items {
        let status = item
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_ascii_uppercase();
        match status.as_str() {
            "READY" => checkpoints.ready += 1,
            "FAILED" => checkpoints.failed += 1,
            "PENDING" => checkpoints.pending += 1,
            "UPLOADING" => checkpoints.uploading += 1,
            _ => checkpoints.other += 1,
        }
    }

    Some(checkpoints)
}

#[derive(Debug, Clone)]
struct HistogramCandidate {
    kind_hint: String,
    step: Option<i64>,
    bins: Vec<HistogramBin>,
}

pub(super) fn parse_distribution_bundle(root: &Value) -> DistributionBundle {
    let mut candidates = Vec::new();
    collect_histogram_candidates(root, "root", None, &mut candidates);
    let step = detect_step(root);

    if candidates.is_empty() {
        return DistributionBundle::default();
    }

    let reward_candidate = pick_distribution_candidate(&candidates, &["reward", "rewards"])
        .or_else(|| pick_distribution_candidate(&candidates, &[]));
    let advantage_candidate =
        pick_distribution_candidate(&candidates, &["advantage", "advantages", "adv"]);

    DistributionBundle {
        step,
        reward: reward_candidate.and_then(|c| distribution_stats_from_bins(&c.bins)),
        advantage: advantage_candidate.and_then(|c| distribution_stats_from_bins(&c.bins)),
    }
}

pub(super) fn bins_to_bar_data(bins: &[HistogramBin]) -> Vec<(String, u64)> {
    bins.iter()
        .map(|b| (format!("{:.3}-{:.3}", b.lower, b.upper), b.count.round().max(0.0) as u64))
        .collect::<Vec<_>>()
}

fn pick_distribution_candidate<'a>(
    candidates: &'a [HistogramCandidate],
    keywords: &[&str],
) -> Option<&'a HistogramCandidate> {
    let mut filtered = candidates
        .iter()
        .filter(|candidate| {
            if keywords.is_empty() {
                return true;
            }
            let hint = candidate.kind_hint.to_ascii_lowercase();
            keywords.iter().any(|keyword| hint.contains(keyword))
        })
        .collect::<Vec<_>>();

    if filtered.is_empty() {
        return None;
    }

    filtered.sort_by(|a, b| {
        let step_a = a.step.unwrap_or(-1);
        let step_b = b.step.unwrap_or(-1);
        step_b
            .cmp(&step_a)
            .then_with(|| b.bins.len().cmp(&a.bins.len()))
            .then_with(|| b.kind_hint.len().cmp(&a.kind_hint.len()))
    });

    filtered.first().copied()
}

fn collect_histogram_candidates(
    root: &Value,
    root_hint: &str,
    inherited_step: Option<i64>,
    out: &mut Vec<HistogramCandidate>,
) {
    let mut stack = Vec::with_capacity(64);
    stack.push((root_hint.to_string(), inherited_step, root, 0usize));

    let mut visited = 0usize;
    while let Some((hint, inherited_step, value, depth)) = stack.pop() {
        if visited >= MAX_HISTOGRAM_NODES {
            break;
        }
        visited += 1;

        match value {
            Value::Object(map) => {
                let step = detect_step(value).or(inherited_step);
                for key in [
                    "bins",
                    "histogram",
                    "distribution",
                    "bucket_counts",
                    "data",
                    "values",
                ] {
                    if let Some(arr) = map.get(key).and_then(Value::as_array)
                        && let Some(bins) = parse_bins_array(arr)
                    {
                        out.push(HistogramCandidate {
                            kind_hint: format!("{hint}.{key}"),
                            step,
                            bins,
                        });
                    }
                }

                if depth + 1 > MAX_HISTOGRAM_DEPTH {
                    continue;
                }

                for (key, child) in map.iter().rev() {
                    stack.push((format!("{hint}.{key}"), step, child, depth + 1));
                }
            }
            Value::Array(items) => {
                if let Some(bins) = parse_bins_array(items) {
                    out.push(HistogramCandidate {
                        kind_hint: hint.clone(),
                        step: inherited_step,
                        bins,
                    });
                }

                if depth + 1 > MAX_HISTOGRAM_DEPTH {
                    continue;
                }

                for (index, child) in items.iter().enumerate().rev() {
                    stack.push((format!("{hint}[{index}]"), inherited_step, child, depth + 1));
                }
            }
            _ => {}
        }
    }
}

fn parse_bins_array(arr: &[Value]) -> Option<Vec<HistogramBin>> {
    if arr.len() < 2 {
        return None;
    }

    let mut bins = Vec::new();
    for item in arr {
        let Some(obj) = item.as_object() else {
            continue;
        };

        let (lower, upper) = if let Some(bin_str) = obj.get("bin").and_then(Value::as_str) {
            parse_bin_range(bin_str)?
        } else {
            let lower = get_obj_f64(
                obj,
                &["lower", "low", "min", "start", "left", "bin_start", "bucket_start", "x0"],
            )
            .or_else(|| get_obj_f64(obj, &["x"]))?;

            let upper = get_obj_f64(
                obj,
                &["upper", "high", "max", "end", "right", "bin_end", "bucket_end", "x1"],
            )
            .or(Some(lower + 1.0))?;

            (lower, upper)
        };

        let count = get_obj_f64(
            obj,
            &["count", "n", "freq", "frequency", "value", "y", "mass"],
        )?;

        let (lower, upper) = if upper >= lower { (lower, upper) } else { (upper, lower) };
        if count.is_finite() && lower.is_finite() && upper.is_finite() && count >= 0.0 {
            bins.push(HistogramBin { lower, upper, count });
        }
    }

    if bins.len() < 2 {
        None
    } else {
        Some(bins)
    }
}

fn parse_bin_range(s: &str) -> Option<(f64, f64)> {
    let s = s.trim();
    if s.len() < 3 {
        return None;
    }

    let chars = s.char_indices().collect::<Vec<_>>();
    for idx in 1..chars.len().saturating_sub(1) {
        let (byte_index, ch) = chars[idx];
        if ch != '-' {
            continue;
        }

        let prev = chars[idx - 1].1;
        let next = chars[idx + 1].1;
        let prev_ok = prev.is_ascii_digit() || prev == '.';
        let next_ok = next.is_ascii_digit() || next == '.' || next == '-';
        if !prev_ok || !next_ok {
            continue;
        }

        let left = s[..byte_index].trim().parse::<f64>().ok()?;
        let right = s[byte_index + 1..].trim().parse::<f64>().ok()?;
        return Some((left, right));
    }

    None
}

fn get_obj_f64(obj: &Map<String, Value>, keys: &[&str]) -> Option<f64> {
    for key in keys {
        if let Some(value) = obj.get(*key).and_then(to_f64) {
            return Some(value);
        }
    }
    None
}

fn distribution_stats_from_bins(bins: &[HistogramBin]) -> Option<DistributionStats> {
    if bins.is_empty() {
        return None;
    }

    let mut bins = bins.to_vec();
    bins.sort_by(|a, b| {
        a.lower
            .partial_cmp(&b.lower)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.upper.partial_cmp(&b.upper).unwrap_or(Ordering::Equal))
    });

    let total_count = bins.iter().map(|b| b.count).sum::<f64>();
    if total_count <= 0.0 {
        return None;
    }

    let mean = bins
        .iter()
        .map(|b| ((b.lower + b.upper) * 0.5) * b.count)
        .sum::<f64>()
        / total_count;
    let variance = bins
        .iter()
        .map(|b| {
            let midpoint = (b.lower + b.upper) * 0.5;
            (midpoint - mean).powi(2) * b.count
        })
        .sum::<f64>()
        / total_count;

    let mut entropy = 0.0;
    for bin in &bins {
        let p = bin.count / total_count;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }

    Some(DistributionStats {
        p10: histogram_quantile(&bins, 0.10, total_count),
        p25: histogram_quantile(&bins, 0.25, total_count),
        p50: histogram_quantile(&bins, 0.50, total_count),
        p75: histogram_quantile(&bins, 0.75, total_count),
        p90: histogram_quantile(&bins, 0.90, total_count),
        bins,
        total_count,
        mean,
        std: variance.sqrt(),
        entropy,
    })
}

fn histogram_quantile(bins: &[HistogramBin], q: f64, total_count: f64) -> f64 {
    if bins.is_empty() {
        return 0.0;
    }

    let target = total_count * q.clamp(0.0, 1.0);
    let mut accumulated = 0.0;
    for bin in bins {
        let next = accumulated + bin.count;
        if next >= target && bin.count > 0.0 {
            let ratio = ((target - accumulated) / bin.count).clamp(0.0, 1.0);
            return bin.lower + ratio * (bin.upper - bin.lower);
        }
        accumulated = next;
    }

    bins.last().map(|bin| bin.upper).unwrap_or(0.0)
}

fn js_divergence(a_bins: &[HistogramBin], b_bins: &[HistogramBin]) -> f64 {
    let min_x = a_bins
        .iter()
        .chain(b_bins.iter())
        .map(|bin| bin.lower)
        .fold(f64::INFINITY, f64::min);
    let max_x = a_bins
        .iter()
        .chain(b_bins.iter())
        .map(|bin| bin.upper)
        .fold(f64::NEG_INFINITY, f64::max);
    if !min_x.is_finite() || !max_x.is_finite() || max_x <= min_x {
        return 0.0;
    }

    let bins = 64usize;
    let p = project_bins_to_grid(a_bins, min_x, max_x, bins);
    let q = project_bins_to_grid(b_bins, min_x, max_x, bins);
    let mut midpoint = vec![0.0; bins];
    for idx in 0..bins {
        midpoint[idx] = 0.5 * (p[idx] + q[idx]);
    }

    0.5 * kl_divergence(&p, &midpoint) + 0.5 * kl_divergence(&q, &midpoint)
}

fn project_bins_to_grid(bins: &[HistogramBin], min_x: f64, max_x: f64, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    if n == 0 || max_x <= min_x {
        return out;
    }

    let total_count = bins.iter().map(|bin| bin.count).sum::<f64>();
    if total_count <= 0.0 {
        return out;
    }

    let width = (max_x - min_x) / n as f64;
    for bin in bins {
        if bin.count <= 0.0 {
            continue;
        }

        let lower = bin.lower.max(min_x);
        let upper = bin.upper.min(max_x);
        if upper <= lower {
            continue;
        }

        let mass = bin.count / total_count;
        let bin_width = (bin.upper - bin.lower).max(1e-9);
        let start_idx = ((lower - min_x) / width).floor().max(0.0) as usize;
        let end_idx = (((upper - min_x) / width).ceil() as usize).min(n);

        for (idx, cell) in out.iter_mut().enumerate().take(end_idx).skip(start_idx) {
            let cell_lower = min_x + (idx as f64) * width;
            let cell_upper = cell_lower + width;
            let overlap = (upper.min(cell_upper) - lower.max(cell_lower)).max(0.0);
            if overlap > 0.0 {
                *cell += mass * (overlap / bin_width);
            }
        }
    }

    let sum = out.iter().sum::<f64>();
    if sum > 0.0 {
        for cell in &mut out {
            *cell /= sum;
        }
    }

    out
}

fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    let eps = 1e-12;
    p.iter()
        .zip(q.iter())
        .map(|(pi, qi)| {
            let p = (*pi).max(eps);
            let q = (*qi).max(eps);
            p * (p / q).ln()
        })
        .sum::<f64>()
}

fn metric_row<F>(
    metric: &str,
    a: f64,
    b: f64,
    higher_is_better: bool,
    format_fn: F,
) -> CompareRowData
where
    F: Fn(f64) -> String,
{
    let delta = b - a;
    let kind = if delta.abs() < 1e-9 {
        DeltaKind::Neutral
    } else if higher_is_better {
        if delta > 0.0 {
            DeltaKind::Better
        } else {
            DeltaKind::Worse
        }
    } else if delta < 0.0 {
        DeltaKind::Better
    } else {
        DeltaKind::Worse
    };

    CompareRowData {
        metric: metric.to_string(),
        a: format_fn(a),
        b: format_fn(b),
        delta: format_signed_f64(delta),
        kind,
    }
}

fn optional_metric_row<F>(
    metric: &str,
    a: Option<f64>,
    b: Option<f64>,
    higher_is_better: bool,
    format_fn: F,
) -> CompareRowData
where
    F: Fn(f64) -> String,
{
    match (a, b) {
        (Some(a), Some(b)) => metric_row(metric, a, b, higher_is_better, format_fn),
        (Some(a), None) => CompareRowData {
            metric: metric.to_string(),
            a: format_fn(a),
            b: "n/a".to_string(),
            delta: "B missing".to_string(),
            kind: DeltaKind::Neutral,
        },
        (None, Some(b)) => CompareRowData {
            metric: metric.to_string(),
            a: "n/a".to_string(),
            b: format_fn(b),
            delta: "A missing".to_string(),
            kind: DeltaKind::Neutral,
        },
        (None, None) => CompareRowData {
            metric: metric.to_string(),
            a: "n/a".to_string(),
            b: "n/a".to_string(),
            delta: "missing".to_string(),
            kind: DeltaKind::Neutral,
        },
    }
}

fn format_signed_f64(v: f64) -> String {
    if v.is_nan() {
        "nan".to_string()
    } else {
        format!("{:+.4}", v)
    }
}

fn format_signed_i64(v: i64) -> String {
    format!("{:+}", v)
}

pub(super) fn to_chart_points(points: &[MetricPoint]) -> Vec<(f64, f64)> {
    points
        .iter()
        .map(|point| (point.step as f64, point.reward))
        .collect::<Vec<_>>()
}
