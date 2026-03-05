use std::cmp::Ordering;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand};
use crossterm::event::{self, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::prelude::*;
use ratatui::symbols;
use ratatui::widgets::{
    Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph, Row, Table, TableState, Wrap,
};
use serde_json::Value;
use throbber_widgets_tui::{BRAILLE_SIX, Throbber, ThrobberState, WhichUse};


#[path = "ui.rs"]
mod ui;

pub(crate) fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(CommandSet::Compare(args)) => run_compare(args),
        None => run_picker_mode(cli.prefetch_runs),
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "pale",
    version,
    about = "Rule-based RL run autopsy and comparison TUI"
)]
struct Cli {
    /// Number of recent runs to prefetch in no-arg picker mode.
    #[arg(long, default_value_t = 30)]
    prefetch_runs: usize,

    #[command(subcommand)]
    command: Option<CommandSet>,
}

#[derive(Debug, Subcommand)]
enum CommandSet {
    /// Compare two RL runs and open the TUI.
    Compare(CompareArgs),
}

#[derive(Debug, Parser)]
struct CompareArgs {
    /// Prime run id for side A.
    #[arg(long)]
    run_a: Option<String>,

    /// Prime run id for side B.
    #[arg(long)]
    run_b: Option<String>,

    /// Path to metrics JSON for side A.
    #[arg(long)]
    metrics_a: Option<PathBuf>,

    /// Path to metrics JSON for side B.
    #[arg(long)]
    metrics_b: Option<PathBuf>,

    /// Optional label for side A in the UI.
    #[arg(long)]
    label_a: Option<String>,

    /// Optional label for side B in the UI.
    #[arg(long)]
    label_b: Option<String>,

    /// Last-N window size for local stats.
    #[arg(long, default_value_t = 50)]
    window: usize,

    /// Reward thresholds to compare time-to-threshold.
    #[arg(long = "threshold")]
    thresholds: Vec<f64>,

    /// Distribution step to fetch from Prime (defaults to latest step).
    #[arg(long)]
    dist_step: Option<i64>,

    /// Log tail size for health analysis.
    #[arg(long, default_value_t = 2000)]
    log_tail: usize,

    /// Skip extra fetches (distributions/logs/checkpoints).
    #[arg(long, default_value_t = false)]
    skip_extras: bool,
}

#[derive(Debug, Clone)]
struct MetricPoint {
    step: i64,
    reward: f64,
}

#[derive(Debug, Clone)]
struct SegmentSummary {
    mean: f64,
    slope: f64,
}

#[derive(Debug, Clone)]
struct Summary {
    steps: usize,
    final_reward: f64,
    best_reward: f64,
    best_step: i64,
    auc_reward: f64,
    last_mean: f64,
    last_std: f64,
    last_slope: f64,
    early: SegmentSummary,
    mid: SegmentSummary,
    late: SegmentSummary,
    time_to_threshold: Vec<(f64, Option<i64>)>,
}

#[derive(Debug, Clone)]
struct RunData {
    label: String,
    run_id: Option<String>,
    points: Vec<MetricPoint>,
    summary: Summary,
}

#[derive(Debug, Clone, Copy)]
enum DeltaKind {
    Better,
    Worse,
    Neutral,
}

#[derive(Debug, Clone)]
struct CompareRowData {
    metric: String,
    a: String,
    b: String,
    delta: String,
    kind: DeltaKind,
}

#[derive(Debug, Clone)]
struct Comparison {
    rows: Vec<CompareRowData>,
    findings: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tab {
    Summary,
    Distributions,
    Health,
}

impl Tab {
    fn title(self) -> &'static str {
        match self {
            Tab::Summary => "Summary",
            Tab::Distributions => "Distributions",
            Tab::Health => "Health",
        }
    }

    fn next(self) -> Self {
        match self {
            Tab::Summary => Tab::Distributions,
            Tab::Distributions => Tab::Health,
            Tab::Health => Tab::Summary,
        }
    }

    fn prev(self) -> Self {
        match self {
            Tab::Summary => Tab::Health,
            Tab::Distributions => Tab::Summary,
            Tab::Health => Tab::Distributions,
        }
    }
}

#[derive(Debug, Clone)]
struct HistogramBin {
    lower: f64,
    upper: f64,
    count: f64,
}

#[derive(Debug, Clone)]
struct DistributionStats {
    bins: Vec<HistogramBin>,
    total_count: f64,
    mean: f64,
    std: f64,
    p10: f64,
    p25: f64,
    p50: f64,
    p75: f64,
    p90: f64,
    entropy: f64,
}

#[derive(Debug, Clone, Default)]
struct DistributionBundle {
    step: Option<i64>,
    reward: Option<DistributionStats>,
    advantage: Option<DistributionStats>,
}

#[derive(Debug, Clone)]
struct DistributionComparison {
    rows: Vec<CompareRowData>,
    findings: Vec<String>,
    reward_bars_a: Vec<(String, u64)>,
    reward_bars_b: Vec<(String, u64)>,
}

#[derive(Debug, Clone)]
struct LogHealth {
    lines: usize,
    warnings: usize,
    errors: usize,
    lag_events: usize,
}

#[derive(Debug, Clone)]
struct CheckpointHealth {
    total: usize,
    ready: usize,
    failed: usize,
    pending: usize,
    uploading: usize,
    other: usize,
}

#[derive(Debug, Clone, Default)]
struct HealthData {
    log: Option<LogHealth>,
    checkpoints: Option<CheckpointHealth>,
}

#[derive(Debug, Clone)]
struct HealthComparison {
    rows: Vec<CompareRowData>,
    findings: Vec<String>,
}

#[derive(Debug, Clone, Default)]
struct SideExtras {
    distributions: DistributionBundle,
    dist_steps: Vec<i64>,
    health: HealthData,
}

struct App {
    left: RunData,
    right: RunData,
    summary_comparison: Comparison,
    distribution_comparison: DistributionComparison,
    health_comparison: HealthComparison,
    reward_chart_left: Vec<(f64, f64)>,
    reward_chart_right: Vec<(f64, f64)>,
    summary_state: TableState,
    distributions_state: TableState,
    health_state: TableState,
    active_tab: Tab,
    dist_steps: Vec<i64>,
    current_dist_index: usize,
    dist_loading: bool,
    dist_error: Option<String>,
    dist_rx: Option<mpsc::Receiver<Result<DistributionFetchResult, String>>>,
    dist_throbber: ThrobberState,
}

#[derive(Debug, Clone)]
struct DistributionFetchResult {
    step: i64,
    left: DistributionBundle,
    right: DistributionBundle,
}

#[derive(Debug, Clone)]
struct RunListItem {
    id: String,
    name: String,
    status: String,
    model: String,
    updated_at: String,
}

struct PickerApp {
    runs: Vec<RunListItem>,
    selected: usize,
    selected_a: Option<usize>,
    selected_b: Option<usize>,
    page: usize,
    pending_page: usize,
    per_page: usize,
    loading: bool,
    error: Option<String>,
    throbber: ThrobberState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppLoopAction {
    Quit,
    BackToPicker,
}

#[derive(Debug, Clone)]
struct PickerFetchResult {
    page: usize,
    runs: Vec<RunListItem>,
}

fn run_picker_mode(prefetch_runs: usize) -> Result<()> {
    enable_raw_mode().context("failed to enable raw mode")?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen).context("failed to enter alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("failed to create terminal")?;

    let res = run_picker_session(&mut terminal, prefetch_runs);

    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
    terminal.show_cursor().ok();

    res
}

fn run_picker_session(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    prefetch_runs: usize,
) -> Result<()> {
    let mut app = PickerApp {
        runs: Vec::new(),
        selected: 0,
        selected_a: None,
        selected_b: None,
        page: 1,
        pending_page: 1,
        per_page: prefetch_runs.max(1),
        loading: true,
        error: None,
        throbber: ThrobberState::default(),
    };

    let mut rx = start_picker_fetch(app.pending_page, app.per_page);

    loop {
        if app.loading {
            app.throbber.calc_next();
            match rx.try_recv() {
                Ok(Ok(payload)) => {
                    app.loading = false;
                    app.error = None;
                    if payload.runs.is_empty() && payload.page > 1 {
                        app.pending_page = app.page;
                        app.error = Some("No runs on that page.".to_string());
                    } else {
                        app.runs = payload.runs;
                        app.page = payload.page;
                        app.pending_page = payload.page;
                        app.selected = 0;
                        app.selected_a = None;
                        app.selected_b = None;
                    }
                }
                Ok(Err(err)) => {
                    app.error = Some(err);
                    app.loading = false;
                    app.pending_page = app.page;
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    app.error = Some("background fetch channel disconnected".to_string());
                    app.loading = false;
                    app.pending_page = app.page;
                }
            }
        }

        terminal.draw(|f| ui::draw_picker_ui(f, &app))?;

        if event::poll(Duration::from_millis(150))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => return Ok(()),
                    KeyCode::Down | KeyCode::Char('j') => {
                        if !app.loading && !app.runs.is_empty() {
                            app.selected = (app.selected + 1).min(app.runs.len() - 1);
                        }
                    }
                    KeyCode::Up | KeyCode::Char('k') => {
                        if !app.loading && !app.runs.is_empty() {
                            app.selected = app.selected.saturating_sub(1);
                        }
                    }
                    KeyCode::Char('a') => {
                        if !app.loading && !app.runs.is_empty() {
                            app.selected_a = Some(app.selected);
                        }
                    }
                    KeyCode::Char('b') => {
                        if !app.loading && !app.runs.is_empty() {
                            app.selected_b = Some(app.selected);
                        }
                    }
                    KeyCode::Right | KeyCode::Char(']') => {
                        if !app.loading {
                            app.pending_page = app.page + 1;
                            app.loading = true;
                            app.error = None;
                            rx = start_picker_fetch(app.pending_page, app.per_page);
                        }
                    }
                    KeyCode::Left | KeyCode::Char('[') => {
                        if !app.loading && app.page > 1 {
                            app.pending_page = app.page - 1;
                            app.loading = true;
                            app.error = None;
                            rx = start_picker_fetch(app.pending_page, app.per_page);
                        }
                    }
                    KeyCode::Enter => {
                        if app.loading {
                            continue;
                        }
                        if let (Some(a_idx), Some(b_idx)) = (app.selected_a, app.selected_b) {
                            if a_idx == b_idx {
                                continue;
                            }

                            let run_a = app.runs[a_idx].id.clone();
                            let run_b = app.runs[b_idx].id.clone();
                            let args = CompareArgs {
                                run_a: Some(run_a),
                                run_b: Some(run_b),
                                metrics_a: None,
                                metrics_b: None,
                                label_a: None,
                                label_b: None,
                                window: 50,
                                thresholds: Vec::new(),
                                dist_step: None,
                                log_tail: 2000,
                                skip_extras: false,
                            };

                            match load_compare_app_with_spinner(
                                terminal,
                                "Loading compare view...",
                                args,
                            ) {
                                Ok(mut compare_app) => match run_app(terminal, &mut compare_app)? {
                                    AppLoopAction::Quit => return Ok(()),
                                    AppLoopAction::BackToPicker => {}
                                },
                                Err(err) => {
                                    app.error = Some(format!("failed to build compare view: {err}"));
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

fn start_picker_fetch(page: usize, per_page: usize) -> mpsc::Receiver<Result<PickerFetchResult, String>> {
    let (tx, rx) = mpsc::channel::<Result<PickerFetchResult, String>>();
    thread::spawn(move || {
        let result = fetch_runs_page(per_page, page)
            .map(|runs| PickerFetchResult { page, runs })
            .map_err(|e| e.to_string());
        let _ = tx.send(result);
    });
    rx
}

fn load_compare_app_with_spinner(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    label: &str,
    args: CompareArgs,
) -> Result<App> {
    let (tx, rx) = mpsc::channel::<Result<App, String>>();
    thread::spawn(move || {
        let res = build_compare_app(args).map_err(|e| e.to_string());
        let _ = tx.send(res);
    });

    let mut throbber = ThrobberState::default();
    loop {
        throbber.calc_next();
        terminal.draw(|f| ui::draw_loading_ui(f, label, &mut throbber))?;

        match rx.try_recv() {
            Ok(Ok(app)) => return Ok(app),
            Ok(Err(err)) => return Err(anyhow!(err)),
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(anyhow!("background compare loader disconnected"));
            }
        }

        thread::sleep(Duration::from_millis(80));
    }
}


fn fetch_runs_page(limit: usize, page: usize) -> Result<Vec<RunListItem>> {
    let limit_s = limit.to_string();
    let page_s = page.to_string();
    let root = run_prime_json(&[
        "rl",
        "list",
        "-o",
        "json",
        "-n",
        limit_s.as_str(),
        "-p",
        page_s.as_str(),
    ])?;
    let runs = root
        .get("runs")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("missing `runs` array in `prime rl list` output"))?;

    let out = runs
        .iter()
        .map(|r| RunListItem {
            id: r
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string(),
            name: r
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string(),
            status: r
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string(),
            model: r
                .get("base_model")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string(),
            updated_at: r
                .get("updated_at")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string(),
        })
        .collect::<Vec<_>>();

    Ok(out)
}


fn run_compare(args: CompareArgs) -> Result<()> {
    enable_raw_mode().context("failed to enable raw mode")?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen).context("failed to enter alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("failed to create terminal")?;

    let res = (|| -> Result<()> {
        let mut app = load_compare_app_with_spinner(&mut terminal, "Loading compare view...", args)?;
        match run_app(&mut terminal, &mut app)? {
            AppLoopAction::Quit => Ok(()),
            AppLoopAction::BackToPicker => run_picker_session(&mut terminal, 30),
        }
    })();

    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
    terminal.show_cursor().ok();

    res
}

fn build_compare_app(args: CompareArgs) -> Result<App> {
    validate_sources(&args)?;

    let thresholds = if args.thresholds.is_empty() {
        vec![0.5, 0.7]
    } else {
        args.thresholds.clone()
    };

    let run_a = args.run_a.clone();
    let run_b = args.run_b.clone();
    let metrics_a = args.metrics_a.clone();
    let metrics_b = args.metrics_b.clone();
    let label_a = args.label_a.clone();
    let label_b = args.label_b.clone();
    let thresholds_a = thresholds.clone();
    let thresholds_b = thresholds.clone();
    let window = args.window;

    let left_handle = thread::spawn(move || {
        load_side(
            run_a.as_ref(),
            metrics_a.as_ref(),
            label_a.as_ref(),
            window,
            &thresholds_a,
            'A',
        )
    });
    let right_handle = thread::spawn(move || {
        load_side(
            run_b.as_ref(),
            metrics_b.as_ref(),
            label_b.as_ref(),
            window,
            &thresholds_b,
            'B',
        )
    });

    let left = left_handle
        .join()
        .map_err(|_| anyhow::anyhow!("left-side loader thread panicked"))??;
    let right = right_handle
        .join()
        .map_err(|_| anyhow::anyhow!("right-side loader thread panicked"))??;

    let summary_comparison = compare_runs(&left, &right, &thresholds);

    let left_clone = left.clone();
    let right_clone = right.clone();
    let log_tail = args.log_tail;
    let dist_step = args.dist_step;
    let skip_extras = args.skip_extras;

    let extras_left_handle = thread::spawn(move || {
        collect_side_extras(&left_clone, log_tail, dist_step, skip_extras)
    });
    let extras_right_handle = thread::spawn(move || {
        collect_side_extras(&right_clone, log_tail, dist_step, skip_extras)
    });

    let extras_left = extras_left_handle
        .join()
        .map_err(|_| anyhow::anyhow!("left-side extras thread panicked"))?;
    let extras_right = extras_right_handle
        .join()
        .map_err(|_| anyhow::anyhow!("right-side extras thread panicked"))?;

    let distribution_comparison =
        build_distribution_comparison(&extras_left.distributions, &extras_right.distributions);
    let health_comparison = build_health_comparison(&extras_left.health, &extras_right.health);

    let mut summary_state = TableState::default();
    summary_state.select(Some(0));

    let mut distributions_state = TableState::default();
    distributions_state.select(Some(0));

    let mut health_state = TableState::default();
    health_state.select(Some(0));

    let dist_steps = merge_distribution_steps(&extras_left.dist_steps, &extras_right.dist_steps);
    let default_dist_index = dist_steps.len().saturating_sub(1);

    let mut app = App {
        reward_chart_left: to_chart_points(&left.points),
        reward_chart_right: to_chart_points(&right.points),
        left,
        right,
        summary_comparison,
        distribution_comparison,
        health_comparison,
        summary_state,
        distributions_state,
        health_state,
        active_tab: Tab::Summary,
        dist_steps,
        current_dist_index: default_dist_index,
        dist_loading: false,
        dist_error: None,
        dist_rx: None,
        dist_throbber: ThrobberState::default(),
    };

    if let Some(step) = args.dist_step {
        if let Some(idx) = app.dist_steps.iter().position(|s| *s == step) {
            app.current_dist_index = idx;
        } else {
            app.dist_steps.push(step);
            app.dist_steps.sort_unstable();
            app.dist_steps.dedup();
            if let Some(idx) = app.dist_steps.iter().position(|s| *s == step) {
                app.current_dist_index = idx;
            }
        }
    } else if let Some(step) = app
        .distribution_comparison
        .rows
        .iter()
        .find(|r| r.metric == "distribution_step")
        .and_then(|r| r.a.parse::<usize>().ok().or_else(|| r.b.parse::<usize>().ok()))
    {
        if let Some(idx) = app.dist_steps.iter().position(|s| *s as usize == step) {
            app.current_dist_index = idx;
        }
    }

    Ok(app)
}

fn validate_sources(args: &CompareArgs) -> Result<()> {
    if args.run_a.is_none() && args.metrics_a.is_none() {
        bail!("Provide one of --run-a or --metrics-a");
    }
    if args.run_b.is_none() && args.metrics_b.is_none() {
        bail!("Provide one of --run-b or --metrics-b");
    }
    if args.run_a.is_some() && args.metrics_a.is_some() {
        bail!("Provide only one source for side A: --run-a OR --metrics-a");
    }
    if args.run_b.is_some() && args.metrics_b.is_some() {
        bail!("Provide only one source for side B: --run-b OR --metrics-b");
    }
    Ok(())
}

fn load_side(
    run_id: Option<&String>,
    metrics_path: Option<&PathBuf>,
    label_override: Option<&String>,
    window: usize,
    thresholds: &[f64],
    side: char,
) -> Result<RunData> {
    let (label, raw_json, run_id) = match (run_id, metrics_path) {
        (Some(id), None) => {
            let raw = fetch_prime_metrics(id)
                .with_context(|| format!("failed to fetch metrics for run {id}"))?;
            let label = label_override
                .cloned()
                .unwrap_or_else(|| format!("run:{id}"));
            (label, raw, Some(id.clone()))
        }
        (None, Some(path)) => {
            let raw = read_json_file(path)
                .with_context(|| format!("failed to read metrics from {}", path.display()))?;
            let default_label = path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| format!("side-{side}"));
            let label = label_override.cloned().unwrap_or(default_label);
            (label, raw, None)
        }
        _ => bail!("invalid source combination on side {side}"),
    };

    let (mut points, _reward_key) = parse_metric_points(&raw_json)?;
    if points.is_empty() {
        bail!("no metric points found for side {side}");
    }

    points.sort_by(|a, b| a.step.cmp(&b.step));
    points.dedup_by(|a, b| a.step == b.step);

    let summary = summarize_points(&points, window, thresholds);

    Ok(RunData {
        label,
        run_id,
        points,
        summary,
    })
}

fn collect_side_extras(
    run: &RunData,
    log_tail: usize,
    dist_step: Option<i64>,
    skip_extras: bool,
) -> SideExtras {
    if skip_extras {
        return SideExtras {
            ..SideExtras::default()
        };
    }

    let Some(run_id) = run.run_id.as_ref() else {
        return SideExtras {
            ..SideExtras::default()
        };
    };
    let dist_steps = match fetch_prime_progress(run_id) {
        Ok(json) => parse_distribution_steps(&json),
        Err(_) => Vec::new(),
    };

    let distributions = match fetch_prime_distributions(run_id, dist_step) {
        Ok(json) => {
            let bundle = parse_distribution_bundle(&json);
            bundle
        }
        Err(_) => DistributionBundle::default(),
    };

    let log = match fetch_prime_logs(run_id, log_tail) {
        Ok(text) => Some(parse_log_health(&text)),
        Err(_) => None,
    };

    let checkpoints = match fetch_prime_checkpoints(run_id) {
        Ok(json) => parse_checkpoint_health(&json),
        Err(_) => None,
    };

    SideExtras {
        distributions,
        dist_steps,
        health: HealthData { log, checkpoints },
    }
}

fn fetch_prime_metrics(run_id: &str) -> Result<Value> {
    run_prime_json(&["rl", "metrics", run_id])
        .with_context(|| format!("`prime rl metrics {run_id}` failed"))
}

fn fetch_prime_distributions(run_id: &str, step: Option<i64>) -> Result<Value> {
    match step {
        Some(s) => {
            let s_arg = s.to_string();
            run_prime_json(&["rl", "distributions", run_id, "--step", s_arg.as_str()])
                .with_context(|| format!("`prime rl distributions {run_id} --step {s}` failed"))
        }
        None => run_prime_json(&["rl", "distributions", run_id])
            .with_context(|| format!("`prime rl distributions {run_id}` failed")),
    }
}

fn fetch_prime_progress(run_id: &str) -> Result<Value> {
    run_prime_json(&["rl", "progress", run_id])
        .with_context(|| format!("`prime rl progress {run_id}` failed"))
}

fn fetch_prime_checkpoints(run_id: &str) -> Result<Value> {
    run_prime_json(&["rl", "checkpoints", run_id, "-o", "json"])
        .with_context(|| format!("`prime rl checkpoints {run_id} -o json` failed"))
}

fn fetch_prime_logs(run_id: &str, tail: usize) -> Result<String> {
    let tail_str = tail.to_string();
    run_prime_text(&["rl", "logs", run_id, "--tail", &tail_str, "--raw"])
        .with_context(|| format!("`prime rl logs {run_id}` failed"))
}

fn run_prime_json(args: &[&str]) -> Result<Value> {
    let text = run_prime_text(args)?;
    serde_json::from_str(&text).with_context(|| format!("prime output was not valid JSON: {text}"))
}

fn run_prime_text(args: &[&str]) -> Result<String> {
    let output = Command::new("prime")
        .args(args)
        .output()
        .with_context(|| "failed to execute `prime` command")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        bail!("prime command failed: {}", stderr);
    }

    String::from_utf8(output.stdout).context("prime output was not UTF-8")
}

fn read_json_file(path: &PathBuf) -> Result<Value> {
    let txt = fs::read_to_string(path)?;
    serde_json::from_str(&txt).with_context(|| format!("invalid JSON at {}", path.display()))
}

fn parse_metric_points(root: &Value) -> Result<(Vec<MetricPoint>, String)> {
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

fn detect_step(row: &Value) -> Option<i64> {
    const STEP_KEYS: &[&str] = &[
        "step",
        "global_step",
        "train_step",
        "iteration",
        "checkpoint_step",
    ];

    for key in STEP_KEYS {
        if let Some(v) = row.get(*key) {
            if let Some(i) = to_i64(v) {
                return Some(i);
            }
        }
    }

    None
}

fn parse_distribution_steps(root: &Value) -> Vec<i64> {
    let mut steps = root
        .get("steps_with_distributions")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(to_i64).collect::<Vec<_>>())
        .unwrap_or_default();
    steps.sort_unstable();
    steps.dedup();
    steps
}

fn merge_distribution_steps(a: &[i64], b: &[i64]) -> Vec<i64> {
    if !a.is_empty() && !b.is_empty() {
        let mut out = a.iter().copied().filter(|x| b.contains(x)).collect::<Vec<_>>();
        out.sort_unstable();
        out.dedup();
        if !out.is_empty() {
            return out;
        }
    }

    let mut out = a.iter().chain(b.iter()).copied().collect::<Vec<_>>();
    out.sort_unstable();
    out.dedup();
    out
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

fn summarize_points(points: &[MetricPoint], window: usize, thresholds: &[f64]) -> Summary {
    let steps = points.len();

    let (best_step, best_reward) = points
        .iter()
        .max_by(|a, b| a.reward.partial_cmp(&b.reward).unwrap_or(Ordering::Equal))
        .map(|p| (p.step, p.reward))
        .unwrap_or((0, 0.0));

    let final_reward = points.last().map(|p| p.reward).unwrap_or(0.0);
    let auc_reward = auc(points);

    let n = points.len().min(window.max(1));
    let tail = &points[points.len() - n..];
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
    let mean_v = mean(slice.iter().map(|p| p.reward));
    let slope_v = linear_slope(slice);

    SegmentSummary {
        mean: mean_v,
        slope: slope_v,
    }
}

fn auc(points: &[MetricPoint]) -> f64 {
    if points.len() < 2 {
        return points.first().map(|p| p.reward).unwrap_or(0.0);
    }

    let mut total = 0.0;
    for w in points.windows(2) {
        let p0 = &w[0];
        let p1 = &w[1];
        let dx = (p1.step - p0.step).max(1) as f64;
        total += 0.5 * (p0.reward + p1.reward) * dx;
    }
    total
}

fn mean<I>(it: I) -> f64
where
    I: Iterator<Item = f64>,
{
    let vals = it.collect::<Vec<_>>();
    if vals.is_empty() {
        return 0.0;
    }
    vals.iter().sum::<f64>() / vals.len() as f64
}

fn stddev<I>(it: I, mu: f64) -> f64
where
    I: Iterator<Item = f64>,
{
    let vals = it.collect::<Vec<_>>();
    if vals.len() < 2 {
        return 0.0;
    }
    let var = vals.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / vals.len() as f64;
    var.sqrt()
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

fn compare_runs(a: &RunData, b: &RunData, thresholds: &[f64]) -> Comparison {
    let sa = &a.summary;
    let sb = &b.summary;

    let mut rows = vec![
        metric_row(
            "steps",
            sa.steps as f64,
            sb.steps as f64,
            false,
            |v| format!("{}", v as usize),
        ),
        metric_row(
            "final_reward",
            sa.final_reward,
            sb.final_reward,
            true,
            |v| format!("{v:.4}"),
        ),
        metric_row(
            "best_reward",
            sa.best_reward,
            sb.best_reward,
            true,
            |v| format!("{v:.4}"),
        ),
        metric_row(
            "best_step (lower better)",
            sa.best_step as f64,
            sb.best_step as f64,
            false,
            |v| format!("{}", v as i64),
        ),
        metric_row(
            "auc_reward",
            sa.auc_reward,
            sb.auc_reward,
            true,
            |v| format!("{v:.2}"),
        ),
        metric_row(
            "last_window_mean",
            sa.last_mean,
            sb.last_mean,
            true,
            |v| format!("{v:.4}"),
        ),
        metric_row(
            "last_window_std (lower better)",
            sa.last_std,
            sb.last_std,
            false,
            |v| format!("{v:.4}"),
        ),
        metric_row(
            "last_window_slope",
            sa.last_slope,
            sb.last_slope,
            true,
            |v| format!("{v:.6}"),
        ),
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

    for t in thresholds {
        let a_step = sa
            .time_to_threshold
            .iter()
            .find(|(th, _)| (*th - *t).abs() < f64::EPSILON)
            .and_then(|(_, s)| *s);
        let b_step = sb
            .time_to_threshold
            .iter()
            .find(|(th, _)| (*th - *t).abs() < f64::EPSILON)
            .and_then(|(_, s)| *s);

        let (a_str, b_str, delta, kind) = match (a_step, b_step) {
            (Some(x), Some(y)) => {
                let d = y - x;
                let kind = if d < 0 {
                    DeltaKind::Better
                } else if d > 0 {
                    DeltaKind::Worse
                } else {
                    DeltaKind::Neutral
                };
                (x.to_string(), y.to_string(), format_signed_i64(d), kind)
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
            metric: format!("time_to_{t:.2}"),
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
        findings.push(
            "Run B appears plateaued in the late segment (slope near zero).".to_string(),
        );
    }

    if findings.is_empty() {
        findings.push(
            "No large deltas passed default summary heuristics; inspect segment rows and chart."
                .to_string(),
        );
    }

    Comparison { rows, findings }
}

fn build_distribution_comparison(a: &DistributionBundle, b: &DistributionBundle) -> DistributionComparison {
    let mut rows = Vec::new();
    rows.push(optional_metric_row(
        "distribution_step",
        a.step.map(|x| x as f64),
        b.step.map(|x| x as f64),
        false,
        |v| format!("{v:.0}"),
    ));

    rows.push(optional_metric_row(
        "reward_count",
        a.reward.as_ref().map(|s| s.total_count),
        b.reward.as_ref().map(|s| s.total_count),
        true,
        |v| format!("{v:.0}"),
    ));
    rows.push(optional_metric_row(
        "reward_mean",
        a.reward.as_ref().map(|s| s.mean),
        b.reward.as_ref().map(|s| s.mean),
        true,
        |v| format!("{v:.4}"),
    ));
    rows.push(optional_metric_row(
        "reward_p10",
        a.reward.as_ref().map(|s| s.p10),
        b.reward.as_ref().map(|s| s.p10),
        true,
        |v| format!("{v:.4}"),
    ));
    rows.push(optional_metric_row(
        "reward_p50",
        a.reward.as_ref().map(|s| s.p50),
        b.reward.as_ref().map(|s| s.p50),
        true,
        |v| format!("{v:.4}"),
    ));
    rows.push(optional_metric_row(
        "reward_p90",
        a.reward.as_ref().map(|s| s.p90),
        b.reward.as_ref().map(|s| s.p90),
        true,
        |v| format!("{v:.4}"),
    ));
    rows.push(optional_metric_row(
        "reward_iqr",
        a.reward.as_ref().map(|s| s.p75 - s.p25),
        b.reward.as_ref().map(|s| s.p75 - s.p25),
        false,
        |v| format!("{v:.4}"),
    ));
    rows.push(optional_metric_row(
        "reward_entropy",
        a.reward.as_ref().map(|s| s.entropy),
        b.reward.as_ref().map(|s| s.entropy),
        false,
        |v| format!("{v:.4}"),
    ));

    rows.push(optional_metric_row(
        "adv_p50",
        a.advantage.as_ref().map(|s| s.p50),
        b.advantage.as_ref().map(|s| s.p50),
        true,
        |v| format!("{v:.4}"),
    ));
    rows.push(optional_metric_row(
        "adv_mean",
        a.advantage.as_ref().map(|s| s.mean),
        b.advantage.as_ref().map(|s| s.mean),
        true,
        |v| format!("{v:.4}"),
    ));
    rows.push(optional_metric_row(
        "adv_iqr",
        a.advantage.as_ref().map(|s| s.p75 - s.p25),
        b.advantage.as_ref().map(|s| s.p75 - s.p25),
        false,
        |v| format!("{v:.4}"),
    ));
    rows.push(optional_metric_row(
        "adv_std",
        a.advantage.as_ref().map(|s| s.std),
        b.advantage.as_ref().map(|s| s.std),
        false,
        |v| format!("{v:.4}"),
    ));

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

    let reward_bars_a = a
        .reward
        .as_ref()
        .map(|x| bins_to_bar_data(&x.bins))
        .unwrap_or_default();
    let reward_bars_b = b
        .reward
        .as_ref()
        .map(|x| bins_to_bar_data(&x.bins))
        .unwrap_or_default();

    DistributionComparison {
        rows,
        findings,
        reward_bars_a,
        reward_bars_b,
    }
}

fn build_health_comparison(a: &HealthData, b: &HealthData) -> HealthComparison {
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
        a.log
            .as_ref()
            .map(|x| per_1k(x.warnings as f64, x.lines as f64)),
        b.log
            .as_ref()
            .map(|x| per_1k(x.warnings as f64, x.lines as f64)),
        false,
        |v| format!("{v:.2}"),
    ));

    rows.push(optional_metric_row(
        "errors_per_1k_lines (lower better)",
        a.log
            .as_ref()
            .map(|x| per_1k(x.errors as f64, x.lines as f64)),
        b.log
            .as_ref()
            .map(|x| per_1k(x.errors as f64, x.lines as f64)),
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
        let err_delta = per_1k(bl.errors as f64, bl.lines as f64) - per_1k(al.errors as f64, al.lines as f64);
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

fn parse_log_health(text: &str) -> LogHealth {
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

fn parse_checkpoint_health(root: &Value) -> Option<CheckpointHealth> {
    let items = if let Some(arr) = root.get("checkpoints").and_then(Value::as_array) {
        arr
    } else if let Some(arr) = root.as_array() {
        arr
    } else {
        return None;
    };

    let mut c = CheckpointHealth {
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
            "READY" => c.ready += 1,
            "FAILED" => c.failed += 1,
            "PENDING" => c.pending += 1,
            "UPLOADING" => c.uploading += 1,
            _ => c.other += 1,
        }
    }

    Some(c)
}

#[derive(Debug, Clone)]
struct HistogramCandidate {
    kind_hint: String,
    step: Option<i64>,
    bins: Vec<HistogramBin>,
}

fn parse_distribution_bundle(root: &Value) -> DistributionBundle {
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

fn bins_to_bar_data(bins: &[HistogramBin]) -> Vec<(String, u64)> {
    bins.iter()
        .map(|b| {
            let label = format!("{:.3}-{:.3}", b.lower, b.upper);
            let value = b.count.round().max(0.0) as u64;
            (label, value)
        })
        .collect::<Vec<_>>()
}

fn pick_distribution_candidate<'a>(
    candidates: &'a [HistogramCandidate],
    keywords: &[&str],
) -> Option<&'a HistogramCandidate> {
    let mut filtered = candidates
        .iter()
        .filter(|c| {
            if keywords.is_empty() {
                true
            } else {
                let hint = c.kind_hint.to_ascii_lowercase();
                keywords.iter().any(|k| hint.contains(k))
            }
        })
        .collect::<Vec<_>>();

    if filtered.is_empty() {
        return None;
    }

    filtered.sort_by(|a, b| {
        let sa = a.step.unwrap_or(-1);
        let sb = b.step.unwrap_or(-1);
        sb.cmp(&sa)
            .then_with(|| b.bins.len().cmp(&a.bins.len()))
            .then_with(|| b.kind_hint.len().cmp(&a.kind_hint.len()))
    });

    filtered.first().copied()
}

fn collect_histogram_candidates(
    value: &Value,
    hint: &str,
    inherited_step: Option<i64>,
    out: &mut Vec<HistogramCandidate>,
) {
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
                if let Some(arr) = map.get(key).and_then(Value::as_array) {
                    if let Some(bins) = parse_bins_array(arr) {
                        out.push(HistogramCandidate {
                            kind_hint: format!("{hint}.{key}"),
                            step,
                            bins,
                        });
                    }
                }
            }

            for (k, v) in map {
                let child_hint = format!("{hint}.{k}");
                collect_histogram_candidates(v, &child_hint, step, out);
            }
        }
        Value::Array(arr) => {
            if let Some(bins) = parse_bins_array(arr) {
                out.push(HistogramCandidate {
                    kind_hint: hint.to_string(),
                    step: inherited_step,
                    bins,
                });
            }
            for (i, v) in arr.iter().enumerate() {
                let child_hint = format!("{hint}[{i}]");
                collect_histogram_candidates(v, &child_hint, inherited_step, out);
            }
        }
        _ => {}
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
                &[
                    "lower",
                    "low",
                    "min",
                    "start",
                    "left",
                    "bin_start",
                    "bucket_start",
                    "x0",
                ],
            )
            .or_else(|| get_obj_f64(obj, &["x"]))?;

            let upper = get_obj_f64(
                obj,
                &[
                    "upper",
                    "high",
                    "max",
                    "end",
                    "right",
                    "bin_end",
                    "bucket_end",
                    "x1",
                ],
            )
            .or_else(|| Some(lower + 1.0))?;

            (lower, upper)
        };

        let count = get_obj_f64(
            obj,
            &[
                "count",
                "n",
                "freq",
                "frequency",
                "value",
                "y",
                "mass",
            ],
        )?;

        let (l, u) = if upper >= lower {
            (lower, upper)
        } else {
            (upper, lower)
        };

        if count.is_finite() && l.is_finite() && u.is_finite() && count >= 0.0 {
            bins.push(HistogramBin {
                lower: l,
                upper: u,
                count,
            });
        }
    }

    if bins.len() < 2 {
        None
    } else {
        Some(bins)
    }
}

fn parse_bin_range(s: &str) -> Option<(f64, f64)> {
    // Prime currently returns bins like:
    // "0.026-0.051" and "-0.385--0.343"
    let s = s.trim();
    if s.len() < 3 {
        return None;
    }

    let chars = s.char_indices().collect::<Vec<_>>();
    for idx in 1..chars.len().saturating_sub(1) {
        let (i, c) = chars[idx];
        if c != '-' {
            continue;
        }

        let prev = chars[idx - 1].1;
        let next = chars[idx + 1].1;

        let prev_ok = prev.is_ascii_digit() || prev == '.';
        let next_ok = next.is_ascii_digit() || next == '.' || next == '-';
        if !prev_ok || !next_ok {
            continue;
        }

        let left = s[..i].trim().parse::<f64>().ok()?;
        let right = s[i + 1..].trim().parse::<f64>().ok()?;
        return Some((left, right));
    }

    None
}

fn get_obj_f64(obj: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<f64> {
    for key in keys {
        if let Some(v) = obj.get(*key).and_then(to_f64) {
            return Some(v);
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

    let p10 = histogram_quantile(&bins, 0.10, total_count);
    let p25 = histogram_quantile(&bins, 0.25, total_count);
    let p50 = histogram_quantile(&bins, 0.50, total_count);
    let p75 = histogram_quantile(&bins, 0.75, total_count);
    let p90 = histogram_quantile(&bins, 0.90, total_count);

    let mean = bins
        .iter()
        .map(|b| ((b.lower + b.upper) * 0.5) * b.count)
        .sum::<f64>()
        / total_count;

    let var = bins
        .iter()
        .map(|b| {
            let m = (b.lower + b.upper) * 0.5;
            (m - mean).powi(2) * b.count
        })
        .sum::<f64>()
        / total_count;
    let std = var.sqrt();

    let mut entropy = 0.0;
    for b in &bins {
        let p = b.count / total_count;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }

    Some(DistributionStats {
        bins,
        total_count,
        mean,
        std,
        p10,
        p25,
        p50,
        p75,
        p90,
        entropy,
    })
}

fn histogram_quantile(bins: &[HistogramBin], q: f64, total_count: f64) -> f64 {
    if bins.is_empty() {
        return 0.0;
    }
    let target = total_count * q.clamp(0.0, 1.0);
    let mut acc = 0.0;

    for b in bins {
        let next = acc + b.count;
        if next >= target && b.count > 0.0 {
            let ratio = ((target - acc) / b.count).clamp(0.0, 1.0);
            return b.lower + ratio * (b.upper - b.lower);
        }
        acc = next;
    }

    bins.last().map(|b| b.upper).unwrap_or(0.0)
}

fn js_divergence(a_bins: &[HistogramBin], b_bins: &[HistogramBin]) -> f64 {
    let min_x = a_bins
        .iter()
        .chain(b_bins.iter())
        .map(|b| b.lower)
        .fold(f64::INFINITY, f64::min);
    let max_x = a_bins
        .iter()
        .chain(b_bins.iter())
        .map(|b| b.upper)
        .fold(f64::NEG_INFINITY, f64::max);

    if !min_x.is_finite() || !max_x.is_finite() || max_x <= min_x {
        return 0.0;
    }

    let n = 64usize;
    let p = project_bins_to_grid(a_bins, min_x, max_x, n);
    let q = project_bins_to_grid(b_bins, min_x, max_x, n);

    let mut m = vec![0.0; n];
    for i in 0..n {
        m[i] = 0.5 * (p[i] + q[i]);
    }

    0.5 * kl_divergence(&p, &m) + 0.5 * kl_divergence(&q, &m)
}

fn project_bins_to_grid(bins: &[HistogramBin], min_x: f64, max_x: f64, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    if n == 0 || max_x <= min_x {
        return out;
    }

    let total_count = bins.iter().map(|b| b.count).sum::<f64>();
    if total_count <= 0.0 {
        return out;
    }

    let width = (max_x - min_x) / n as f64;

    for b in bins {
        if b.count <= 0.0 {
            continue;
        }

        let l = b.lower.max(min_x);
        let u = b.upper.min(max_x);
        if u <= l {
            continue;
        }

        let mass = b.count / total_count;
        let bin_width = (b.upper - b.lower).max(1e-9);

        let start_idx = ((l - min_x) / width).floor().max(0.0) as usize;
        let end_idx = (((u - min_x) / width).ceil() as usize).min(n);

        for i in start_idx..end_idx {
            let cell_l = min_x + (i as f64) * width;
            let cell_u = cell_l + width;
            let overlap = (u.min(cell_u) - l.max(cell_l)).max(0.0);
            if overlap > 0.0 {
                out[i] += mass * (overlap / bin_width);
            }
        }
    }

    let s = out.iter().sum::<f64>();
    if s > 0.0 {
        for x in &mut out {
            *x /= s;
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
        (Some(av), Some(bv)) => metric_row(metric, av, bv, higher_is_better, format_fn),
        (Some(av), None) => CompareRowData {
            metric: metric.to_string(),
            a: format_fn(av),
            b: "n/a".to_string(),
            delta: "B missing".to_string(),
            kind: DeltaKind::Neutral,
        },
        (None, Some(bv)) => CompareRowData {
            metric: metric.to_string(),
            a: "n/a".to_string(),
            b: format_fn(bv),
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

fn to_chart_points(points: &[MetricPoint]) -> Vec<(f64, f64)> {
    points
        .iter()
        .map(|p| (p.step as f64, p.reward))
        .collect::<Vec<_>>()
}

fn run_app(
    terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>,
    app: &mut App,
) -> Result<AppLoopAction> {
    loop {
        poll_distribution_fetch(app);
        if app.dist_loading {
            app.dist_throbber.calc_next();
        }

        terminal.draw(|f| ui::draw_ui(f, app))?;

        if event::poll(Duration::from_millis(150))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => return Ok(AppLoopAction::Quit),
                    KeyCode::Char('b') => return Ok(AppLoopAction::BackToPicker),
                    KeyCode::Char('1') => app.active_tab = Tab::Summary,
                    KeyCode::Char('2') => app.active_tab = Tab::Distributions,
                    KeyCode::Char('3') => app.active_tab = Tab::Health,
                    KeyCode::Right => app.active_tab = app.active_tab.next(),
                    KeyCode::Left => app.active_tab = app.active_tab.prev(),
                    KeyCode::Char('l') => {
                        if app.active_tab == Tab::Distributions {
                            shift_distribution_step(app, 1);
                        }
                    }
                    KeyCode::Char('h') => {
                        if app.active_tab == Tab::Distributions {
                            shift_distribution_step(app, -1);
                        }
                    }
                    KeyCode::Down | KeyCode::Char('j') => move_active_selection(app, 1),
                    KeyCode::Up | KeyCode::Char('k') => move_active_selection(app, -1),
                    _ => {}
                }
            }
        }
    }

}

fn shift_distribution_step(app: &mut App, delta: isize) {
    if app.dist_steps.is_empty() || app.dist_loading {
        return;
    }

    let cur = app.current_dist_index as isize;
    let next = (cur + delta).clamp(0, (app.dist_steps.len() as isize) - 1) as usize;
    if next == app.current_dist_index {
        return;
    }
    app.current_dist_index = next;

    if let Some(step) = app.dist_steps.get(next).copied() {
        start_distribution_step_fetch(app, step);
    }
}

fn start_distribution_step_fetch(app: &mut App, step: i64) {
    let (Some(left_id), Some(right_id)) = (app.left.run_id.clone(), app.right.run_id.clone()) else {
        return;
    };

    let (tx, rx) = mpsc::channel::<Result<DistributionFetchResult, String>>();
    thread::spawn(move || {
        let left_h = thread::spawn(move || {
            fetch_prime_distributions(&left_id, Some(step))
                .map(|v| parse_distribution_bundle(&v))
                .map_err(|e| format!("A distribution fetch failed: {e}"))
        });
        let right_h = thread::spawn(move || {
            fetch_prime_distributions(&right_id, Some(step))
                .map(|v| parse_distribution_bundle(&v))
                .map_err(|e| format!("B distribution fetch failed: {e}"))
        });

        let left = left_h
            .join()
            .map_err(|_| "A distribution fetch thread panicked".to_string())
            .and_then(|x| x);
        let right = right_h
            .join()
            .map_err(|_| "B distribution fetch thread panicked".to_string())
            .and_then(|x| x);

        match (left, right) {
            (Ok(l), Ok(r)) => {
                let _ = tx.send(Ok(DistributionFetchResult {
                    step,
                    left: l,
                    right: r,
                }));
            }
            (Err(e1), Err(e2)) => {
                let _ = tx.send(Err(format!("{e1}; {e2}")));
            }
            (Err(e), _) | (_, Err(e)) => {
                let _ = tx.send(Err(e));
            }
        }
    });

    app.dist_loading = true;
    app.dist_error = None;
    app.dist_rx = Some(rx);
}

fn poll_distribution_fetch(app: &mut App) {
    let Some(rx) = app.dist_rx.as_ref() else {
        return;
    };

    match rx.try_recv() {
        Ok(Ok(payload)) => {
            app.distribution_comparison = build_distribution_comparison(&payload.left, &payload.right);
            app.dist_loading = false;
            app.dist_error = None;
            app.dist_rx = None;
            if let Some(idx) = app.dist_steps.iter().position(|s| *s == payload.step) {
                app.current_dist_index = idx;
            }
        }
        Ok(Err(err)) => {
            app.dist_loading = false;
            app.dist_error = Some(err);
            app.dist_rx = None;
        }
        Err(mpsc::TryRecvError::Empty) => {}
        Err(mpsc::TryRecvError::Disconnected) => {
            app.dist_loading = false;
            app.dist_error = Some("distribution fetch channel disconnected".to_string());
            app.dist_rx = None;
        }
    }
}

fn move_active_selection(app: &mut App, delta: isize) {
    let (state, len) = match app.active_tab {
        Tab::Summary => (&mut app.summary_state, app.summary_comparison.rows.len()),
        Tab::Distributions => (
            &mut app.distributions_state,
            app.distribution_comparison.rows.len(),
        ),
        Tab::Health => (&mut app.health_state, app.health_comparison.rows.len()),
    };

    if len == 0 {
        state.select(None);
        return;
    }

    let current = state.selected().unwrap_or(0) as isize;
    let mut next = current + delta;
    if next < 0 {
        next = 0;
    }
    if next >= len as isize {
        next = (len as isize) - 1;
    }
    state.select(Some(next as usize));
}
