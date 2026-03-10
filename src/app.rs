use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand};
use crossterm::event::{self, Event, KeyCode};
use ratatui::widgets::TableState;
use serde_json::Value;
use throbber_widgets_tui::ThrobberState;

mod analysis;
mod model;
mod prime;
mod runner;
mod ui;

use self::analysis::*;
use self::model::*;
use self::prime::*;
use self::runner::{TerminalBackend, with_terminal};

const PICKER_CACHE_PAGES: usize = 8;

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
    #[arg(long, default_value_t = 30)]
    prefetch_runs: usize,

    #[command(subcommand)]
    command: Option<CommandSet>,
}

#[derive(Debug, Subcommand)]
enum CommandSet {
    Compare(CompareArgs),
}

#[derive(Debug, Parser)]
struct CompareArgs {
    #[arg(long)]
    run_a: Option<String>,

    #[arg(long)]
    run_b: Option<String>,

    #[arg(long)]
    metrics_a: Option<PathBuf>,

    #[arg(long)]
    metrics_b: Option<PathBuf>,

    #[arg(long)]
    label_a: Option<String>,

    #[arg(long)]
    label_b: Option<String>,

    #[arg(long, default_value_t = 50)]
    window: usize,

    #[arg(long = "threshold")]
    thresholds: Vec<f64>,

    #[arg(long)]
    dist_step: Option<i64>,

    #[arg(long, default_value_t = 2000)]
    log_tail: usize,

    #[arg(long, default_value_t = false)]
    skip_extras: bool,
}

fn run_picker_mode(prefetch_runs: usize) -> Result<()> {
    with_terminal(|terminal| run_picker_session(terminal, prefetch_runs))
}

fn run_picker_session(terminal: &mut TerminalBackend, prefetch_runs: usize) -> Result<()> {
    let (tx, rx) = mpsc::channel::<PickerFetchOutcome>();
    let mut app = PickerApp {
        runs: Vec::new(),
        selected: 0,
        selected_a: None,
        selected_b: None,
        page: 1,
        pending_page: 1,
        per_page: prefetch_runs.max(1),
        page_cache: Default::default(),
        page_cache_order: Default::default(),
        in_flight_pages: Default::default(),
        cache_capacity: PICKER_CACHE_PAGES,
        loading: false,
        error: None,
        throbber: ThrobberState::default(),
    };
    request_picker_page(&mut app, &tx, 1);

    loop {
        poll_picker_fetches(&mut app, &rx, &tx);
        if app.loading {
            app.throbber.calc_next();
        }

        terminal.draw(|f| ui::draw_picker_ui(f, &app))?;

        if event::poll(Duration::from_millis(150))?
            && let Event::Key(key) = event::read()?
            && handle_picker_key(terminal, &mut app, &tx, key.code)?
        {
            return Ok(());
        }
    }
}

fn apply_picker_payload(app: &mut PickerApp, payload: PickerFetchResult) {
    app.loading = false;
    app.error = None;
    if payload.runs.is_empty() && payload.page > 1 {
        app.pending_page = app.page;
        app.error = Some("No runs on that page.".to_string());
        return;
    }

    app.runs = payload.runs;
    app.page = payload.page;
    app.pending_page = payload.page;
    app.selected = 0;
    app.selected_a = None;
    app.selected_b = None;
}

fn request_picker_page(app: &mut PickerApp, tx: &mpsc::Sender<PickerFetchOutcome>, page: usize) {
    app.pending_page = page;
    app.error = None;

    if let Some(runs) = app.page_cache.get(&page).cloned() {
        touch_picker_page_cache(app, page);
        app.loading = false;
        apply_picker_payload(app, PickerFetchResult { page, runs });
        prefetch_picker_neighbors(app, tx);
        return;
    }

    app.loading = true;
    queue_picker_fetch(app, tx, page);
}

fn poll_picker_fetches(
    app: &mut PickerApp,
    rx: &mpsc::Receiver<PickerFetchOutcome>,
    tx: &mpsc::Sender<PickerFetchOutcome>,
) {
    loop {
        match rx.try_recv() {
            Ok(outcome) => {
                app.in_flight_pages.remove(&outcome.page);
                match outcome.result {
                    Ok(runs) => {
                        cache_picker_page(app, outcome.page, runs.clone());
                        if app.loading && outcome.page == app.pending_page {
                            apply_picker_payload(
                                app,
                                PickerFetchResult {
                                    page: outcome.page,
                                    runs,
                                },
                            );
                            prefetch_picker_neighbors(app, tx);
                        }
                    }
                    Err(err) => {
                        if app.loading && outcome.page == app.pending_page {
                            app.error = Some(err);
                            app.loading = false;
                            app.pending_page = app.page;
                        }
                    }
                }
            }
            Err(mpsc::TryRecvError::Empty) => break,
            Err(mpsc::TryRecvError::Disconnected) => {
                if app.loading {
                    app.error = Some("background fetch channel disconnected".to_string());
                    app.loading = false;
                    app.pending_page = app.page;
                }
                break;
            }
        }
    }
}

fn queue_picker_fetch(app: &mut PickerApp, tx: &mpsc::Sender<PickerFetchOutcome>, page: usize) {
    if app.page_cache.contains_key(&page) || app.in_flight_pages.contains(&page) {
        return;
    }

    app.in_flight_pages.insert(page);
    let tx = tx.clone();
    let per_page = app.per_page;
    thread::spawn(move || {
        let outcome = PickerFetchOutcome {
            page,
            result: fetch_runs_page(per_page, page).map_err(|e| e.to_string()),
        };
        let _ = tx.send(outcome);
    });
}

fn prefetch_picker_neighbors(app: &mut PickerApp, tx: &mpsc::Sender<PickerFetchOutcome>) {
    if app.page > 1 {
        queue_picker_fetch(app, tx, app.page - 1);
    }
    queue_picker_fetch(app, tx, app.page + 1);
}

fn cache_picker_page(app: &mut PickerApp, page: usize, runs: Vec<RunListItem>) {
    app.page_cache.insert(page, runs);
    touch_picker_page_cache(app, page);

    while app.page_cache.len() > app.cache_capacity {
        let evict = app
            .page_cache_order
            .iter()
            .copied()
            .find(|candidate| *candidate != app.page && *candidate != app.pending_page);
        let Some(evict_page) = evict else {
            break;
        };
        app.page_cache.remove(&evict_page);
        if let Some(pos) = app
            .page_cache_order
            .iter()
            .position(|page| *page == evict_page)
        {
            let _ = app.page_cache_order.remove(pos);
        }
    }
}

fn touch_picker_page_cache(app: &mut PickerApp, page: usize) {
    if let Some(pos) = app
        .page_cache_order
        .iter()
        .position(|cached| *cached == page)
    {
        let _ = app.page_cache_order.remove(pos);
    }
    app.page_cache_order.push_back(page);
}

fn handle_picker_key(
    terminal: &mut TerminalBackend,
    app: &mut PickerApp,
    tx: &mpsc::Sender<PickerFetchOutcome>,
    key: KeyCode,
) -> Result<bool> {
    match key {
        KeyCode::Char('q') => return Ok(true),
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
                request_picker_page(app, tx, app.page + 1);
            }
        }
        KeyCode::Left | KeyCode::Char('[') => {
            if !app.loading && app.page > 1 {
                request_picker_page(app, tx, app.page - 1);
            }
        }
        KeyCode::Enter => {
            if app.loading {
                return Ok(false);
            }
            if let (Some(a_idx), Some(b_idx)) = (app.selected_a, app.selected_b) {
                if a_idx == b_idx {
                    return Ok(false);
                }

                let args = CompareArgs {
                    run_a: Some(app.runs[a_idx].id.clone()),
                    run_b: Some(app.runs[b_idx].id.clone()),
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

                match load_compare_app_with_spinner(terminal, "Loading compare view...", args) {
                    Ok(mut compare_app) => match run_app(terminal, &mut compare_app)? {
                        AppLoopAction::Quit => return Ok(true),
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

    Ok(false)
}

fn load_compare_app_with_spinner(
    terminal: &mut TerminalBackend,
    label: &str,
    args: CompareArgs,
) -> Result<App> {
    let (tx, rx) = mpsc::channel::<Result<App, String>>();
    thread::spawn(move || {
        let result = build_compare_app(args).map_err(|e| e.to_string());
        let _ = tx.send(result);
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
    let limit_arg = limit.to_string();
    let page_arg = page.to_string();
    let root = run_prime_json(&[
        "rl",
        "list",
        "-o",
        "json",
        "-n",
        limit_arg.as_str(),
        "-p",
        page_arg.as_str(),
    ])?;
    let runs = root
        .get("runs")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("missing `runs` array in `prime rl list` output"))?;

    Ok(runs
        .iter()
        .map(|run| RunListItem {
            id: run
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string(),
            name: run
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string(),
            status: run
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string(),
            model: run
                .get("base_model")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string(),
            updated_at: run
                .get("updated_at")
                .and_then(Value::as_str)
                .unwrap_or("?")
                .to_string(),
        })
        .collect::<Vec<_>>())
}

fn run_compare(args: CompareArgs) -> Result<()> {
    with_terminal(|terminal| {
        let mut app = load_compare_app_with_spinner(terminal, "Loading compare view...", args)?;
        match run_app(terminal, &mut app)? {
            AppLoopAction::Quit => Ok(()),
            AppLoopAction::BackToPicker => run_picker_session(terminal, 30),
        }
    })
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
        .map_err(|_| anyhow!("left-side loader thread panicked"))??;
    let right = right_handle
        .join()
        .map_err(|_| anyhow!("right-side loader thread panicked"))??;

    let summary_comparison = compare_runs(&left, &right, &thresholds);
    let log_tail = args.log_tail;
    let dist_step = args.dist_step;
    let skip_extras = args.skip_extras;
    let left_run_id = left.run_id.clone();
    let right_run_id = right.run_id.clone();

    let extras_left_handle =
        thread::spawn(move || collect_side_extras(left_run_id, log_tail, dist_step, skip_extras));
    let extras_right_handle =
        thread::spawn(move || collect_side_extras(right_run_id, log_tail, dist_step, skip_extras));

    let extras_left = extras_left_handle
        .join()
        .map_err(|_| anyhow!("left-side extras thread panicked"))?;
    let extras_right = extras_right_handle
        .join()
        .map_err(|_| anyhow!("right-side extras thread panicked"))?;

    let dist_steps = merge_distribution_steps(&extras_left.dist_steps, &extras_right.dist_steps);
    let default_dist_index = dist_steps.len().saturating_sub(1);
    let (dist_tx, dist_rx) = mpsc::channel::<DistributionFetchOutcome>();

    let mut app = App {
        reward_chart_left: to_chart_points(&left.points),
        reward_chart_right: to_chart_points(&right.points),
        summary_comparison,
        distribution_comparison: build_distribution_comparison(
            &extras_left.distributions,
            &extras_right.distributions,
        ),
        health_comparison: build_health_comparison(&extras_left.health, &extras_right.health),
        left,
        right,
        summary_state: selected_state(),
        distributions_state: selected_state(),
        health_state: selected_state(),
        active_tab: Tab::Summary,
        dist_steps,
        current_dist_index: default_dist_index,
        dist_loading: false,
        dist_pending_step: None,
        dist_error: None,
        dist_rx,
        dist_tx,
        dist_cache: Default::default(),
        dist_in_flight: Default::default(),
        dist_throbber: ThrobberState::default(),
    };

    resolve_initial_distribution_index(&mut app, args.dist_step);
    if let Some(step) = current_distribution_step(&app) {
        app.dist_cache.insert(
            step,
            DistributionFetchResult {
                step,
                left: extras_left.distributions.clone(),
                right: extras_right.distributions.clone(),
            },
        );
        prefetch_distribution_neighbors(&mut app);
    }
    Ok(app)
}

fn current_distribution_step(app: &App) -> Option<i64> {
    app.dist_steps.get(app.current_dist_index).copied()
}

fn selected_state() -> TableState {
    let mut state = TableState::default();
    state.select(Some(0));
    state
}

fn resolve_initial_distribution_index(app: &mut App, requested_step: Option<i64>) {
    if let Some(step) = requested_step {
        ensure_step_selected(app, step);
        return;
    }

    if let Some(step) = app
        .distribution_comparison
        .rows
        .iter()
        .find(|row| row.metric == "distribution_step")
        .and_then(|row| {
            row.a
                .parse::<usize>()
                .ok()
                .or_else(|| row.b.parse::<usize>().ok())
        })
        && let Some(index) = app.dist_steps.iter().position(|s| *s as usize == step)
    {
        app.current_dist_index = index;
    }
}

fn ensure_step_selected(app: &mut App, step: i64) {
    if let Some(index) = app
        .dist_steps
        .iter()
        .position(|candidate| *candidate == step)
    {
        app.current_dist_index = index;
        return;
    }

    app.dist_steps.push(step);
    app.dist_steps.sort_unstable();
    app.dist_steps.dedup();
    if let Some(index) = app
        .dist_steps
        .iter()
        .position(|candidate| *candidate == step)
    {
        app.current_dist_index = index;
    }
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
                .and_then(|stem| stem.to_str())
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

    Ok(RunData {
        label,
        run_id,
        summary: summarize_points(&points, window, thresholds),
        points,
    })
}

fn collect_side_extras(
    run_id: Option<String>,
    log_tail: usize,
    dist_step: Option<i64>,
    skip_extras: bool,
) -> SideExtras {
    if skip_extras {
        return SideExtras::default();
    }

    let Some(run_id) = run_id else {
        return SideExtras::default();
    };

    let dist_steps = match fetch_prime_progress(&run_id) {
        Ok(json) => parse_distribution_steps(&json),
        Err(_) => Vec::new(),
    };
    let distributions = match fetch_prime_distributions(&run_id, dist_step) {
        Ok(json) => parse_distribution_bundle(&json),
        Err(_) => DistributionBundle::default(),
    };
    let log = match fetch_prime_logs(&run_id, log_tail) {
        Ok(text) => Some(parse_log_health(&text)),
        Err(_) => None,
    };
    let checkpoints = match fetch_prime_checkpoints(&run_id) {
        Ok(json) => parse_checkpoint_health(&json),
        Err(_) => None,
    };

    SideExtras {
        distributions,
        dist_steps,
        health: HealthData { log, checkpoints },
    }
}

fn run_app(terminal: &mut TerminalBackend, app: &mut App) -> Result<AppLoopAction> {
    loop {
        poll_distribution_fetch(app);
        if app.dist_loading {
            app.dist_throbber.calc_next();
        }

        terminal.draw(|f| ui::draw_ui(f, app))?;

        if event::poll(Duration::from_millis(150))?
            && let Event::Key(key) = event::read()?
        {
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

fn shift_distribution_step(app: &mut App, delta: isize) {
    if app.dist_steps.is_empty() || app.dist_loading {
        return;
    }

    let current = app.current_dist_index as isize;
    let next = (current + delta).clamp(0, app.dist_steps.len() as isize - 1) as usize;
    if next == app.current_dist_index {
        return;
    }

    app.current_dist_index = next;
    if let Some(step) = app.dist_steps.get(next).copied() {
        request_distribution_step(app, step);
    }
}

fn request_distribution_step(app: &mut App, step: i64) {
    if let Some(payload) = app.dist_cache.get(&step).cloned() {
        apply_distribution_payload(app, payload);
        prefetch_distribution_neighbors(app);
        return;
    }

    app.dist_loading = true;
    app.dist_pending_step = Some(step);
    app.dist_error = None;
    queue_distribution_fetch(app, step);
}

fn queue_distribution_fetch(app: &mut App, step: i64) {
    if app.dist_cache.contains_key(&step) || app.dist_in_flight.contains(&step) {
        return;
    }

    let (Some(left_id), Some(right_id)) = (app.left.run_id.clone(), app.right.run_id.clone())
    else {
        return;
    };

    app.dist_in_flight.insert(step);
    let tx = app.dist_tx.clone();
    thread::spawn(move || {
        let left_handle = thread::spawn(move || {
            fetch_prime_distributions(&left_id, Some(step))
                .map(|value| parse_distribution_bundle(&value))
                .map_err(|err| format!("A distribution fetch failed: {err}"))
        });
        let right_handle = thread::spawn(move || {
            fetch_prime_distributions(&right_id, Some(step))
                .map(|value| parse_distribution_bundle(&value))
                .map_err(|err| format!("B distribution fetch failed: {err}"))
        });

        let left = left_handle
            .join()
            .map_err(|_| "A distribution fetch thread panicked".to_string())
            .and_then(|result| result);
        let right = right_handle
            .join()
            .map_err(|_| "B distribution fetch thread panicked".to_string())
            .and_then(|result| result);

        match (left, right) {
            (Ok(left), Ok(right)) => {
                let _ = tx.send(DistributionFetchOutcome {
                    step,
                    result: Ok(DistributionFetchResult { step, left, right }),
                });
            }
            (Err(left), Err(right)) => {
                let _ = tx.send(DistributionFetchOutcome {
                    step,
                    result: Err(format!("{left}; {right}")),
                });
            }
            (Err(err), _) | (_, Err(err)) => {
                let _ = tx.send(DistributionFetchOutcome {
                    step,
                    result: Err(err),
                });
            }
        }
    });
}

fn prefetch_distribution_neighbors(app: &mut App) {
    if app.dist_steps.is_empty() {
        return;
    }

    let current = app.current_dist_index;
    if current > 0 {
        let step = app.dist_steps[current - 1];
        queue_distribution_fetch(app, step);
    }
    if let Some(step) = app.dist_steps.get(current + 1).copied() {
        queue_distribution_fetch(app, step);
    }
}

fn apply_distribution_payload(app: &mut App, payload: DistributionFetchResult) {
    app.distribution_comparison = build_distribution_comparison(&payload.left, &payload.right);
    app.dist_loading = false;
    app.dist_pending_step = None;
    app.dist_error = None;
    if let Some(index) = app.dist_steps.iter().position(|step| *step == payload.step) {
        app.current_dist_index = index;
    }
}

fn poll_distribution_fetch(app: &mut App) {
    loop {
        match app.dist_rx.try_recv() {
            Ok(outcome) => {
                app.dist_in_flight.remove(&outcome.step);
                match outcome.result {
                    Ok(payload) => {
                        app.dist_cache.insert(outcome.step, payload.clone());
                        if app.dist_pending_step == Some(outcome.step) {
                            apply_distribution_payload(app, payload);
                            prefetch_distribution_neighbors(app);
                        }
                    }
                    Err(err) => {
                        if app.dist_pending_step == Some(outcome.step) {
                            app.dist_loading = false;
                            app.dist_pending_step = None;
                            app.dist_error = Some(err);
                        }
                    }
                }
            }
            Err(mpsc::TryRecvError::Empty) => break,
            Err(mpsc::TryRecvError::Disconnected) => {
                if app.dist_loading {
                    app.dist_loading = false;
                    app.dist_pending_step = None;
                    app.dist_error = Some("distribution fetch channel disconnected".to_string());
                }
                break;
            }
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
    let next = (current + delta).clamp(0, len as isize - 1);
    state.select(Some(next as usize));
}
