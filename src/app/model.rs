use std::sync::mpsc;

use anyhow::Result;
use ratatui::widgets::TableState;
use throbber_widgets_tui::ThrobberState;

#[derive(Debug, Clone)]
pub(crate) struct MetricPoint {
    pub(crate) step: i64,
    pub(crate) reward: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct SegmentSummary {
    pub(crate) mean: f64,
    pub(crate) slope: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct Summary {
    pub(crate) steps: usize,
    pub(crate) final_reward: f64,
    pub(crate) best_reward: f64,
    pub(crate) best_step: i64,
    pub(crate) auc_reward: f64,
    pub(crate) last_mean: f64,
    pub(crate) last_std: f64,
    pub(crate) last_slope: f64,
    pub(crate) early: SegmentSummary,
    pub(crate) mid: SegmentSummary,
    pub(crate) late: SegmentSummary,
    pub(crate) time_to_threshold: Vec<(f64, Option<i64>)>,
}

#[derive(Debug, Clone)]
pub(crate) struct RunData {
    pub(crate) label: String,
    pub(crate) run_id: Option<String>,
    pub(crate) points: Vec<MetricPoint>,
    pub(crate) summary: Summary,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum DeltaKind {
    Better,
    Worse,
    Neutral,
}

#[derive(Debug, Clone)]
pub(crate) struct CompareRowData {
    pub(crate) metric: String,
    pub(crate) a: String,
    pub(crate) b: String,
    pub(crate) delta: String,
    pub(crate) kind: DeltaKind,
}

#[derive(Debug, Clone)]
pub(crate) struct Comparison {
    pub(crate) rows: Vec<CompareRowData>,
    pub(crate) findings_text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tab {
    Summary,
    Distributions,
    Health,
}

impl Tab {
    pub(crate) fn title(self) -> &'static str {
        match self {
            Self::Summary => "Summary",
            Self::Distributions => "Distributions",
            Self::Health => "Health",
        }
    }

    pub(crate) fn next(self) -> Self {
        match self {
            Self::Summary => Self::Distributions,
            Self::Distributions => Self::Health,
            Self::Health => Self::Summary,
        }
    }

    pub(crate) fn prev(self) -> Self {
        match self {
            Self::Summary => Self::Health,
            Self::Distributions => Self::Summary,
            Self::Health => Self::Distributions,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct HistogramBin {
    pub(crate) lower: f64,
    pub(crate) upper: f64,
    pub(crate) count: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct DistributionStats {
    pub(crate) bins: Vec<HistogramBin>,
    pub(crate) total_count: f64,
    pub(crate) mean: f64,
    pub(crate) std: f64,
    pub(crate) p10: f64,
    pub(crate) p25: f64,
    pub(crate) p50: f64,
    pub(crate) p75: f64,
    pub(crate) p90: f64,
    pub(crate) entropy: f64,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct DistributionBundle {
    pub(crate) step: Option<i64>,
    pub(crate) reward: Option<DistributionStats>,
    pub(crate) advantage: Option<DistributionStats>,
}

#[derive(Debug, Clone)]
pub(crate) struct DistributionComparison {
    pub(crate) rows: Vec<CompareRowData>,
    pub(crate) findings_text: String,
    pub(crate) reward_bars_a: Vec<(String, u64)>,
    pub(crate) reward_bars_b: Vec<(String, u64)>,
}

#[derive(Debug, Clone)]
pub(crate) struct LogHealth {
    pub(crate) lines: usize,
    pub(crate) warnings: usize,
    pub(crate) errors: usize,
    pub(crate) lag_events: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct CheckpointHealth {
    pub(crate) total: usize,
    pub(crate) ready: usize,
    pub(crate) failed: usize,
    pub(crate) pending: usize,
    pub(crate) uploading: usize,
    pub(crate) other: usize,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct HealthData {
    pub(crate) log: Option<LogHealth>,
    pub(crate) checkpoints: Option<CheckpointHealth>,
}

#[derive(Debug, Clone)]
pub(crate) struct HealthComparison {
    pub(crate) rows: Vec<CompareRowData>,
    pub(crate) findings_text: String,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SideExtras {
    pub(crate) distributions: DistributionBundle,
    pub(crate) dist_steps: Vec<i64>,
    pub(crate) health: HealthData,
}

pub(crate) struct App {
    pub(crate) left: RunData,
    pub(crate) right: RunData,
    pub(crate) summary_comparison: Comparison,
    pub(crate) distribution_comparison: DistributionComparison,
    pub(crate) health_comparison: HealthComparison,
    pub(crate) reward_chart_left: Vec<(f64, f64)>,
    pub(crate) reward_chart_right: Vec<(f64, f64)>,
    pub(crate) summary_state: TableState,
    pub(crate) distributions_state: TableState,
    pub(crate) health_state: TableState,
    pub(crate) active_tab: Tab,
    pub(crate) dist_steps: Vec<i64>,
    pub(crate) current_dist_index: usize,
    pub(crate) dist_loading: bool,
    pub(crate) dist_error: Option<String>,
    pub(crate) dist_rx: Option<mpsc::Receiver<Result<DistributionFetchResult, String>>>,
    pub(crate) dist_throbber: ThrobberState,
}

#[derive(Debug, Clone)]
pub(crate) struct DistributionFetchResult {
    pub(crate) step: i64,
    pub(crate) left: DistributionBundle,
    pub(crate) right: DistributionBundle,
}

#[derive(Debug, Clone)]
pub(crate) struct RunListItem {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) status: String,
    pub(crate) model: String,
    pub(crate) updated_at: String,
}

pub(crate) struct PickerApp {
    pub(crate) runs: Vec<RunListItem>,
    pub(crate) selected: usize,
    pub(crate) selected_a: Option<usize>,
    pub(crate) selected_b: Option<usize>,
    pub(crate) page: usize,
    pub(crate) pending_page: usize,
    pub(crate) per_page: usize,
    pub(crate) loading: bool,
    pub(crate) error: Option<String>,
    pub(crate) throbber: ThrobberState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AppLoopAction {
    Quit,
    BackToPicker,
}

#[derive(Debug, Clone)]
pub(crate) struct PickerFetchResult {
    pub(crate) page: usize,
    pub(crate) runs: Vec<RunListItem>,
}
