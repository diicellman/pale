use super::model::{App, CompareRowData, DeltaKind, PickerApp, Tab};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::prelude::Frame;
use ratatui::style::{Color, Modifier, Style};
use ratatui::symbols;
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph, Row, Table, TableState, Wrap,
};
use throbber_widgets_tui::{BRAILLE_SIX, Throbber, ThrobberState, WhichUse};

struct LineChartSpec<'a> {
    title: &'a str,
    left_name: &'a str,
    right_name: &'a str,
    x_title: &'a str,
    y_title: &'a str,
}

pub(super) fn draw_loading_ui(f: &mut Frame, label: &str, throbber_state: &mut ThrobberState) {
    draw_loading_centered(f, f.area(), label, throbber_state);
}

fn draw_loading_centered(
    f: &mut Frame,
    area: Rect,
    label: &str,
    throbber_state: &mut ThrobberState,
) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title("Loading")
        .title_alignment(Alignment::Center);
    f.render_widget(block.clone(), area);
    let popup = block.inner(area);
    let inner = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(45),
            Constraint::Length(1),
            Constraint::Percentage(45),
        ])
        .split(popup);

    let throbber = Throbber::default()
        .label(label)
        .throbber_set(BRAILLE_SIX)
        .use_type(WhichUse::Spin)
        .throbber_style(
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        );
    let line = throbber.to_line(throbber_state);
    let p = Paragraph::new(line).alignment(Alignment::Center);
    f.render_widget(p, inner[1]);
}

pub(super) fn draw_picker_ui(f: &mut Frame, app: &PickerApp) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(10), Constraint::Length(3)])
        .split(f.area());

    if app.loading {
        let mut spin = app.throbber.clone();
        draw_loading_centered(
            f,
            chunks[0],
            &format!("Loading runs page {}", app.pending_page),
            &mut spin,
        );
    } else if let Some(err) = &app.error {
        let p = Paragraph::new(err.clone())
            .block(Block::default().borders(Borders::ALL).title("Fetch error"))
            .wrap(Wrap { trim: true });
        f.render_widget(p, chunks[0]);
    } else {
        let header = Row::new(vec!["sel", "id", "name", "status", "model", "updated"]).style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        );

        let rows = app
            .runs
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let sel = if Some(i) == app.selected_a {
                    "A"
                } else if Some(i) == app.selected_b {
                    "B"
                } else {
                    ""
                };

                Row::new(vec![
                    sel.to_string(),
                    r.id.clone(),
                    r.name.clone(),
                    r.status.clone(),
                    r.model.clone(),
                    r.updated_at.clone(),
                ])
            })
            .collect::<Vec<_>>();

        let mut state = TableState::default();
        if !app.runs.is_empty() {
            state.select(Some(app.selected));
        }

        let table = Table::new(
            rows,
            [
                Constraint::Length(3),
                Constraint::Length(26),
                Constraint::Percentage(28),
                Constraint::Length(11),
                Constraint::Percentage(24),
                Constraint::Percentage(20),
            ],
        )
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!("Runs (page {})", app.page)),
        )
        .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED));

        f.render_stateful_widget(table, chunks[0], &mut state);
    }

    let foot = Paragraph::new("[/]/←→ page | a/b mark | enter compare | q quit")
        .block(Block::default().borders(Borders::ALL).title("Keys"));
    f.render_widget(foot, chunks[1]);
}

pub(super) fn draw_ui(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(3),
        ])
        .split(f.area());

    let left_id = app.left.run_id.as_deref().unwrap_or("-");
    let right_id = app.right.run_id.as_deref().unwrap_or("-");
    let title = Paragraph::new(format!("A: {} | B: {}", left_id, right_id))
        .block(Block::default().borders(Borders::ALL).title("Session"));
    f.render_widget(title, chunks[0]);

    draw_tabs(f, app, chunks[1]);

    match app.active_tab {
        Tab::Summary => draw_summary_tab(f, app, chunks[2]),
        Tab::Distributions => draw_distributions_tab(f, app, chunks[2]),
        Tab::Health => draw_health_tab(f, app, chunks[2]),
    }

    let nav =
        Paragraph::new("q quit | b picker | 1/2/3 tabs | ←/→ tabs | j/k rows | h/l dist-step")
            .block(Block::default().borders(Borders::ALL).title("Keys"));
    f.render_widget(nav, chunks[3]);
}

fn draw_tabs(f: &mut Frame, app: &App, area: Rect) {
    let tabs = [Tab::Summary, Tab::Distributions, Tab::Health];
    let mut spans = Vec::new();

    for (idx, tab) in tabs.iter().enumerate() {
        if idx > 0 {
            spans.push(Span::raw("  "));
        }

        let style = if *tab == app.active_tab {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        };

        spans.push(Span::styled(
            format!("[{}] {}", idx + 1, tab.title()),
            style,
        ));
    }

    let line = Line::from(spans);
    let widget = Paragraph::new(line).block(Block::default().borders(Borders::ALL).title("Tabs"));
    f.render_widget(widget, area);
}

fn draw_summary_tab(f: &mut Frame, app: &mut App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(16),
            Constraint::Min(8),
            Constraint::Length(12),
        ])
        .split(area);

    render_compare_table(
        f,
        chunks[0],
        "Summary metrics",
        &app.summary_comparison.rows,
        &mut app.summary_state,
    );

    let findings = Paragraph::new(app.summary_comparison.findings_text.as_str())
        .block(Block::default().borders(Borders::ALL).title("Key findings"))
        .wrap(Wrap { trim: true });
    f.render_widget(findings, chunks[1]);

    let chart = LineChartSpec {
        title: "Reward trajectory",
        left_name: &app.left.label,
        right_name: &app.right.label,
        x_title: "step",
        y_title: "reward",
    };
    draw_line_chart(
        f,
        chunks[2],
        &chart,
        &app.reward_chart_left,
        &app.reward_chart_right,
    );
}

fn draw_distributions_tab(f: &mut Frame, app: &mut App, area: Rect) {
    let min_middle = 10u16;
    let mut top_h = ((area.height as f32) * 0.24).round() as u16;
    let mut bottom_h = ((area.height as f32) * 0.18).round() as u16;
    top_h = top_h.clamp(5, 12);
    bottom_h = bottom_h.clamp(3, 7);

    if top_h + bottom_h + min_middle > area.height {
        let mut overflow = top_h + bottom_h + min_middle - area.height;

        let cut_bottom = overflow.min(bottom_h.saturating_sub(3));
        bottom_h = bottom_h.saturating_sub(cut_bottom);
        overflow = overflow.saturating_sub(cut_bottom);

        let cut_top = overflow.min(top_h.saturating_sub(5));
        top_h = top_h.saturating_sub(cut_top);
    }

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(top_h),
            Constraint::Min(min_middle),
            Constraint::Length(bottom_h),
        ])
        .split(area);

    render_compare_table(
        f,
        chunks[0],
        "Distribution metrics",
        &app.distribution_comparison.rows,
        &mut app.distributions_state,
    );

    let middle = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);

    let step_hint = app
        .dist_steps
        .get(app.current_dist_index)
        .map(|s| s.to_string())
        .unwrap_or_else(|| "latest".to_string());

    if app.dist_loading {
        draw_loading_centered(
            f,
            middle[0],
            "Loading A distribution",
            &mut app.dist_throbber,
        );
        draw_loading_centered(
            f,
            middle[1],
            "Loading B distribution",
            &mut app.dist_throbber,
        );
    } else {
        draw_distribution_bars(
            f,
            middle[0],
            &format!("A bins:rewards | step={step_hint}"),
            app.distribution_comparison.reward_bars_a.as_slice(),
            Color::Cyan,
        );
        draw_distribution_bars(
            f,
            middle[1],
            &format!("B bins:rewards | step={step_hint}"),
            app.distribution_comparison.reward_bars_b.as_slice(),
            Color::Magenta,
        );
    }

    let findings_text = if let Some(err) = &app.dist_error {
        format!(
            "{}\n\nFetch error: {err}",
            app.distribution_comparison.findings_text
        )
    } else {
        app.distribution_comparison.findings_text.clone()
    };

    let findings = Paragraph::new(findings_text)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Distribution findings"),
        )
        .wrap(Wrap { trim: true });
    f.render_widget(findings, chunks[2]);
}

fn draw_distribution_bars(
    f: &mut Frame,
    area: Rect,
    title: &str,
    bins: &[(String, u64)],
    color: Color,
) {
    if bins.is_empty() {
        let empty = Paragraph::new("No histogram bins available for this run/step.")
            .block(Block::default().borders(Borders::ALL).title(title));
        f.render_widget(empty, area);
        return;
    }
    let block = Block::default().borders(Borders::ALL).title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);

    if inner.height == 0 || inner.width < 16 {
        return;
    }

    let max_count = bins.iter().map(|(_, c)| *c).max().unwrap_or(1).max(1);
    render_distribution_lines(f, inner, bins, max_count, color, true);
}

fn render_distribution_lines(
    f: &mut Frame,
    area: Rect,
    bins: &[(String, u64)],
    max_count: u64,
    color: Color,
    show_overflow_hint: bool,
) {
    if area.height == 0 || area.width < 12 {
        return;
    }

    let max_rows = area.height as usize;
    let count_w = (max_count.ilog10() as usize + 1).max(3);
    let label_w = 15usize;
    let reserved = label_w + 1 + count_w + 1;
    let bar_w = area.width.saturating_sub(reserved as u16) as usize;

    let mut lines = Vec::new();
    if bins.len() <= max_rows {
        for (label, count) in bins {
            lines.push(distribution_line(
                label, *count, label_w, count_w, bar_w, max_count, color,
            ));
        }
    } else if max_rows <= 2 {
        for (label, count) in bins.iter().rev().take(max_rows).rev() {
            lines.push(distribution_line(
                label, *count, label_w, count_w, bar_w, max_count, color,
            ));
        }
    } else {
        let visible_rows = max_rows - 1;
        let head_rows = visible_rows / 2;
        let tail_rows = visible_rows - head_rows;

        for (label, count) in bins.iter().take(head_rows) {
            lines.push(distribution_line(
                label, *count, label_w, count_w, bar_w, max_count, color,
            ));
        }

        if show_overflow_hint {
            lines.push(Line::from(Span::styled(
                format!("... +{} hidden bins ...", bins.len() - visible_rows),
                Style::default().fg(Color::DarkGray),
            )));
        }

        for (label, count) in bins.iter().skip(bins.len() - tail_rows) {
            lines.push(distribution_line(
                label, *count, label_w, count_w, bar_w, max_count, color,
            ));
        }
    }

    f.render_widget(Paragraph::new(lines), area);
}

fn distribution_line(
    label: &str,
    count: u64,
    label_w: usize,
    count_w: usize,
    bar_w: usize,
    max_count: u64,
    color: Color,
) -> Line<'static> {
    let truncated = if label.len() > label_w {
        let mut s = label[..label_w.saturating_sub(1)].to_string();
        s.push('~');
        s
    } else {
        format!("{label:<label_w$}")
    };

    let len = if count == 0 || bar_w == 0 {
        0
    } else {
        ((count as f64 / max_count as f64) * bar_w as f64)
            .round()
            .max(1.0) as usize
    };

    Line::from(vec![
        Span::styled(truncated, Style::default().fg(Color::Gray)),
        Span::raw(" "),
        Span::styled(
            format!("{count:>count_w$}"),
            Style::default().fg(Color::White),
        ),
        Span::raw(" "),
        Span::styled("█".repeat(len), Style::default().fg(color)),
    ])
}

fn draw_health_tab(f: &mut Frame, app: &mut App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(14), Constraint::Min(10)])
        .split(area);

    render_compare_table(
        f,
        chunks[0],
        "Run health metrics",
        &app.health_comparison.rows,
        &mut app.health_state,
    );

    let findings = Paragraph::new(app.health_comparison.findings_text.as_str())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Health findings"),
        )
        .wrap(Wrap { trim: true });
    f.render_widget(findings, chunks[1]);
}

fn render_compare_table(
    f: &mut Frame,
    area: Rect,
    title: &str,
    rows: &[CompareRowData],
    state: &mut TableState,
) {
    let header = Row::new(vec!["metric", "A", "B", "delta (B-A)"]).style(
        Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD),
    );

    let data_rows = rows
        .iter()
        .map(|r| {
            let style = match r.kind {
                DeltaKind::Better => Style::default().fg(Color::Green),
                DeltaKind::Worse => Style::default().fg(Color::Red),
                DeltaKind::Neutral => Style::default().fg(Color::Gray),
            };
            Row::new(vec![
                r.metric.clone(),
                r.a.clone(),
                r.b.clone(),
                r.delta.clone(),
            ])
            .style(style)
        })
        .collect::<Vec<_>>();

    let table = Table::new(
        data_rows,
        [
            Constraint::Percentage(38),
            Constraint::Percentage(18),
            Constraint::Percentage(18),
            Constraint::Percentage(26),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title(title))
    .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED));

    f.render_stateful_widget(table, area, state);
}

fn draw_line_chart(
    f: &mut Frame,
    area: Rect,
    spec: &LineChartSpec<'_>,
    a_points: &[(f64, f64)],
    b_points: &[(f64, f64)],
) {
    if a_points.is_empty() && b_points.is_empty() {
        let empty = Paragraph::new("No chart data")
            .block(Block::default().borders(Borders::ALL).title(spec.title));
        f.render_widget(empty, area);
        return;
    }

    let (min_x, max_x, min_y, max_y) = chart_bounds(a_points, b_points);

    let datasets = vec![
        Dataset::default()
            .name(spec.left_name)
            .graph_type(GraphType::Line)
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Cyan))
            .data(a_points),
        Dataset::default()
            .name(spec.right_name)
            .graph_type(GraphType::Line)
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Magenta))
            .data(b_points),
    ];

    let mid_x = (min_x + max_x) / 2.0;
    let mid_y = (min_y + max_y) / 2.0;

    let chart = Chart::new(datasets)
        .block(Block::default().borders(Borders::ALL).title(spec.title))
        .x_axis(
            Axis::default()
                .title(spec.x_title)
                .bounds([min_x, max_x])
                .labels(vec![
                    Line::from(format!("{min_x:.2}")),
                    Line::from(format!("{mid_x:.2}")),
                    Line::from(format!("{max_x:.2}")),
                ]),
        )
        .y_axis(
            Axis::default()
                .title(spec.y_title)
                .bounds([min_y, max_y])
                .labels(vec![
                    Line::from(format!("{min_y:.2}")),
                    Line::from(format!("{mid_y:.2}")),
                    Line::from(format!("{max_y:.2}")),
                ]),
        );

    f.render_widget(chart, area);
}

fn chart_bounds(a: &[(f64, f64)], b: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    let mut all = Vec::with_capacity(a.len() + b.len());
    all.extend_from_slice(a);
    all.extend_from_slice(b);

    if all.is_empty() {
        return (0.0, 1.0, 0.0, 1.0);
    }

    let min_x = all.iter().map(|(x, _)| *x).fold(f64::INFINITY, f64::min);
    let max_x = all
        .iter()
        .map(|(x, _)| *x)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = all.iter().map(|(_, y)| *y).fold(f64::INFINITY, f64::min);
    let max_y = all
        .iter()
        .map(|(_, y)| *y)
        .fold(f64::NEG_INFINITY, f64::max);

    let y_pad = ((max_y - min_y) * 0.1).max(0.02);

    (
        min_x,
        if (max_x - min_x).abs() < f64::EPSILON {
            max_x + 1.0
        } else {
            max_x
        },
        min_y - y_pad,
        max_y + y_pad,
    )
}
