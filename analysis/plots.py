"""Interactive Plotly figures built only from stored measurements.

Conventions (dataviz method): fixed categorical colour order assigned to the
metric (never re-cycled by rank), one y-axis per panel — metrics with different
units become stacked small multiples sharing the time axis, never a dual axis.
Screen estimates are drawn dashed and labelled "(estimated)" so identity never
depends on colour alone. Anomalies use the reserved 'critical' status colour
with an ✕ marker and a hover label.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analysis.stats import SeriesData

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CRITICAL = "#d03b3b"
WARNING = "#fab219"
MUTED = "#8a8985"

_color_by_metric: dict[str, str] = {}


def color_for(metric: str) -> str:
    """Colour follows the entity: the first time a metric is plotted it takes the
    next slot in fixed order and keeps it for the session. Past eight metrics,
    extra series fall back to a muted grey rather than generating new hues."""
    key = metric.lower()
    if key not in _color_by_metric:
        used = len(_color_by_metric)
        _color_by_metric[key] = CATEGORICAL[used] if used < len(CATEGORICAL) else MUTED
    return _color_by_metric[key]


def trend_figure(
    series: list[SeriesData],
    title: str = "",
    anomalies: dict[str, list[dict[str, Any]]] | None = None,
    thresholds: dict[str, list[tuple[str, float]]] | None = None,
    moving_avg: dict[str, Any] | None = None,
) -> go.Figure:
    series = [s for s in series if s.n > 0]
    units = []
    for s in series:
        u = s.unit or s.metric
        if u not in units:
            units.append(u)
    rows = max(1, len(units))
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=[u for u in units] if rows > 1 else None)
    for s in series:
        row = units.index(s.unit or s.metric) + 1
        est = s.status in ("estimated", "mixed")
        name = s.metric + (" (estimated)" if est else "")
        color = color_for(s.metric)
        fig.add_trace(
            go.Scatter(
                x=s.series.index,
                y=s.series.values,
                name=name,
                mode="lines+markers" if s.n <= 60 else "lines",
                line={"color": color, "width": 2, "dash": "dash" if est else "solid"},
                marker={"size": 8, "color": color, "line": {"width": 2, "color": "rgba(255,255,255,0.9)"}},
                hovertemplate=f"<b>{s.metric}</b><br>%{{x|%Y-%m-%d %H:%M:%S}} UTC<br>%{{y:.4g}} {s.unit}<br>{'estimate from screen' if est else 'source: ' + s.source}<extra></extra>",
                legendgroup=s.metric,
            ),
            row=row,
            col=1,
        )
        if moving_avg and s.metric in moving_avg:
            ma = moving_avg[s.metric]
            fig.add_trace(
                go.Scatter(x=ma.index, y=ma.values, name=f"{s.metric} moving avg", mode="lines", line={"color": color, "width": 1, "dash": "dot"}, opacity=0.7, legendgroup=s.metric, hovertemplate="moving avg %{y:.4g}<extra></extra>"),
                row=row,
                col=1,
            )
        for a in (anomalies or {}).get(s.metric, []):
            fig.add_trace(
                go.Scatter(
                    x=[a["timestamp"]],
                    y=[a["value"]],
                    mode="markers",
                    name="anomaly",
                    showlegend=False,
                    marker={"symbol": "x", "size": 12, "color": CRITICAL, "line": {"width": 2, "color": CRITICAL}},
                    hovertemplate=f"<b>Anomaly ({a.get('direction', '')})</b><br>%{{x|%H:%M:%S}}<br>%{{y:.4g}} {s.unit}<br>robust z = {a.get('robust_z', 0):.2f}<extra></extra>",
                ),
                row=row,
                col=1,
            )
        for label, value in (thresholds or {}).get(s.metric, []):
            fig.add_hline(y=value, line={"color": WARNING, "width": 1.5, "dash": "dash"}, annotation_text=f"⚠ {label} {value:g}", annotation_position="top left", row=row, col=1)
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 15}},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 10, "r": 10, "t": 60 if title else 30, "b": 10},
        height=260 + 170 * rows,
    )
    fig.update_xaxes(showgrid=False, showspikes=True, spikemode="across", spikethickness=1)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def screen_chart_figure(chart_summary_series: list[dict[str, Any]], title: str, unit: str) -> go.Figure:
    """Re-plot series reconstructed from a screenshot chart (all estimates)."""
    fig = go.Figure()
    for s in chart_summary_series:
        c = color_for(s["name"])
        fig.add_trace(go.Scatter(x=s["x"], y=s["y"], name=f"{s['name']} (estimated)", mode="lines", line={"color": c, "width": 2, "dash": "dash"}, hovertemplate=f"%{{x:.4g}}<br>%{{y:.4g}} {unit}<extra>{s['name']}</extra>"))
    fig.update_layout(title={"text": title + " — reconstructed from screenshot", "x": 0, "font": {"size": 14}}, height=320, margin={"l": 10, "r": 10, "t": 50, "b": 10}, hovermode="x unified")
    return fig
