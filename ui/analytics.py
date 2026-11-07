"""Analytics: KPI cards, interactive trends, window comparison, anomalies, correlation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analysis import plots
from analysis.stats import InsufficientData, compare_windows, correlation, data_quality, descriptive, detect_anomalies, moving_average, rate_of_change, trend
from analysis.windows import WindowError, parse_window
from core.runtime import Runtime

WINDOWS = ["last 5 minutes", "last 10 minutes", "last 30 minutes", "last 1 hour", "last 24 hours", "today", "all"]


def _fmt(v: float | None, unit: str = "") -> str:
    return "—" if v is None else f"{v:,.4g} {unit}".strip()


def analytics_tab(rt: Runtime) -> None:
    names = rt.service.metric_names()
    if not names:
        st.info("No measurements yet. Upload a file in **Data Sources** or start monitoring in **Live Monitor**.")
        return
    c1, c2 = st.columns([2, 1])
    chosen = c1.multiselect("Metrics", names, default=names[:2], max_selections=6, key="an_metrics")
    win_label = c2.selectbox("Window", WINDOWS, index=WINDOWS.index("all"), key="an_window")
    if not chosen:
        return
    series, metas = [], []
    for m in chosen:
        try:
            sd, win, meta = rt.service.get_series(m, win_label)
        except (KeyError, WindowError) as exc:
            st.warning(str(exc))
            continue
        series.append(sd)
        metas.append(meta)
    for sd, meta in zip(series, metas, strict=True):
        if sd.n == 0:
            st.caption(f"{sd.metric}: no data in {meta['window_description']}.")
            continue
        d = descriptive(sd)
        rate = rate_of_change(sd) if sd.n >= 2 and d["span_seconds"] != 0 else None
        tag = " · estimated from screen" if sd.status != "verified" else ""
        st.markdown(f"**{sd.metric}** <small>source `{meta['source_id']}`{tag} · {meta['window_description']}</small>", unsafe_allow_html=True)
        k = st.columns(5)
        k[0].metric("Latest", _fmt(d["last"], sd.unit), help=f"at {d['last_at'][:19]} UTC")
        k[1].metric("Mean", _fmt(d["mean"], sd.unit))
        k[2].metric("Min / Max", f"{d['min']:.4g} / {d['max']:.4g}")
        k[3].metric("Change", _fmt(d["change_pct"], "%") if d["change_pct"] is not None else "—", help=d["change_pct_note"] or "first → last in the window")
        k[4].metric("Rate", f"{rate['slope']:.3g}" if rate else "—", help=rate["rate_unit"] if rate else None)
        for n in sd.notes:
            st.caption(f"ℹ {n}")
    good = [s for s in series if s.n > 0]
    if not good:
        return
    an = {s.metric: detect_anomalies(s)["anomalies"] for s in good if s.n >= 5}
    thr: dict[str, list[tuple[str, float]]] = {}
    for s in good:
        for r in rt.alerts.rules_for(s.metric):
            if r.kind in ("above", "below") and r.threshold is not None:
                thr.setdefault(s.metric, []).append((r.kind, r.threshold))
    ma = {s.metric: moving_average(s, max(3, s.n // 20)) for s in good if s.n >= 20}
    st.plotly_chart(plots.trend_figure(good, "", an, thr, ma), key="an_fig")
    t1, t2, t3, t4 = st.tabs(["Window comparison", "Anomalies", "Trend & quality", "Correlation"])
    with t1:
        cc = st.columns(3)
        m = cc[0].selectbox("Metric", [s.metric for s in good], key="cmp_metric")
        wa = cc[1].text_input("Earlier window", "previous 10 minutes", key="cmp_a")
        wb = cc[2].text_input("Later window", "last 10 minutes", key="cmp_b")
        try:
            sid, cands, _ = rt.service.choose_source(m)
            end = max(c["last"] for c in cands if c["source_id"] == sid)
            now = pd.Timestamp.now(tz="UTC").to_pydatetime()
            A, B = parse_window(wa, data_end=end, now=now), parse_window(wb, data_end=end, now=now)
            sa, _, _ = rt.service.get_series(m, A, sid)
            sb, _, _ = rt.service.get_series(m, B, sid)
            res = compare_windows(sa, sb, A.label, B.label)
            st.caption(f"{A.describe()}  vs  {B.describe()}" + ("  · relative to the latest data timestamp" if A.anchor == "latest-data" else ""))
            st.dataframe(pd.DataFrame({A.label: res[A.label], B.label: res[B.label]}))
            p = res["significance"]["p_value"]
            st.markdown(f"Mean difference **{res['mean_difference']:.4g}** ({_fmt(res['mean_change_pct'], '%')}); Mann–Whitney p = {p:.3g}" if p is not None else f"Mean difference **{res['mean_difference']:.4g}**; {res['significance']['note']}")
        except (InsufficientData, WindowError, KeyError, ValueError) as exc:
            st.caption(str(exc))
    with t2:
        rows = [{"metric": k, **a} for k, v in an.items() for a in v]
        st.dataframe(pd.DataFrame(rows), hide_index=True) if rows else st.caption("No anomalies (robust z ≥ 3.5) in the selected window.")
    with t3:
        rows = []
        for s in good:
            tr = trend(s) if s.n >= 3 else {}
            q = data_quality(s)
            rows.append({"metric": s.metric, "status": s.status, "n": s.n, "trend": tr.get("direction"), "slope": tr.get("slope"), "slope unit": tr.get("slope_unit"), "Kendall p": tr.get("p_value"), "R²": tr.get("r_squared"), "quality": q["score"], "completeness": q.get("completeness")})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    with t4:
        if len(good) < 2:
            st.caption("Select at least two metrics.")
        else:
            rows = []
            for i, a in enumerate(good):
                for b in good[i + 1 :]:
                    c = correlation(a, b)
                    rows.append({"A": a.metric, "B": b.metric, "pairs": c["paired_samples"], "Pearson r": c["pearson_r"], "p": c["pearson_p"], "Spearman ρ": c["spearman_rho"], "note": c.get("note", "")})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("Correlation shows association only, not cause.")
