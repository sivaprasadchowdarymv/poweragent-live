"""Data Sources: CSV/XLSX upload, validation, preview, column mapping, ingestion,
and authoritative-source selection when sources disagree."""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from ingest.files import FileValidationError, list_sheets, load_dataset
from agents.tools import ToolContext, t_compare_sources
from core.runtime import Runtime
from storage.db import metric_key

TIMEZONES = ["UTC", "Asia/Kolkata", "Europe/London", "Europe/Berlin", "America/New_York", "America/Chicago", "America/Los_Angeles", "Asia/Singapore", "Asia/Tokyo", "Australia/Sydney"]


def data_sources_tab(rt: Runtime) -> None:
    st.markdown("Upload CSV or Excel files. Files are read in memory and never modified; only parsed values and a SHA-256 fingerprint are stored.")
    files = st.file_uploader("CSV or XLSX files", type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="uploader")
    cache = st.session_state.upload_cache
    for f in files or []:
        data = f.getvalue()
        digest = hashlib.sha256(data).hexdigest()[:16]
        key = f"{f.name}:{digest}"
        with st.expander(f"📄 {f.name} · {len(data) / 1024:.0f} KB", expanded=key not in cache or not cache[key].get("ingested")):
            sheet: str | int = 0
            if f.name.lower().endswith(".xlsx"):
                sheets = list_sheets(data)
                if len(sheets) > 1:
                    sheet = st.selectbox("Sheet", sheets, key=f"sheet_{key}")
            ck = f"{key}:{sheet}"
            if ck not in cache:
                try:
                    cache[ck] = {"ds": load_dataset(f.name, data, rt.settings.max_upload_mb, rt.settings.max_rows, rt.settings.max_columns, sheet), "ingested": False}
                except FileValidationError as exc:
                    st.error(str(exc))
                    continue
            cache[key] = cache[ck]
            entry = cache[ck]
            ds = entry["ds"]
            prof = ds.profile
            st.caption(f"{prof.rows:,} rows × {prof.columns} columns · {prof.duplicate_rows} duplicate rows · {prof.missing_total} missing cells")
            st.dataframe(ds.raw.head(50), height=220)
            st.dataframe(
                pd.DataFrame([{"column": c.name, "type": c.kind, "missing": c.missing, "non-numeric": c.non_numeric, "unique": c.unique, "unit (from header)": c.unit, "sample": ", ".join(map(str, c.sample))} for c in prof.column_profiles.values()]),
                hide_index=True,
            )
            cols = list(ds.raw.columns)
            ts_opts = ["(no timestamp column — use row order)"] + cols
            default_ts = ts_opts.index(ds.timestamp_col) if ds.timestamp_col in cols else 0
            c1, c2 = st.columns(2)
            ts_choice = c1.selectbox("Timestamp column", ts_opts, index=default_ts, key=f"ts_{ck}")
            tz = c2.selectbox("Timezone of naive timestamps", TIMEZONES, index=0, key=f"tz_{ck}", help="Timestamps without a timezone are interpreted in this zone and stored as UTC.")
            numeric = [c for c in cols if c != ts_choice and prof.column_profiles[c].kind == "numeric"]
            metrics = st.multiselect("Metric columns", [c for c in cols if c != ts_choice], default=[c for c in ds.metric_cols if c in numeric], key=f"m_{ck}")
            units = {}
            if metrics:
                ucols = st.columns(min(4, len(metrics)))
                for i, m in enumerate(metrics):
                    units[m] = ucols[i % len(ucols)].text_input(f"Unit · {m}", value=ds.units.get(m) or prof.column_profiles[m].unit, max_chars=16, key=f"u_{ck}_{m}")
            if st.button("Ingest" if not entry["ingested"] else "Re-ingest with these settings", type="primary", key=f"ing_{ck}", disabled=not metrics):
                ds.timestamp_col = None if ts_choice.startswith("(") else ts_choice
                ds.timezone = tz
                ds.metric_cols = metrics
                ds.units = units
                with st.spinner("Normalizing and storing observations…"):
                    rep = rt.service.ingest_dataset(ds)
                entry["ingested"] = True
                entry["report"] = rep
                st.success(f"Stored {rep['stored']:,} observations · quality score {rep['quality_score']:.2f}")
            rep = entry.get("report")
            if rep:
                issues = []
                if rep["invalid_timestamps"]:
                    issues.append(f"{len(rep['invalid_timestamps'])} row(s) with invalid/missing timestamps (rows {rep['invalid_timestamps'][:8]}…)")
                for c, rows in rep["non_numeric_cells"].items():
                    if rows:
                        issues.append(f"{c}: {len(rows)} non-numeric cell(s) (rows {rows[:8]}…)")
                if rep["duplicate_timestamps"]:
                    issues.append(f"{rep['duplicate_timestamps']} duplicate timestamp(s) — averaged during analysis")
                if rep["unsorted_timestamps"]:
                    issues.append("Timestamps were not in order — sorted during analysis")
                if rep["time_basis"] == "row_index":
                    issues.append("No timestamp column: time-based rates are reported per row")
                for issue in issues:
                    st.caption(f"⚠ {issue}")
    st.divider()
    sources_overview(rt)


def sources_overview(rt: Runtime) -> None:
    st.markdown("##### Stored sources")
    srcs = rt.db.sources()
    if not srcs:
        st.caption("Nothing stored yet.")
        return
    for s in srcs:
        c = st.container(horizontal=True, vertical_alignment="center")
        c.markdown(f"**{s['name']}** `{s['id']}` · {s['type']} · {s['observations']:,} observations · {(s['first'] or '')[:19]} → {(s['last'] or '')[:19]}")
        if c.button("Delete", key=f"del_{s['id']}", type="tertiary"):
            rt.service.remove_dataset(s["id"])
            st.rerun()
    # Metrics available from more than one source → choose authority, show discrepancies
    by_key: dict[str, list[dict]] = {}
    for row in rt.service.metric_catalog():
        by_key.setdefault(row["metric_key"], []).append(row)
    multi = {k: v for k, v in by_key.items() if len({r["source_id"] for r in v}) > 1}
    if multi:
        st.markdown("##### Metrics with more than one source")
        st.caption("Sources are never merged. Choose which source answers questions about each metric.")
        for k, rows in multi.items():
            ids = list(dict.fromkeys(r["source_id"] for r in rows))
            labels = {r["source_id"]: f"{r['source_id']} ({r['status']}, {r['count']} pts)" for r in rows}
            cur = rt.service.preferred_source.get(k)
            choice = st.selectbox(f"Authoritative source for **{rows[0]['metric']}**", ["(automatic: verified first)"] + ids, index=(ids.index(cur) + 1) if cur in ids else 0, format_func=lambda x, _l=labels: _l.get(x, x), key=f"pref_{k}")
            if choice.startswith("("):
                rt.service.preferred_source.pop(k, None)
            else:
                rt.service.preferred_source[metric_key(rows[0]["metric"])] = choice
            res = t_compare_sources(ToolContext(rt.service, rt.alerts), rows[0]["metric"], ids[0], ids[1])
            if res.get("paired_samples"):
                st.caption(f"Agreement {res['agreement_pct']:.1f}% over {res['paired_samples']} time-aligned pairs; mean |difference| {res['mean_abs_difference']:.4g}; {res['discrepancies']} discrepancies beyond tolerance.")
                if res.get("examples"):
                    st.dataframe(pd.DataFrame(res["examples"]), hide_index=True, height=160)
            else:
                st.caption(res.get("note", "No overlapping samples to compare."))
