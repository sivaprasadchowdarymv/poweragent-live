"""Live Monitor: target selection, capture controls, preview and extraction results."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from PIL import ImageDraw

from capture.screen import CaptureError, CaptureTarget, Region, list_monitors, list_windows
from core.runtime import Runtime
from core.services import SCREEN_SOURCE_ID


def _target_picker(rt: Runtime) -> CaptureTarget | None:
    monitors = list_monitors()
    windows = list_windows()
    if not monitors:
        st.error("No display detected. Run the app in your logged-in desktop session (not as a service), keep Remote Desktop sessions open, and on macOS grant Screen Recording permission.")
        return None
    kinds = ["Monitor", "Region"] + (["Window"] if windows else [])
    kind = st.segmented_control("Capture target", kinds, default="Monitor", key="target_kind") or "Monitor"
    if kind == "Window":
        title = st.selectbox("Application window", [w.title for w in windows], key="target_window")
        return CaptureTarget("window", window_title=title)
    idx = st.selectbox(
        "Monitor",
        list(range(1, len(monitors) + 1)),
        format_func=lambda i: f"Monitor {i}: {monitors[i - 1].width}×{monitors[i - 1].height} at ({monitors[i - 1].left},{monitors[i - 1].top})",
        key="target_monitor",
    )
    mon = monitors[idx - 1]
    if kind == "Monitor":
        return CaptureTarget("monitor", monitor_index=idx)
    st.caption("Drag the sliders to crop the monitor to the dashboard area; the preview shows the selection.")
    c1, c2 = st.columns(2)
    x0, x1 = c1.slider("Horizontal range (%)", 0, 100, (0, 100), key="reg_x")
    y0, y1 = c2.slider("Vertical range (%)", 0, 100, (0, 100), key="reg_y")
    if x1 - x0 < 2 or y1 - y0 < 2:
        st.warning("The region is too small.")
        return None
    region = Region(
        mon.left + int(mon.width * x0 / 100),
        mon.top + int(mon.height * y0 / 100),
        max(1, int(mon.width * (x1 - x0) / 100)),
        max(1, int(mon.height * (y1 - y0) / 100)),
    )
    st.caption(f"Region: {region.width}×{region.height} px at ({region.left}, {region.top})")
    return CaptureTarget("region", monitor_index=idx, region=region)


def _local_capture_available() -> bool:
    try:
        return bool(list_monitors())
    except Exception:  # noqa: BLE001 - no display (cloud server, container)
        return False


def monitor_tab(rt: Runtime) -> None:
    st.markdown("##### 🌐 Share from this browser (works locally and on the hosted app)")
    c = st.columns([2, 1])
    c[0].caption("Click **Share a tab** in the sidebar and pick the tab/window with your dashboard (Chrome or Edge). "
                 "A frame is sent every few seconds and read with OCR + the chart reader; sharing keeps running while you chat.")
    c[1].number_input("Send a frame every (s)", 1.0, 60.0, float(st.session_state.get("bc_interval", 2.0)), 1.0, key="bc_interval")
    if not _local_capture_available():
        st.info("This server has no display (hosted app), so only browser sharing is available.")
        live_view(rt)
        return
    st.markdown("##### 🖥️ Capture this computer (local run)")
    target = _target_picker(rt)
    c = st.columns([1, 1, 1, 1])
    interval = c[0].number_input("Capture every (s)", min_value=0.25, max_value=600.0, value=float(rt.capture.interval_s), step=0.5, key="cap_interval", help="Target interval. Measured intervals are shown below and in Diagnostics.")
    if target is not None:
        rt.capture.configure(target=target, interval_s=float(interval))
    state = rt.capture.state
    b = st.container(horizontal=True)
    if state == "stopped":
        if b.button("▶ Start monitoring", type="primary", disabled=target is None, key="mon_start"):
            rt.start_monitoring()
            st.rerun()
    else:
        if state == "running" and b.button("⏸ Pause", key="mon_pause"):
            rt.capture.pause()
            st.rerun()
        if state == "paused" and b.button("▶ Resume", type="primary", key="mon_resume"):
            rt.capture.resume()
            st.rerun()
        if b.button("■ Stop", key="mon_stop"):
            rt.stop_monitoring()
            st.rerun()
    if b.button("📸 Capture & analyze now", disabled=target is None, key="mon_now"):
        with st.spinner("Capturing and reading the screen…"):
            try:
                a = rt.manual_capture()
                st.toast(f"Analyzed frame {a.frame_seq}: {len(a.charts)} chart(s), {len(a.kpis)} KPI(s).")
            except CaptureError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Capture failed: {exc}")
    st.caption("Captured images stay in memory on this computer and are not saved unless you opt in under Settings. Avoid capturing passwords or private content.")
    live_view(rt)


@st.fragment(run_every=2.0)
def live_view(rt: Runtime) -> None:
    frame = rt.latest_frame()
    analysis = rt.service.latest_screen
    if frame is None:
        st.info("No capture yet. Start monitoring or press **Capture & analyze now**.")
        return
    img = frame.image.copy()
    if analysis is not None and analysis.frame_seq == frame.seq:
        draw = ImageDraw.Draw(img)
        for ch in analysis.charts:
            draw.rectangle(ch.bbox, outline=(42, 120, 214), width=max(2, img.width // 400))
            draw.text((ch.bbox[0] + 4, ch.bbox[1] + 4), ch.id, fill=(42, 120, 214))
        for k in analysis.kpis:
            draw.rectangle(k.bbox, outline=(235, 104, 52), width=2)
    ai_reading_panel(rt)
    preview = img.copy()
    preview.thumbnail((1100, 1100))
    st.image(preview, caption=f"Frame {frame.seq} · {frame.target} · captured {frame.captured_at.strftime('%H:%M:%S')} UTC · blue = detected plot areas, orange = KPI values", width="stretch")
    m = rt.capture.metrics.summary()
    cols = st.columns(5)
    cols[0].metric("Captures", m["captures"])
    cols[1].metric("Duplicates skipped", m["duplicates_skipped"])
    cols[2].metric("Dropped frames", m["dropped_frames"])
    cols[3].metric("Measured interval", f"{m['interval_mean_s']:.2f} s" if m["interval_mean_s"] else "—", help=f"p95 {m['interval_p95_s']:.2f} s" if m["interval_p95_s"] else None)
    em = rt.extraction.metrics.summary()
    cols[4].metric("Extraction", f"{em['ocr_ms']['mean'] + em['chart_ms']['mean']:.0f} ms" if em["ocr_ms"]["mean"] else "—", help=f"Effective extraction interval {em['effective_interval_s']:.1f} s")
    if m["last_error"]:
        st.warning(f"Last capture error: {m['last_error']}")
    if analysis is None:
        return
    age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(analysis.captured_at)).total_seconds()
    st.markdown(f"**Latest extraction** — frame {analysis.frame_seq}, {age:.0f} s ago. All values below are *estimates* from pixels/OCR.")
    for w in analysis.warnings:
        st.caption(f"⚠ {w}")
    rows = []
    for ch in analysis.charts:
        for s in ch.series:
            rows.append({"chart": f"{ch.id} · {ch.title or 'untitled'}", "type": ch.chart_type, "series": s.name, "latest ≈": s.latest, "min ≈": s.minimum, "max ≈": s.maximum, "unit": ch.unit, "y-axis": f"{ch.y_axis.scale}{' reversed' if ch.y_axis.reversed else ''}", "confidence": round(s.confidence, 2), "clipped": s.clipped})
        if not ch.series:
            rows.append({"chart": f"{ch.id} · {ch.title or 'untitled'}", "type": ch.chart_type, "series": "—", "confidence": round(ch.confidence, 2)})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={"confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=1, format="%.2f")})
    if analysis.kpis:
        st.dataframe(pd.DataFrame([{"KPI": k.label, "value ≈": k.value, "unit": k.unit, "confidence": round(k.confidence, 2)} for k in analysis.kpis]), hide_index=True)
    if not rows and not analysis.kpis:
        st.caption("No charts or KPI values were recognized in this frame. Try cropping to the dashboard region or enabling the vision model in Settings.")
    chart_model_panel(rt, analysis)
    with st.expander("Correct an estimate"):
        correction_form(rt)


def ai_reading_panel(rt: Runtime) -> None:
    """The vision model's latest inferences about the screen (vision-first)."""
    vn = rt.service.vision_notes
    st.markdown("##### 🧠 AI reading of the screen")
    if rt.vision_last_error:
        st.caption(f"⚠ Last vision call: {rt.vision_last_error}")
    if not vn:
        st.caption("No vision reading yet. It runs on changed frames every "
                   f"{rt.settings.vision_interval_s:.0f} s (Settings → background vision), or press the button.")
    else:
        icon = {"critical": "🔴", "warning": "🟠"}
        st.markdown(f"**{vn.get('summary', '')}**")
        for f in vn.get("findings", []):
            st.markdown(f"{icon.get(f['severity'], '🔵')} {f['finding']} — *{f['evidence']}*")
        if vn.get("assessment"):
            st.info(vn["assessment"])
        rows = [{"chart": c["title"], "series": s_["name"], "trend": s_["trend"], "latest": s_["latest_value"],
                 "min": s_["min_value"], "max": s_["max_value"], "notes": s_["notes"]} for c in vn.get("charts", []) for s_ in c["series"]]
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        if vn.get("kpis"):
            st.dataframe(pd.DataFrame(vn["kpis"]), hide_index=True)
        st.caption(f"Model {vn.get('model', '')} · frame {vn.get('frame_seq')} · {str(vn.get('captured_at', ''))[11:19]} UTC · estimates read from the image")
    if st.button("🧠 Read the screen with AI now", key="ai_read_now"):
        f = rt.latest_frame()
        if f is None:
            st.warning("No frame yet.")
        else:
            from llm.providers import LLMError
            from vision.vision_model import describe_screen

            with st.spinner("The vision model is reading the screen…"):
                try:
                    rt.service.set_vision_notes(describe_screen(rt.router, f.image, 1280), f.seq, f.captured_at)
                    rt.vision_last_error = ""
                    st.rerun()
                except LLMError as exc:
                    st.error(f"{exc.user_message} · tried: {getattr(exc, 'attempts', [])}")


def chart_model_panel(rt: Runtime, analysis) -> None:  # type: ignore[no-untyped-def]
    if not rt.settings.chart_model_enabled and not rt.service.chart_model:
        return
    data = rt.service.chart_model
    busy = rt.extraction.chart_model_busy
    title = "Chart-model readings" + (" · reading…" if busy else "") + (f" · frame {data['frame_seq']}" if data else "")
    with st.expander(title, expanded=bool(data)):
        if st.button("Read charts with the chart model now", disabled=busy or not rt.settings.chart_model_enabled, key="chart_model_now"):
            ok = rt.chart_model_now()
            st.toast("chart-model reading started." if ok else "Nothing to read yet — capture first.")
        if not data:
            st.caption(f"No chart-model reading yet. It runs every {rt.settings.chart_model_interval_s:g} s while monitoring, or press the button.")
            return
        import pandas as pd

        for r in data["readings"]:
            cc = r["cross_check_with_pixel_reader"]
            icon = {"agree": "✅", "partial": "🟡", "disagree": "⚠️"}.get(cc["status"], "➖")
            st.markdown(f"**{r['chart_title'] or r['chart_id']}** ({r.get('model', 'chart model')}) — {icon} cross-check with pixel reader: **{cc['status']}**" + (f" ({cc['agreement_pct']}% of {cc['compared_points']} points within 3% of the axis span; mean error {cc['error_pct_of_axis']}% of span)" if cc.get("compared_points") else "") + f" · {r['inference_ms'] / 1000:.1f} s")
            if r.get("error"):
                st.caption(f"⚠ {r['error']}")
            tbl = r.get("table")
            if tbl and tbl["series"]:
                frame = pd.DataFrame({k: pd.Series({p["x"]: p["value"] for p in v}) for k, v in tbl["series"].items()})
                st.dataframe(frame, height=min(300, 38 + 35 * len(frame)))
        st.caption("Chart-model values are estimates read by a model from the image; they are stored under the source `screen_chart_model` and never merged with other sources.")


def correction_form(rt: Runtime) -> None:
    df = rt.db.query_observations(source_id=SCREEN_SOURCE_ID, limit=300)
    if df.empty:
        st.caption("No screen estimates stored yet.")
        return
    latest = df.sort_values("timestamp").groupby("metric").tail(1)
    options = {f"{r.metric} = {r.value:.6g} {r.unit} at {r.timestamp:%H:%M:%S}": int(r.id) for r in latest.itertuples()}
    with st.form("correct_form", clear_on_submit=True):
        choice = st.selectbox("Estimate", list(options))
        value = st.number_input("Correct value", value=float(latest.iloc[0]["value"]), format="%.6g")
        note = st.text_input("Note (optional)", max_chars=200)
        if st.form_submit_button("Save correction"):
            rt.db.correct_observation(options[choice], value, note)
            st.success("Saved. The value is now marked as 'corrected' and used with full confidence.")
