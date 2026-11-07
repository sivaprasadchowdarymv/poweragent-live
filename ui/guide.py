"""In-app guide: live checklist (share → AI reading → ask) plus the full how-to."""
from __future__ import annotations

import streamlit as st

from core.runtime import Runtime

QUESTIONS = {
    "Understand the screen": ["What is on my screen?", "Explain each chart on the screen.", "What are the key values right now?"],
    "Inferences": ["What are the inferences from these charts?", "Is anything abnormal or out of limits?",
                   "Is the trend rising or falling, and what does it mean?"],
    "Change over time": ["What changed in the last 10 minutes?", "Which value is changing fastest?"],
    "Action": ["What should I check next?", "What do you recommend?", "Did your suggestion work?"],
}


def status(rt: Runtime) -> dict:
    f = rt.latest_frame()
    vn = rt.service.vision_notes
    return {"providers": rt.router.enabled and rt.router.has_role("vision"), "frame": f is not None,
            "sharing": bool(st.session_state.get("bc_last_seq")) or rt.capture.state == "running",
            "reading": bool(vn), "frame_info": f"{f.target} · {f.captured_at.strftime('%H:%M:%S')} UTC" if f else ""}


def checklist(rt: Runtime, compact: bool = False) -> None:
    s = status(rt)
    steps = [
        (s["providers"], "AI models ready", "Keys set (Groq + OpenRouter). If not: add them to .env and restart."),
        (s["frame"], "Share your dashboard",
         "Sidebar → **🖥️ Share a tab** → pick the Chrome tab / window / entire screen → **Share**. "
         "The sidebar shows *Sharing · frame N sent*. (Or attach an image with 📎 in the chat box.)"
         + (f" ✅ Latest frame: {s['frame_info']}" if s["frame"] else "")),
        (s["reading"], "Let the AI read it",
         "**Live Monitor → 🧠 Read the screen with AI now** (10-40 s). Afterwards it re-reads changed screens about once a minute."),
        (None, "Ask questions here in Chat", "Use the buttons below or type in the box at the bottom. "
         "Look for 🖼 in the answer's info line: it means the AI looked at your screen."),
    ]
    with st.container(border=True):
        st.markdown("**How to use** - share → AI reads → ask")
        for i, (ok, title, how) in enumerate(steps, 1):
            icon = "✅" if ok else ("⬜" if ok is False else "💬")
            if compact and ok:
                st.markdown(f"{icon} **{i}. {title}**")
            else:
                st.markdown(f"{icon} **{i}. {title}** - {how}")


def question_buttons(prefix: str = "gq") -> None:
    for group, qs in QUESTIONS.items():
        st.caption(group)
        cols = st.columns(len(qs))
        for i, q in enumerate(qs):
            if cols[i].button(q, key=f"{prefix}-{group}-{i}", width="stretch"):
                st.session_state.pending_q = q
                st.switch_page(st.session_state.get("chat_page"))


def guide_page(rt: Runtime) -> None:
    st.markdown("## 📘 How to use PowerAgent Live")
    checklist(rt)
    st.markdown("""
### 1 · Open what you want analysed
Open your dashboard / chart / web page in a **separate browser tab or window** (Grafana, SCADA, power-quality tool, Excel chart…).

### 2 · Share it
1. In the **left sidebar** under **🌐 Live screen share** click **🖥️ Share a tab**.
2. In the browser popup choose:
   - **Chrome tab** - best for web dashboards (sharing continues when you come back here),
   - **Window** - for desktop programs (don't minimise it),
   - **Entire screen** - everything on the monitor.
3. Click **Share**. The sidebar shows a preview and *Sharing · frame N sent* (the number keeps rising).

### 3 · Let the AI read the screen
**Live Monitor → 🧠 Read the screen with AI now.** The panel *AI reading of the screen* shows a summary, findings
(🔴 critical · 🟠 warning · 🔵 info), each chart line with its trend and values, and the KPIs.
After that it re-reads automatically about once a minute when the screen changes, so a history builds up.

### 4 · Ask questions (Chat)
Type in the box at the bottom or click a question below. Answers are structured as *What the screen shows ·
Inferences · Abnormal / attention · What to check next*. **≈ Partly verified** is normal for image answers
(values read from pictures are estimates). Click **Evidence and agent activity** to see exactly what the AI used.

### 5 · Stop
Sidebar **■ Stop sharing**, or *Stop sharing* on the browser's bar.

**Picture instead of live screen:** in Chat click **📎**, choose an image, ask *"What are the inferences from this image?"*.
""")
    st.markdown("### Example questions (click to ask)")
    question_buttons("guide")
    st.markdown("""
### If something goes wrong
| What you see | What to do |
|---|---|
| No *Share a tab* button | Open the sidebar (**>>** top-left). |
| Share popup doesn't open | Use Chrome/Edge on a computer; allow screen sharing in the address bar. |
| Frame number not rising | The shared window was minimised/closed - restore it or share again. |
| "rate_limit / busy" | Free models are busy - wait 1-2 min, press *Read now* again (the next free model is tried automatically). |
| No 🖼 in the answer's info line | Open *Evidence and agent activity* and read the **Route:** line for the reason. |
| Vague answer | Share the specific tab (not the whole screen) and zoom the dashboard so charts are large. |
""")
