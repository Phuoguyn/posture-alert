# app.py — Phase 3: full UI (webcam still disabled)

import sys, platform, time, collections, importlib
from dataclasses import dataclass, field
from typing import Deque, Dict, Tuple

import numpy as np
import streamlit as st

# ---------- Page ----------
st.set_page_config(page_title="Typing Posture Alert", page_icon="🧍", layout="wide")
st.success(f"✅ App booted • Python {sys.version.split()[0]} • {platform.system()}")

# ---------- Helper: show installed versions ----------
def v(name, attr="__version__"):
    try:
        m = importlib.import_module(name)
        return getattr(m, attr, "(no __version__)")
    except Exception as e:
        return f"❌ {type(e).__name__}"

with st.expander("📦 Dependency check"):
    st.table({
        "package": [
            "streamlit", "streamlit-webrtc", "mediapipe", "opencv-python-headless (cv2)",
            "av", "aiortc", "numpy", "pandas", "pyarrow", "protobuf"
        ],
        "version / status": [
            v("streamlit"), v("streamlit_webrtc"), v("mediapipe"), v("cv2"),
            v("av"), v("aiortc"), v("numpy"), v("pandas"), v("pyarrow"), v("google.protobuf", "__version__")
        ],
    })

# ---------- Defaults & session ----------
NECK_TILT_MAX_DEFAULT = 25.0
BACK_ANGLE_MIN_DEFAULT = 150.0
HEAD_PITCH_MAX_DEFAULT = 20.0

SMOOTH_WINDOW = 30
BAD_POSTURE_PERSIST_SEC = 60.0
MOVE_BODY_INTERVAL_SEC = 30 * 60
MOBILE_MAX_WIDTH = 720

def ss_get(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

def ss_set(key, val):
    st.session_state[key] = val

ss_get("consented", False)
ss_get("calibrated", False)
ss_get("baseline_angles", {"neck": None, "back": None, "head": None})
ss_get("thresholds", {
    "NECK_TILT_MAX": NECK_TILT_MAX_DEFAULT,
    "BACK_ANGLE_MIN": BACK_ANGLE_MIN_DEFAULT,
    "HEAD_PITCH_MAX": HEAD_PITCH_MAX_DEFAULT,
})
ss_get("stats", {
    "start_ts": None,
    "last_good_ts": None,
    "last_move_reminder": None,
    "bad_streak_start": None,
    "timeline": [],
    "good_frames": 0,
    "bad_frames": 0,
})
ss_get("show_landmarks", True)
ss_get("mirror_video", True)
ss_get("sensitivity", 1.0)

# ---------- Fake processor placeholders (so UI works without webcam) ----------
@dataclass
class PostureState:
    hist: Deque[int] = field(default_factory=lambda: collections.deque(maxlen=SMOOTH_WINDOW))
    last_frame_ts: float = 0.0

# ---------- UI sections ----------
def consent_gate():
    st.title("🧍 Typing Posture Alert")
    st.markdown(
        "**This app uses your webcam in your browser** to analyze your posture locally.\n\n"
        "Please confirm consent below to proceed."
    )
    agree = st.checkbox("I consent to webcam access and local posture analysis.")
    if st.button("Continue", type="primary", disabled=not agree):
        ss_set("consented", True)
        st.rerun()

def sidebar_info():
    with st.sidebar:
        st.header("Calibration")
        if not st.session_state["calibrated"]:
            st.info("Calibration sets your personal baseline angles.")
            if st.button("Mark as calibrated (demo only)"):
                ss_set("baseline_angles", {"neck": 8.0, "back": 165.0, "head": 10.0})
                ss_set("thresholds", {
                    "NECK_TILT_MAX": 18.0,
                    "BACK_ANGLE_MIN": 155.0,
                    "HEAD_PITCH_MAX": 15.0,
                })
                ss_set("calibrated", True)
        else:
            st.success("Calibrated ✅")
            st.write("**Baseline angles (demo):**")
            st.json(st.session_state["baseline_angles"])
            st.write("**Personalized thresholds (demo):**")
            st.json(st.session_state["thresholds"])
            if st.button("Re-calibrate"):
                ss_set("calibrated", False)

        st.divider()
        st.subheader("Privacy")
        st.caption("• Video stays in your browser session.\n"
                   "• No data is stored.\n"
                   "• Close the tab to end the session.")

def monitor_section():
    st.header("Live Monitor")

    colL, colR = st.columns([2, 1], gap="large")
    with colL:
        st.toggle("Mirror video", value=st.session_state["mirror_video"], key="mirror_video")
        st.toggle("Show landmarks", value=st.session_state["show_landmarks"], key="show_landmarks")
        st.slider("Sensitivity (higher = stricter)", 0.7, 1.5, key="sensitivity", value=1.0, step=0.05)

        # ------- Webcam temporarily disabled in Phase-3 -------
        with st.expander("Webcam (temporarily disabled for smoke test)", expanded=True):
            st.warning(
                "Webcam init is disabled in Phase-3 so the page always renders. "
                "Next step (Phase-4) will turn WebRTC back on."
            )
            st.code(
                "from streamlit_webrtc import webrtc_streamer, WebRtcMode\n"
                "webrtc_streamer(key='posture-monitor', mode=WebRtcMode.SENDRECV,\n"
                "               media_stream_constraints={'video': True, 'audio': False})",
                language="python",
            )

    with colR:
        st.subheader("Analytics (demo data)")
        stats = st.session_state["stats"]

        # create a tiny fake timeline so charts render
        if not stats["timeline"]:
            t0 = time.time()
            stats["start_ts"] = t0
            stats["timeline"] = [(0.0, 1), (5.0, 1), (10.0, 0), (15.0, 1)]
            stats["good_frames"] = 120
            stats["bad_frames"] = 30

        total = max(1, stats["good_frames"] + stats["bad_frames"])
        pct = 100.0 * stats["good_frames"] / total
        st.metric("Posture % Good", f"{pct:.1f}%")
        st.caption("Percentage of frames assessed as 'good' this session (demo).")

        tl = stats["timeline"]
        if len(tl) >= 2:
            import pandas as pd
            df = pd.DataFrame(tl, columns=["t_sec", "good"])
            df["minutes"] = df["t_sec"] / 60.0
            df.set_index("minutes", inplace=True)
            st.line_chart(df["good"], height=160, use_container_width=True)
            st.caption("Timeline: 1 = good posture, 0 = slouching (sampled).")
        else:
            st.info("Timeline will appear after ~10 seconds of monitoring.")

        st.divider()
        if st.button("Reset analytics"):
            ss_set("stats", {
                "start_ts": None,
                "last_good_ts": None,
                "last_move_reminder": None,
                "bad_streak_start": None,
                "timeline": [],
                "good_frames": 0,
                "bad_frames": 0,
            })
            st.toast("Analytics reset.", icon="🔄")
            st.rerun()

# ---------- Main ----------
def main():
    st.markdown(
        f"<style>.block-container {{ max-width: {MOBILE_MAX_WIDTH}px; }}</style>",
        unsafe_allow_html=True,
    )
    if not st.session_state["consented"]:
        consent_gate()
        return
    st.caption("Phase-3: UI loaded (webcam disabled).")
    sidebar_info()
    monitor_section()

if __name__ == "__main__":
    main()
