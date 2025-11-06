# -----------------------------------------------------------
# Typing Posture Alert (Streamlit)
# Safe boot even if webcam libs aren't available
# -----------------------------------------------------------
import sys, platform, streamlit as st

st.set_page_config(page_title="Typing Posture Alert", page_icon="🧍", layout="wide")
st.success(f"✅ App booted • Python {sys.version.split()[0]} • {platform.system()}")

ENABLE_WEBRTC = True  # flip to False to disable camera quickly

def webcam_deps_ok():
    if not ENABLE_WEBRTC:
        return False, "Webcam disabled by config.", None
    try:
        import av  # noqa: F401
        import cv2  # noqa: F401
        import mediapipe as mp  # noqa: F401
        from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase  # noqa: F401
        return True, "OK", dict()
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", None

have_cam, cam_msg, _ = webcam_deps_ok()
if not have_cam:
    st.warning(
        "⚠️ Webcam not available: " + cam_msg +
        "\nTip: use `opencv-contrib-python-headless` (not `opencv-python`) to avoid `libGL.so.1`."
    )

# ------------------------------------------------------------------
# Only import heavy libs if camera stack is available
# ------------------------------------------------------------------
if have_cam:
    import time, collections
    from dataclasses import dataclass, field
    from typing import Deque, Dict, Tuple

    import av
    import cv2
    import numpy as np
    import mediapipe as mp
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
else:
    # Light imports for "no-camera" mode
    import time, collections
    from dataclasses import dataclass, field
    from typing import Deque, Dict, Tuple
    import numpy as np

# ------------------- App state & helpers (works in both modes) -------------------
NECK_TILT_MAX_DEFAULT = 25.0
BACK_ANGLE_MIN_DEFAULT = 150.0
HEAD_PITCH_MAX_DEFAULT = 20.0
SMOOTH_WINDOW = 30
BAD_POSTURE_PERSIST_SEC = 60.0
MOVE_BODY_INTERVAL_SEC = 30 * 60
MOBILE_MAX_WIDTH = 720

def ss_get(k, v):
    if k not in st.session_state:
        st.session_state[k] = v
    return st.session_state[k]

def ss_set(k, v): st.session_state[k] = v

ss_get("consented", False)
ss_get("calibrated", False)
ss_get("baseline_angles", {"neck": None, "back": None, "head": None})
ss_get("thresholds", {
    "NECK_TILT_MAX": NECK_TILT_MAX_DEFAULT,
    "BACK_ANGLE_MIN": BACK_ANGLE_MIN_DEFAULT,
    "HEAD_PITCH_MAX": HEAD_PITCH_MAX_DEFAULT,
})
ss_get("stats", {
    "start_ts": None, "last_good_ts": None, "last_move_reminder": None,
    "bad_streak_start": None, "timeline": [], "good_frames": 0, "bad_frames": 0,
})
ss_get("show_landmarks", True)
ss_get("mirror_video", True)
ss_get("sensitivity", 1.0)

from dataclasses import dataclass, field  # repeated import OK
@dataclass
class PostureState:
    hist: Deque[int] = field(default_factory=lambda: collections.deque(maxlen=SMOOTH_WINDOW))
    last_frame_ts: float = 0.0

# ----------------------------- UI sections -----------------------------
def consent_gate():
    st.title("🧍 Typing Posture Alert")
    st.markdown(
        "**This app uses your webcam in your browser** to analyze your posture locally.\n\n"
        "Please confirm consent to proceed."
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
            if st.button("Mark as calibrated (demo)"):
                ss_set("baseline_angles", {"neck": 8.0, "back": 165.0, "head": 10.0})
                ss_set("thresholds", {"NECK_TILT_MAX": 18.0, "BACK_ANGLE_MIN": 155.0, "HEAD_PITCH_MAX": 15.0})
                ss_set("calibrated", True)
        else:
            st.success("Calibrated ✅")
            st.json({"baseline": st.session_state["baseline_angles"],
                     "thresholds": st.session_state["thresholds"]})
            if st.button("Re-calibrate"):
                ss_set("calibrated", False)

        st.divider()
        st.subheader("Privacy")
        st.caption("• Video stays in your browser session.\n• No data stored.\n• Close the tab to end.")

def monitor_section():
    st.header("Live Monitor")
    colL, colR = st.columns([2, 1], gap="large")

    with colL:
        st.toggle("Mirror video", value=st.session_state["mirror_video"], key="mirror_video")
        st.toggle("Show landmarks", value=st.session_state["show_landmarks"], key="show_landmarks")
        st.slider("Sensitivity (higher = stricter)", 0.7, 1.5, key="sensitivity", value=1.0, step=0.05)

        if have_cam:
            # Minimal camera hook; add your VideoProcessor later
            webrtc_streamer(
                key="posture-monitor",
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={"video": True, "audio": False},
            )
        else:
            with st.expander("Webcam unavailable (running in demo mode)", expanded=True):
                st.warning("Camera stack not loaded. Once dependencies resolve, the webcam section will appear here.")
                st.code(
                    "from streamlit_webrtc import webrtc_streamer, WebRtcMode\n"
                    "webrtc_streamer(key='posture-monitor', mode=WebRtcMode.SENDRECV,\n"
                    "               media_stream_constraints={'video': True, 'audio': False})",
                    language="python"
                )

    with colR:
        st.subheader("Analytics")
        stats = st.session_state["stats"]
        if not stats["timeline"]:
            t0 = time.time()
            stats["start_ts"] = t0
            stats["timeline"] = [(0.0, 1), (5.0, 1), (10.0, 0), (15.0, 1)]
            stats["good_frames"] = 120
            stats["bad_frames"] = 30

        total = max(1, stats["good_frames"] + stats["bad_frames"])
        pct = 100.0 * stats["good_frames"] / total
        st.metric("Posture % Good", f"{pct:.1f}%")

        import pandas as pd
        df = pd.DataFrame(stats["timeline"], columns=["t_sec", "good"])
        df["minutes"] = df["t_sec"] / 60.0
        df.set_index("minutes", inplace=True)
        st.line_chart(df["good"], height=160, use_container_width=True)
        st.caption("Timeline: 1 = good posture, 0 = slouching.")

        st.divider()
        if st.button("Reset analytics"):
            ss_set("stats", {
                "start_ts": None, "last_good_ts": None, "last_move_reminder": None,
                "bad_streak_start": None, "timeline": [], "good_frames": 0, "bad_frames": 0,
            })
            st.toast("Analytics reset.", icon="🔄")
            st.rerun()

def main():
    st.markdown("<style>.block-container{max-width:720px}</style>", unsafe_allow_html=True)
    if not st.session_state["consented"]:
        consent_gate()
        return
    st.caption("UI loaded. Webcam will appear when dependencies are available.")
    sidebar_info()
    monitor_section()

if __name__ == "__main__":
    main()
