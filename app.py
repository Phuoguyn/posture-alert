# -----------------------------------------------------------
# Typing Posture Alert (Streamlit) - Minimal working posture detector
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
# Imports
# ------------------------------------------------------------------
import time, collections
from dataclasses import dataclass, field
from typing import Deque

import numpy as np

if have_cam:
    import av
    import cv2
    import mediapipe as mp
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# ------------------- App state & helpers -------------------
NECK_TILT_MAX_DEFAULT = 25.0         # degrees
BACK_ANGLE_MIN_DEFAULT = 150.0       # degrees (closer to 180 = straighter)
HEAD_PITCH_MAX_DEFAULT = 20.0        # degrees
SMOOTH_WINDOW = 30
BAD_POSTURE_PERSIST_SEC = 60.0
MOBILE_MAX_WIDTH = 720

def ss_get(k, v):
    if k not in st.session_state:
        st.session_state[k] = v
    return st.session_state[k]

def ss_set(k, v):
    st.session_state[k] = v

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

@dataclass
class PostureState:
    hist: Deque[int] = field(default_factory=lambda: collections.deque(maxlen=SMOOTH_WINDOW))
    last_frame_ts: float = 0.0

# ------------- posture math helpers (simple but effective) -------------
def angle_between(v1, v2):
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def update_stats(is_good: bool):
    stats = st.session_state["stats"]
    now = time.time()
    if stats["start_ts"] is None:
        stats["start_ts"] = now

    t = now - stats["start_ts"]
    stats["timeline"].append((t, 1 if is_good else 0))

    if is_good:
        stats["good_frames"] += 1
        stats["last_good_ts"] = now
        stats["bad_streak_start"] = None
    else:
        stats["bad_frames"] += 1
        if stats["bad_streak_start"] is None:
            stats["bad_streak_start"] = now

# ------------------------- Video Processor -------------------------
if have_cam:
    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.state = PostureState()
            self.pose = mp.solutions.pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.draw = mp.solutions.drawing_utils
            self.styles = mp.solutions.drawing_styles

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            # Mirror if enabled
            if st.session_state.get("mirror_video", True):
                img = cv2.flip(img, 1)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)

            is_good = True  # default assume ok

            if results.pose_landmarks:
                h, w, _ = img.shape
                lm = results.pose_landmarks.landmark

                # Key points (using left side; mirrored already if needed)
                def pt(i):
                    return np.array([lm[i].x * w, lm[i].y * h])

                shoulder = pt(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER)
                hip = pt(mp.solutions.pose.PoseLandmark.LEFT_HIP)
                ear = pt(mp.solutions.pose.PoseLandmark.LEFT_EAR)
                nose = pt(mp.solutions.pose.PoseLandmark.NOSE)

                # Back angle: hip -> shoulder vs vertical
                vertical = np.array([0, -1])
                back_vec = shoulder - hip
                back_angle = 180.0 - angle_between(back_vec, vertical)  # ~180 when straight

                # Neck tilt: shoulder -> ear vs vertical
                neck_vec = ear - shoulder
                neck_tilt = angle_between(neck_vec, vertical)

                # Head pitch (rough): shoulder -> nose vs vertical
                head_vec = nose - shoulder
                head_pitch = angle_between(head_vec, vertical)

                th = st.session_state["thresholds"]
                sens = float(st.session_state.get("sensitivity", 1.0))

                neck_ok = neck_tilt <= th["NECK_TILT_MAX"] / sens
                back_ok = back_angle >= th["BACK_ANGLE_MIN"] * sens / 180.0 * 180.0
                head_ok = head_pitch <= th["HEAD_PITCH_MAX"] / sens

                is_good = neck_ok and back_ok and head_ok

                # smooth
                self.state.hist.append(1 if is_good else 0)
                smoothed = 1 if (sum(self.state.hist) / max(1, len(self.state.hist))) >= 0.5 else 0
                is_good = bool(smoothed)

                # Draw landmarks & a simple status text
                if st.session_state.get("show_landmarks", True):
                    self.draw.draw_landmarks(
                        img,
                        results.pose_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.styles.get_default_pose_landmarks_style(),
                    )

                status_text = "GOOD" if is_good else "SLOUCH"
                color = (0, 180, 0) if is_good else (0, 0, 255)
                cv2.putText(img, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Update global stats (safe enough for this simple app)
            update_stats(is_good)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

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
                ss_set("thresholds", {
                    "NECK_TILT_MAX": 18.0,
                    "BACK_ANGLE_MIN": 155.0,
                    "HEAD_PITCH_MAX": 15.0
                })
                ss_set("calibrated", True)
        else:
            st.success("Calibrated ✅")
            st.json({
                "baseline": st.session_state["baseline_angles"],
                "thresholds": st.session_state["thresholds"],
            })
            if st.button("Re-calibrate"):
                ss_set("calibrated", False)

        st.divider()
        st.subheader("Privacy")
        st.caption("• Video stays in your browser session.\n• No data stored.\n• Close the tab to end.")

def monitor_section():
    st.header("Live Monitor")
    colL, colR = st.columns([2, 1], gap="large")

    with colL:
        st.toggle("Mirror video", key="mirror_video")
        st.toggle("Show landmarks", key="show_landmarks")
        st.slider("Sensitivity (higher = stricter)", 0.7, 1.5, key="sensitivity", step=0.05)

        if have_cam:
            webrtc_streamer(
                key="posture-monitor",
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=VideoProcessor,   # 🔴 THIS MAKES DETECTION RUN
            )
        else:
            with st.expander("Webcam unavailable (running in demo mode)", expanded=True):
                st.warning("Camera stack not loaded. Once dependencies resolve, the webcam section will appear here.")

    with colR:
        st.subheader("Analytics")
        stats = st.session_state["stats"]

        # Basic posture %
        total = max(1, stats["good_frames"] + stats["bad_frames"])
        pct = 100.0 * stats["good_frames"] / total
        st.metric("Posture % Good", f"{pct:.1f}%")

        # Plot timeline if any
        import pandas as pd
        if stats["timeline"]:
            df = pd.DataFrame(stats["timeline"], columns=["t_sec", "good"])
            df["minutes"] = df["t_sec"] / 60.0
            df.set_index("minutes", inplace=True)
            st.line_chart(df["good"], height=160, use_container_width=True)
            st.caption("Timeline: 1 = good posture, 0 = slouching.")

        # Alert if bad posture persists
        if stats["bad_streak_start"] is not None:
            bad_secs = time.time() - stats["bad_streak_start"]
            if bad_secs >= BAD_POSTURE_PERSIST_SEC:
                st.error("⚠️ You've had poor posture for a while. Time to straighten up and take a break!")

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
    st.caption("UI loaded. Webcam will appear once dependencies and permissions are OK.")
    sidebar_info()
    monitor_section()

if __name__ == "__main__":
    main()
