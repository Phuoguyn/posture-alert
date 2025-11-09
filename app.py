# -----------------------------------------------------------
# Typing Posture Alert (Streamlit) - Stable posture detector
# -----------------------------------------------------------
import sys
import platform
import time
import collections
from dataclasses import dataclass, field
from typing import Deque

import streamlit as st
import numpy as np

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

# Import heavy deps only if available
if have_cam:
    import av
    import cv2
    import mediapipe as mp
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# ------------------- Config & State -------------------
NECK_TILT_MAX_DEFAULT = 25.0        # deg
BACK_ANGLE_MIN_DEFAULT = 150.0      # deg
HEAD_PITCH_MAX_DEFAULT = 20.0       # deg
SMOOTH_WINDOW = 30
BAD_POSTURE_PERSIST_SEC = 60.0

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

# ------------- Helpers -------------
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
            # SUPER IMPORTANT: never crash here or camera dies
            try:
                img = frame.to_ndarray(format="bgr24")

                # Mirror if user chose so (defensive: use get)
                if st.session_state.get("mirror_video", True):
                    img = cv2.flip(img, 1)

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb)

                is_good = True  # default

                if results.pose_landmarks:
                    h, w, _ = img.shape
                    lm = results.pose_landmarks.landmark

                    def pt(i):
                        return np.array([lm[i].x * w, lm[i].y * h])

                    shoulder = pt(mp.solutions.pose.PoseLandmark.LEFT_SHOULDER)
                    hip = pt(mp.solutions.pose.PoseLandmark.LEFT_HIP)
                    ear = pt(mp.solutions.pose.PoseLandmark.LEFT_EAR)
                    nose = pt(mp.solutions.pose.PoseLandmark.NOSE)

                    vertical = np.array([0, -1])

                    back_vec = shoulder - hip
                    neck_vec = ear - shoulder
                    head_vec = nose - shoulder

                    # Back more vertical → better (angle vs vertical)
                    back_angle = 180.0 - angle_between(back_vec, vertical)
                    neck_tilt = angle_between(neck_vec, vertical)
                    head_pitch = angle_between(head_vec, vertical)

                    th = st.session_state.get("thresholds", {
                        "NECK_TILT_MAX": NECK_TILT_MAX_DEFAULT,
                        "BACK_ANGLE_MIN": BACK_ANGLE_MIN_DEFAULT,
                        "HEAD_PITCH_MAX": HEAD_PITCH_MAX_DEFAULT,
                    })
                    sens = float(st.session_state.get("sensitivity", 1.0))

                    neck_ok = neck_tilt <= th["NECK_TILT_MAX"] / sens
                    back_ok = back_angle >= th["BACK_ANGLE_MIN"] * sens / 180.0 * 180.0
                    head_ok = head_pitch <= th["HEAD_PITCH_MAX"] / sens

                    is_good = neck_ok and back_ok and head_ok

                    # smooth: majority of last N frames
                    self.state.hist.append(1 if is_good else 0)
                    avg = sum(self.state.hist) / max(1, len(self.state.hist))
                    is_good = bool(avg >= 0.5)

                    # Draw landmarks
                    if st.session_state.get("show_landmarks", True):
                        self.draw.draw_landmarks(
                            img,
                            results.pose_landmarks,
                            mp.solutions.pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=self.styles.get_default_pose_landmarks_style(),
                        )

                    # Overlay status
                    label = "GOOD" if is_good else "SLOUCH"
                    color = (0, 200, 0) if is_good else (0, 0, 255)
                    cv2.putText(img, label, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

                # Update global stats (wrapped to avoid crash if session_state unavailable)
                try:
                    update_stats(is_good)
                except Exception:
                    pass

                return av.VideoFrame.from_ndarray(img, format="bgr24")

            except Exception:
                # If anything goes wrong, just return original frame so webcam keeps working
                return frame

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
                    "HEAD_PITCH_MAX": 15.0,
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
                video_processor_factory=VideoProcessor,
            )
        else:
            st.warning("Running in demo mode (no webcam dependencies).")

    with colR:
        st.subheader("Analytics")
        stats = st.session_state["stats"]

        total = max(1, stats["good_frames"] + stats["bad_frames"])
        pct = 100.0 * stats["good_frames"] / total
        st.metric("Posture % Good", f"{pct:.1f}%")

        if stats["timeline"]:
            import pandas as pd
            df = pd.DataFrame(stats["timeline"], columns=["t_sec", "good"])
            df["minutes"] = df["t_sec"] / 60.0
            df.set_index("minutes", inplace=True)
            st.line_chart(df["good"], height=160, use_container_width=True)
            st.caption("Timeline: 1 = good posture, 0 = slouching.")

        # Long slouch alert
        if stats["bad_streak_start"] is not None:
            bad_secs = time.time() - stats["bad_streak_start"]
            if bad_secs >= BAD_POSTURE_PERSIST_SEC:
                st.error("⚠️ You've had poor posture for a while. Straighten up and take a quick break!")

        if st.button("Reset analytics"):
            ss_set("stats", {
                "start_ts": None,
                "last_good_ts": None,
                "bad_streak_start": None,
                "timeline": [],
                "good_frames": 0,
                "bad_frames": 0,
            })
            st.toast("Analytics reset.", icon="🔄")
            st.rerun()

def main():
    st.markdown("<style>.block-container{max-width:720px}</style>", unsafe_allow_html=True)
    if not st.session_state["consented"]:
        consent_gate()
        return
    st.caption("UI loaded. If you don’t see a camera prompt, check the warning above.")
    sidebar_info()
    monitor_section()

if __name__ == "__main__":
    main()
