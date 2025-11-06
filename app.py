# app.py — Streamlit posture app with safe smoke test + lazy imports
# Works on Streamlit Cloud when paired with:
#   runtime.txt:  python-3.11.14
#   requirements.txt pins:
#     streamlit==1.32.0
#     streamlit-webrtc==0.63.11
#     mediapipe==0.10.14
#     opencv-python-headless==4.9.0.80
#     numpy==1.26.4
#     pandas>=2.2,<2.4
#     av==14.0.1
#     protobuf>=4.25.3,<5

import sys, platform, streamlit as st

st.set_page_config(page_title="Posture Alert", page_icon="🧍", layout="wide")
st.info(f"✅ App booted • Python {sys.version.split()[0]} • {platform.system()}")
st.write("If you see this header instantly, modules will load next…")

# ---- Quick toggle to avoid import-build failures blocking the page ----
enable_modules = st.toggle("Load posture modules (cv2 / mediapipe / av / webrtc)", value=False)

if not enable_modules:
    st.warning(
        "Modules are not loaded yet. Flip the toggle when your Cloud logs look clean.\n\n"
        "Tip: Keep the webcam disabled (below) for the first successful render."
    )
    st.stop()

# ---- Heavy deps: import only when user enables them ----
try:
    import time, collections, math
    from dataclasses import dataclass, field
    from typing import Deque, Dict, Tuple

    import numpy as np
    import cv2, av, mediapipe as mp
    import pandas as pd
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

    st.success("✅ Modules imported. Proceeding to load the app UI…")
except Exception as e:
    st.error(f"Import error while loading modules: {e}")
    st.stop()

# =========================
# ====== APP LOGIC ========
# =========================

# ---- GLOBAL CONSTANTS ----
NECK_TILT_MAX_DEFAULT = 25.0      # deg, larger => slouch
BACK_ANGLE_MIN_DEFAULT = 150.0    # deg, smaller => slouch
HEAD_PITCH_MAX_DEFAULT = 20.0     # deg, larger => slouch

SMOOTH_WINDOW = 30                # frames smoothing
BAD_POSTURE_PERSIST_SEC = 60.0    # 1 min continuous bad posture => strong alert
MOVE_BODY_INTERVAL_SEC = 30 * 60  # every 30 min, nudge to move
MOBILE_MAX_WIDTH = 720

# ---- SESSION STATE HELPERS ----
def ss_get(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

def ss_set(key, val):
    st.session_state[key] = val

# Initialize session state
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

# ---- GEOMETRY / FEATURES ----
def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    v1u, v2u = _unit(v1), _unit(v2)
    dot = float(np.clip(np.dot(v1u, v2u), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))

VERTICAL = np.array([0.0, -1.0])

def vec(pA, pB, W, H) -> np.ndarray:
    return np.array([(pB.x - pA.x) * W, (pB.y - pA.y) * H], dtype=float)

def features_from_landmarks(lm, W: int, H: int) -> Tuple[float, float, float]:
    """Returns (neck_tilt, back_angle, head_pitch) in degrees."""
    ls, rs, lh, rh, nose = lm[11], lm[12], lm[23], lm[24], lm[0]
    mk = lambda x, y: type("P", (), dict(x=x, y=y))()

    ear_like = mk((ls.x + nose.x) / 2, (ls.y + nose.y) / 2)
    neck_tilt  = angle_deg(vec(ls, ear_like, W, H), VERTICAL)
    back_angle = angle_deg(vec(lh, ls, W, H), VERTICAL)
    mid = mk((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
    head_pitch = angle_deg(vec(mid, nose, W, H), VERTICAL)
    return neck_tilt, back_angle, head_pitch

# ---- VIDEO PROCESSOR ----
@dataclass
class PostureState:
    hist: Deque[int] = field(default_factory=lambda: collections.deque(maxlen=SMOOTH_WINDOW))
    last_frame_ts: float = 0.0

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.draw = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles
        self.state = PostureState()

    def _get_thresholds(self) -> Dict[str, float]:
        th = st.session_state["thresholds"].copy()
        s = float(st.session_state.get("sensitivity", 1.0))
        th["NECK_TILT_MAX"] = th["NECK_TILT_MAX"] * (1.0 / s)
        th["HEAD_PITCH_MAX"] = th["HEAD_PITCH_MAX"] * (1.0 / s)
        th["BACK_ANGLE_MIN"] = th["BACK_ANGLE_MIN"] * (1.0 / s)
        return th

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        H, W = img.shape[:2]

        if st.session_state.get("mirror_video", True):
            img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        neck = back = head = None
        is_slouch = False
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            neck, back, head = features_from_landmarks(lm, W, H)

            if st.session_state.get("show_landmarks", True):
                self.draw.draw_landmarks(
                    img, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.styles.get_default_pose_landmarks_style()
                )

            th = self._get_thresholds()
            bad_neck = (neck is not None and neck > th["NECK_TILT_MAX"])
            bad_back = (back is not None and back < th["BACK_ANGLE_MIN"])
            bad_head = (head is not None and head > th["HEAD_PITCH_MAX"])
            is_slouch = bad_neck or bad_back or bad_head

            # Coaching
            if is_slouch:
                offenders = []
                if bad_back: offenders.append(("back", abs(back - th["BACK_ANGLE_MIN"])))
                if bad_head: offenders.append(("head", abs(head - th["HEAD_PITCH_MAX"])))
                if bad_neck: offenders.append(("neck", abs(neck - th["NECK_TILT_MAX"])))
                offenders.sort(key=lambda x: x[1], reverse=True)
                top = offenders[0][0] if offenders else None
                msg = {
                    "back": "Adjust your back (sit upright).",
                    "head": "Bring your head back (reduce forward head).",
                    "neck": "Relax neck; stack ears over shoulders.",
                }.get(top, "Adjust posture (sit tall).")
            else:
                msg = "Good posture 👍"

            # HUD
            y = 28
            def put(txt, color=(0,255,0)):
                nonlocal y
                cv2.putText(img, txt, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                y += 28

            if neck is not None:
                put(f"Neck tilt : {neck:5.1f}°")
                put(f"Back angle: {back:5.1f}°")
                put(f"Head pitch: {head:5.1f}°")

            if is_slouch:
                cv2.rectangle(img, (0,0), (W,60), (0,0,255), -1)
                cv2.putText(img, msg + "  (1-min timer running)", (10,42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2, cv2.LINE_AA)
            else:
                cv2.rectangle(img, (0,0), (W,40), (0,128,0), -1)
                cv2.putText(img, msg, (10,28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2, cv2.LINE_AA)

        # Stats
        now = time.time()
        stats = st.session_state["stats"]
        if stats["start_ts"] is None:
            stats["start_ts"] = now
            stats["last_move_reminder"] = now

        self.state.hist.append(1 if is_slouch else 0)
        slouch_ratio = sum(self.state.hist) / max(1, len(self.state.hist))
        is_good = (slouch_ratio < 0.5)

        if is_good:
            stats["good_frames"] += 1
            stats["last_good_ts"] = now
        else:
            stats["bad_frames"] += 1

        if not is_good:
            if stats["bad_streak_start"] is None:
                stats["bad_streak_start"] = now
            elif (now - stats["bad_streak_start"]) >= BAD_POSTURE_PERSIST_SEC:
                st.toast("⏰ Bad posture for 1 minute — please reset posture.", icon="⚠️")
                stats["bad_streak_start"] = now
        else:
            stats["bad_streak_start"] = None

        if stats["last_move_reminder"] is not None and (now - stats["last_move_reminder"]) >= MOVE_BODY_INTERVAL_SEC:
            st.toast("🕒 Time to move your body for a minute!", icon="⏳")
            stats["last_move_reminder"] = now

        if (now - self.state.last_frame_ts) > 2.0:
            t_rel = now - stats["start_ts"]
            stats["timeline"].append((t_rel, 1 if is_good else 0))
            self.state.last_frame_ts = now

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---- CALIBRATION (optional; can run later) ----
def run_calibration(seconds: int = 5):
    st.info("Sit in your best posture. We’ll average angles to set personalized thresholds.")
    mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    st.write("Click **Start camera** below, then hold still in good posture.")
    calib_ctx = webrtc_streamer(
        key="calibration",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=None,  # raw frames
    )

    readings = []
    start = None
    progress = st.progress(0.0, text="Waiting for camera…")
    while calib_ctx.state.playing:
        frame = calib_ctx.video_receiver.get_frame(timeout=1)
        if frame is None:
            continue
        img = frame.to_ndarray(format="bgr24")
        H, W = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = mp_pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            neck, back, head = features_from_landmarks(lm, W, H)
            readings.append((neck, back, head))
            if start is None:
                start = time.time()
            elapsed = time.time() - start
            progress.progress(min(1.0, elapsed/seconds), text=f"Calibrating… {min(seconds, int(elapsed))}/{seconds}s")
            if elapsed >= seconds:
                break

    if readings:
        arr = np.array(readings)
        neck_mean, back_mean, head_mean = arr.mean(axis=0).tolist()
        ss_set("baseline_angles", {"neck": neck_mean, "back": back_mean, "head": head_mean})
        personalized = {
            "NECK_TILT_MAX": max(10.0, neck_mean + 10.0),
            "BACK_ANGLE_MIN": min(179.0, back_mean - 10.0),
            "HEAD_PITCH_MAX": max(8.0,  head_mean + 8.0),
        }
        ss_set("thresholds", personalized)
        ss_set("calibrated", True)
        st.success("Calibration complete! Personalized thresholds set.")
    else:
        st.warning("No pose detected. You can skip or try again.")

# ---- UI PIECES ----
def consent_gate():
    st.title("🧍 Typing Posture Alert")
    st.markdown(
        "**This app uses your webcam in your browser** to analyze posture locally (no video is uploaded)."
    )
    agree = st.checkbox("I consent to webcam access and local posture analysis.")
    if st.button("Continue", type="primary", disabled=not agree):
        ss_set("consented", True)
        st.rerun()

def monitor_section():
    st.header("Live Monitor")
    colL, colR = st.columns([2, 1])

    with colL:
        st.toggle("Mirror video", value=st.session_state["mirror_video"], key="mirror_video")
        st.toggle("Show landmarks", value=st.session_state["show_landmarks"], key="show_landmarks")
        st.slider("Sensitivity (higher = stricter)", 0.7, 1.5, key="sensitivity", value=1.0, step=0.05)

        # ---- TEMP: disable webcam to avoid WebRTC hang during first deployments ----
        with st.expander("Webcam (temporarily disabled for smoke test)", expanded=True):
            st.warning("Webcam init is disabled for this test. We’ll turn it back on after the page is stable.")
            st.caption("If the page loads and you see this, the app is healthy. Any previous hang was inside WebRTC.")
            # Re-enable this once imports & deps are stable:
            # with st.spinner("Starting webcam…"):
            #     webrtc_streamer(
            #         key="posture-monitor",
            #         mode=WebRtcMode.SENDRECV,
            #         media_stream_constraints={"video": True, "audio": False},
            #         video_processor_factory=VideoProcessor,
            #     )

    with colR:
        st.subheader("Analytics")
        stats = st.session_state["stats"]
        good, bad = stats["good_frames"], stats["bad_frames"]
        total = max(1, good + bad)
        pct = 100.0 * good / total
        st.metric("Posture % Good", f"{pct:.1f}%")
        st.caption("Percent of frames assessed as 'good' this session.")

        tl = stats["timeline"]
        if len(tl) >= 2:
            df = pd.DataFrame(tl, columns=["t_sec", "good"])
            df["minutes"] = df["t_sec"] / 60.0
            df.set_index("minutes", inplace=True)
            st.line_chart(df["good"], height=160, use_container_width=True)
            st.caption("Timeline: 1 = good posture, 0 = slouching (sampled ~every 2s).")
        else:
            st.info("Timeline will appear after ~10 seconds of monitoring (once webcam is enabled).")

        if stats["start_ts"]:
            elapsed_min = (time.time() - stats["start_ts"]) / 60.0
            st.metric("Session length", f"{elapsed_min:.1f} min")

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

    st.info("Coaching: If you see a red banner, adjust your **back / head / neck**. "
            "After 1 min of continuous slouching you’ll get a stronger alert. "
            "Every 30 minutes, you’ll be reminded to move.")

def sidebar_info():
    with st.sidebar:
        st.header("Calibration")
        if not st.session_state["calibrated"]:
            if st.button("Start 5s calibration"):
                run_calibration(5)
        else:
            base = st.session_state["baseline_angles"]
            th = st.session_state["thresholds"]
            st.success("Calibrated ✅")
            st.write("**Baseline angles (your best):**")
            st.json({k: round(v,1) if v else None for k,v in base.items()})
            st.write("**Personalized thresholds:**")
            st.json({k: round(v,1) for k,v in th.items()})
            if st.button("Re-calibrate"):
                ss_set("calibrated", False)

        st.divider()
        st.subheader("Privacy")
        st.caption("• Video stays in your browser session.\n"
                   "• No dataset is collected or stored.\n"
                   "• Close the tab to end the session.")

# ---- MAIN ----
def main():
    st.markdown(
        f"""
        <style>
        .block-container {{ max-width: {MOBILE_MAX_WIDTH}px; }}
        </style>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state["consented"]:
        consent_gate()
        return

    st.title("🧍 Typing Posture Alert")
    st.caption("Webcam + MediaPipe detects slouching posture and gently alerts. "
               "Calibrate once, then monitor your study session.")
    sidebar_info()
    monitor_section()

if __name__ == "__main__":
    main()
