# app.py — FULL APP (Webcam temporarily disabled inside monitor section)
import sys, platform
import streamlit as st

# ---------- Fast header so the page renders immediately ----------
st.set_page_config(page_title="Posture Alert", layout="wide")
st.info(f"✅ App booted • Python {sys.version.split()[0]} • {platform.system()}")
st.write("If you see this header instantly, the app is loading correctly. Modules load next…")

# ---------- Lazy-load heavy dependencies so UI draws first ----------
with st.spinner("Loading posture modules..."):
    import math, time, collections
    from dataclasses import dataclass, field
    from typing import Deque, Dict, Tuple

    import numpy as np
    import cv2, av, mediapipe as mp
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# ---------- Global constants ----------
NECK_TILT_MAX_DEFAULT = 25.0      # deg, larger => slouch
BACK_ANGLE_MIN_DEFAULT = 150.0    # deg, smaller => slouch
HEAD_PITCH_MAX_DEFAULT = 20.0     # deg, larger => slouch
SMOOTH_WINDOW = 30
BAD_POSTURE_PERSIST_SEC = 60.0
MOVE_BODY_INTERVAL_SEC = 30 * 60
MOBILE_MAX_WIDTH = 720

# ---------- Session state helpers ----------
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

# ---------- Geometry / features ----------
VERTICAL = np.array([0.0, -1.0])

def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    v1u, v2u = _unit(v1), _unit(v2)
    dot = float(np.clip(np.dot(v1u, v2u), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))

def vec(pA, pB, W, H) -> np.ndarray:
    return np.array([(pB.x - pA.x) * W, (pB.y - pA.y) * H], dtype=float)

def features_from_landmarks(lm, W: int, H: int) -> Tuple[float, float, float]:
    """
    Returns (neck_tilt, back_angle, head_pitch) in degrees.
    Uses MediaPipe Pose indices: 11,12=shoulders; 23,24=hips; 0=nose
    """
    ls, rs, lh, rh, nose = lm[11], lm[12], lm[23], lm[24], lm[0]
    make_point = lambda x, y: type('P', (), dict(x=x, y=y))()

    ear_like   = make_point((ls.x + nose.x) / 2, (ls.y + nose.y) / 2)
    neck_tilt  = angle_deg(vec(ls, ear_like, W, H), VERTICAL)
    back_angle = angle_deg(vec(lh, ls, W, H), VERTICAL)
    mid        = make_point((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)
    head_pitch = angle_deg(vec(mid, nose, W, H), VERTICAL)
    return neck_tilt, back_angle, head_pitch

# ---------- Video processor ----------
@dataclass
class PostureState:
    hist: Deque[int] = field(default_factory=lambda: collections.deque(maxlen=SMOOTH_WINDOW))
    last_frame_ts: float = 0.0

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.draw   = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles
        self.state  = PostureState()

    def _get_thresholds(self) -> Dict[str, float]:
        th = st.session_state["thresholds"].copy()
        s  = float(st.session_state.get("sensitivity", 1.0))
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

            # Coaching message
            msg = "Good posture 👍"
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

            # Overlay numbers + message
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

        # ---- Stats / analytics ----
        now   = time.time()
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

        if stats["last_move_reminder"] is not None and \
           (now - stats["last_move_reminder"]) >= MOVE_BODY_INTERVAL_SEC:
            st.toast("🕒 Time to move your body for a minute!", icon="⏳")
            stats["last_move_reminder"] = now

        if (now - self.state.last_frame_ts) > 2.0:
            t_rel = now - stats["start_ts"]
            stats["timeline"].append((t_rel, 1 if is_good else 0))
            self.state.last_frame_ts = now

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------- Calibration ----------
def run_calibration(seconds: int = 5):
    st.info("Sit in your best posture. Calibration will average your angles and set personalized thresholds.")
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
    progress = st.progress(0.0, text="Waiting for camera...")
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
            progress.progress(min(1.0, elapsed/seconds), text=f"Calibrating... {min(seconds, int(elapsed))}/{seconds}s")
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
        st.warning("No pose detected during calibration. You can skip calibration or try again.")

# ---------- UI blocks ----------
def consent_gate():
    st.title("🧍 Typing Posture Alert")
    st.markdown(
        """
        **This app uses your webcam in your browser** to analyze your posture locally (no video is uploaded).
        Please confirm:
        - I allow the app to access my webcam.
        - I understand posture analysis runs in this session only and is not stored.
        """
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

        # --- TEMP: Webcam disabled for smoke test (uncomment block below later) ---
        with st.expander("Webcam (temporarily disabled for smoke test)", expanded=True):
            st.warning("Webcam init is disabled for this test. We'll turn it back on after the page loads.")
            st.caption("If you see this, the app is healthy. The hang was inside WebRTC.")
            # with st.spinner("Starting webcam..."):
            #     webrtc_streamer(
            #         key="posture-monitor",
            #         mode=WebRtcMode.SENDRECV,
            #         media_stream_constraints={"video": True, "audio": False},
            #         video_processor_factory=VideoProcessor,
            #     )

    with colR:
        st.subheader("Analytics")
        stats = st.session_state["stats"]
        start_ts = stats["start_ts"]
        good, bad = stats["good_frames"], stats["bad_frames"]
        total = max(1, good + bad)
        st.metric("Posture % Good", f"{100.0 * good / total:.1f}%")
        st.caption("Percentage of frames assessed as 'good' during this session.")

        tl = stats["timeline"]
        if len(tl) >= 2:
            import pandas as pd
            df = pd.DataFrame(tl, columns=["t_sec", "good"])
            df["minutes"] = df["t_sec"] / 60.0
            df.set_index("minutes", inplace=True)
            st.line_chart(df["good"], height=160, use_container_width=True)
            st.caption("Timeline: 1 = good posture, 0 = slouching (sampled ~every 2s).")
        else:
            st.info("Timeline will appear after ~10 seconds of monitoring.")

        if start_ts:
            elapsed_min = (time.time() - start_ts) / 60.0
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
            "If bad posture persists for 1 minute, you’ll get a strong reminder. "
            "Every 30 minutes, you’ll be nudged to move your body.")

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

# ---------- Main ----------
def main():
    st.markdown(
        f"""
        <style>.block-container {{ max-width: {MOBILE_MAX_WIDTH}px; }}</style>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state["consented"]:
        consent_gate()
        return

    st.title("🧍 Typing Posture Alert")
    st.caption("Webcam + MediaPipe detects slouching posture and gently alerts. Calibrate once, then monitor your study session.")
    sidebar_info()
    monitor_section()

if __name__ == "__main__":
    main()
