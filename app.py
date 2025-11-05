# app.py — Full app with webcam temporarily disabled
import sys, platform, streamlit as st
st.set_page_config(page_title="Posture Alert", layout="wide")
st.info(f"✅ App booted • Python {sys.version.split()[0]} • {platform.system()}")
st.write("If you see this header instantly, modules will load next…")

# Lazy-load heavy deps so UI draws first
with st.spinner("Loading posture modules..."):
    import math, time, collections
    from dataclasses import dataclass, field
    from typing import Deque, Dict, Tuple

    import numpy as np
    import cv2, av, mediapipe as mp
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# ---------- constants ----------
NECK_TILT_MAX_DEFAULT = 25.0
BACK_ANGLE_MIN_DEFAULT = 150.0
HEAD_PITCH_MAX_DEFAULT = 20.0
SMOOTH_WINDOW = 30
BAD_POSTURE_PERSIST_SEC = 60.0
MOVE_BODY_INTERVAL_SEC = 30 * 60
MOBILE_MAX_WIDTH = 720

# ---------- session ----------
def ss_get(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]
def ss_set(key, val): st.session_state[key] = val

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

# ---------- geometry / features ----------
VERTICAL = np.array([0.0, -1.0])
def _unit(v): n=np.linalg.norm(v); return v if n==0 else v/n
def angle_deg(v1, v2): 
    v1u, v2u = _unit(v1), _unit(v2)
    return float(np.degrees(np.arccos(float(np.clip(np.dot(v1u, v2u), -1.0, 1.0)))))
def vec(pA, pB, W, H): return np.array([(pB.x-pA.x)*W, (pB.y-pA.y)*H], float)

def features_from_landmarks(lm, W, H) -> Tuple[float,float,float]:
    ls, rs, lh, rh, nose = lm[11], lm[12], lm[23], lm[24], lm[0]
    make_point = lambda x,y: type('P', (), dict(x=x, y=y))()
    ear_like   = make_point((ls.x+nose.x)/2, (ls.y+nose.y)/2)
    mid        = make_point((ls.x+rs.x)/2, (ls.y+rs.y)/2)
    neck_tilt  = angle_deg(vec(ls, ear_like, W, H), VERTICAL)
    back_angle = angle_deg(vec(lh, ls, W, H), VERTICAL)
    head_pitch = angle_deg(vec(mid, nose, W, H), VERTICAL)
    return neck_tilt, back_angle, head_pitch

# ---------- processor ----------
@dataclass
class PostureState:
    hist: Deque[int] = field(default_factory=lambda: collections.deque(maxlen=SMOOTH_WINDOW))
    last_frame_ts: float = 0.0

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp.solutions.pose.Pose(0.5, 0.5)
        self.draw = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles
        self.state = PostureState()
    def _get_thresholds(self) -> Dict[str, float]:
        th = st.session_state["thresholds"].copy()
        s  = float(st.session_state.get("sensitivity", 1.0))
        th["NECK_TILT_MAX"] *= (1.0/s); th["HEAD_PITCH_MAX"] *= (1.0/s); th["BACK_ANGLE_MIN"] *= (1.0/s)
        return th
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24"); H, W = img.shape[:2]
        if st.session_state.get("mirror_video", True): img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); res = self.pose.process(rgb)

        neck = back = head = None; is_slouch = False
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            neck, back, head = features_from_landmarks(lm, W, H)
            if st.session_state.get("show_landmarks", True):
                self.draw.draw_landmarks(img, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.styles.get_default_pose_landmarks_style())
            th = self._get_thresholds()
            bad_neck = (neck is not None and neck > th["NECK_TILT_MAX"])
            bad_back = (back is not None and back < th["BACK_ANGLE_MIN"])
            bad_head = (head is not None and head > th["HEAD_PITCH_MAX"])
            is_slouch = bad_neck or bad_back or bad_head

            msg = "Good posture 👍"
            if is_slouch:
                offenders=[]
                if bad_back: offenders.append(("back", abs(back-th["BACK_ANGLE_MIN"])))
                if bad_head: offenders.append(("head", abs(head-th["HEAD_PITCH_MAX"])))
                if bad_neck: offenders.append(("neck", abs(neck-th["NECK_TILT_MAX"])))
                offenders.sort(key=lambda x:x[1], reverse=True)
                msg = {"back":"Adjust back (sit upright).","head":"Bring head back.","neck":"Stack ears over shoulders."}.get(
                    offenders[0][0] if offenders else "", "Adjust posture (sit tall)."
                )
            y=28
            def put(t,c=(0,255,0)): 
                nonlocal y; cv2.putText(img,t,(10,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,c,2,cv2.LINE_AA); y+=28
            if neck is not None:
                put(f"Neck tilt : {neck:5.1f}°"); put(f"Back angle: {back:5.1f}°"); put(f"Head pitch: {head:5.1f}°")
            if is_slouch:
                cv2.rectangle(img,(0,0),(W,60),(0,0,255),-1)
                cv2.putText(img, msg+"  (1-min timer running)", (10,42), cv2.FONT_HERSHEY_SIMPLEX, 0.85,(255,255,255),2,cv2.LINE_AA)
            else:
                cv2.rectangle(img,(0,0),(W,40),(0,128,0),-1)
                cv2.putText(img, msg, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.85,(255,255,255),2,cv2.LINE_AA)

        now = time.time(); stats = st.session_state["stats"]
        if stats["start_ts"] is None:
            stats["start_ts"] = now; stats["last_move_reminder"] = now
        self.state.hist.append(1 if is_slouch else 0)
        slouch_ratio = sum(self.state.hist)/max(1,len(self.state.hist))
        is_good = (slouch_ratio < 0.5)
        if is_good: stats["good_frames"] += 1; stats["last_good_ts"] = now
        else: stats["bad_frames"] += 1
        if not is_good:
            if stats["bad_streak_start"] is None: stats["bad_streak_start"] = now
            elif (now - stats["bad_streak_start"]) >= BAD_POSTURE_PERSIST_SEC:
                st.toast("⏰ Bad posture for 1 minute — please reset posture.", icon="⚠️")
                stats["bad_streak_start"] = now
        else: stats["bad_streak_start"] = None
        if stats["last_move_reminder"] and (now - stats["last_move_reminder"]) >= MOVE_BODY_INTERVAL_SEC:
            st.toast("🕒 Time to move your body for a minute!", icon="⏳"); stats["last_move_reminder"] = now
        if (now - self.state.last_frame_ts) > 2.0:
            t_rel = now - stats["start_ts"]; stats["timeline"].append((t_rel, 1 if is_good else 0)); self.state.last_frame_ts = now
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------- calibration ----------
def run_calibration(seconds: int = 5):
    st.info("Sit in your best posture. We'll average angles and set personalized thresholds.")
    mp_pose = mp.solutions.pose.Pose(0.5, 0.5)
    st.write("Click **Start camera** below, then hold still in good posture.")
    calib_ctx = webrtc_streamer(key="calibration", mode=WebRtcMode.SENDRECV,
                                media_stream_constraints={"video": True, "audio": False},
                                video_processor_factory=None)
    readings=[]; start=None; progress=st.progress(0.0, text="Waiting for camera...")
    while calib_ctx.state.playing:
        frame = calib_ctx.video_receiver.get_frame(timeout=1)
        if frame is None: continue
        img = frame.to_ndarray(format="bgr24"); H, W = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); res = mp_pose.process(rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            neck, back, head = features_from_landmarks(lm, W, H)
            readings.append((neck, back, head))
            if start is None: start = time.time()
            elapsed = time.time()-start
            progress.progress(min(1.0, elapsed/seconds), text=f"Calibrating... {min(seconds,int(elapsed))}/{seconds}s")
            if elapsed >= seconds: break
    if readings:
        arr=np.array(readings); neck_mean, back_mean, head_mean = arr.mean(axis=0).tolist()
        ss_set("baseline_angles", {"neck": neck_mean, "back": back_mean, "head": head_mean})
        personalized = {
            "NECK_TILT_MAX": max(10.0, neck_mean + 10.0),
            "BACK_ANGLE_MIN": min(179.0, back_mean - 10.0),
            "HEAD_PITCH_MAX": max(8.0,  head_mean + 8.0),
        }
        ss_set("thresholds", personalized); ss_set("calibrated", True)
        st.success("Calibration complete! Personalized thresholds set.")
    else:
        st.warning("No pose detected during calibration. You can skip calibration or try again.")

# ---------- UI ----------
def consent_gate():
    st.title("🧍 Typing Posture Alert")
    st.markdown("**This app uses your webcam locally in your browser.** No video is uploaded.")
    agree = st.checkbox("I consent to webcam access and local posture analysis.")
    if st.button("Continue", type="primary", disabled=not agree):
        ss_set("consented", True); st.rerun()

def monitor_section():
    st.header("Live Monitor")
    colL, colR = st.columns([2,1])
    with colL:
        st.toggle("Mirror video", value=st.session_state["mirror_video"], key="mirror_video")
        st.toggle("Show landmarks", value=st.session_state["show_landmarks"], key="show_landmarks")
        st.slider("Sensitivity (higher = stricter)", 0.7, 1.5, key="sensitivity", value=1.0, step=0.05)
        # Webcam disabled for smoke test
        with st.expander("Webcam (temporarily disabled for smoke test)", expanded=True):
            st.warning("Webcam init is disabled for this test. We'll turn it back on after the page loads.")
            st.caption("If you see this, the app is healthy. The hang (if any) is inside WebRTC.")
            # To enable later, replace this expander with the webrtc_streamer(...) block.
    with colR:
        st.subheader("Analytics")
        stats = st.session_state["stats"]; start_ts = stats["start_ts"]
        good, bad = stats["good_frames"], stats["bad_frames"]; total = max(1, good+bad)
        st.metric("Posture % Good", f"{100.0*good/total:.1f}%")
        st.caption("Percentage of frames assessed as 'good' during this session.")
        tl = stats["timeline"]
        if len(tl) >= 2:
            import pandas as pd
            df = pd.DataFrame(tl, columns=["t_sec","good"]); df["minutes"] = df["t_sec"]/60.0
            df.set_index("minutes", inplace=True); st.line_chart(df["good"], height=160, use_container_width=True)
            st.caption("Timeline: 1 = good posture, 0 = slouching (sampled ~every 2s).")
        else:
            st.info("Timeline will appear after ~10 seconds of monitoring.")
        if start_ts:
            st.metric("Session length", f"{(time.time()-start_ts)/60.0:.1f} min")
        st.divider()
        if st.button("Reset analytics"):
            ss_set("stats", {"start_ts": None,"last_good_ts": None,"last_move_reminder": None,
                             "bad_streak_start": None,"timeline": [],"good_frames": 0,"bad_frames": 0})
            st.toast("Analytics reset.", icon="🔄"); st.rerun()
    st.info("Coaching: Adjust **back / head / neck** when banner turns red. 1-min slouch → warning. 30-min → move reminder.")

def sidebar_info():
    with st.sidebar:
        st.header("Calibration")
        if not st.session_state["calibrated"]:
            if st.button("Start 5s calibration"): run_calibration(5)
        else:
            base = st.session_state["baseline_angles"]; th = st.session_state["thresholds"]
            st.success("Calibrated ✅")
            st.write("**Baseline angles (your best):**"); st.json({k: round(v,1) if v else None for k,v in base.items()})
            st.write("**Personalized thresholds:**");   st.json({k: round(v,1) for k,v in th.items()})
            if st.button("Re-calibrate"): ss_set("calibrated", False)
        st.divider(); st.subheader("Privacy")
        st.caption("• Video stays in your browser session. • No dataset is collected. • Close the tab to end the session.")

def main():
    st.markdown(f"<style>.block-container {{ max-width: {MOBILE_MAX_WIDTH}px; }}</style>", unsafe_allow_html=True)
    if not st.session_state["consented"]:
        consent_gate(); return
    st.title("🧍 Typing Posture Alert")
    st.caption("Webcam + MediaPipe detects slouching posture and gently alerts. Calibrate once, then monitor.")
    sidebar_info(); monitor_section()

if __name__ == "__main__":
    main()
