# -----------------------------------------------------------
# Typing Posture Alert - Minimal debug + posture detector
# -----------------------------------------------------------
import sys
import platform
import collections
from dataclasses import dataclass, field
from typing import Deque

import streamlit as st
import numpy as np

st.set_page_config(page_title="Typing Posture Alert", page_icon="🧍", layout="wide")
st.success(f"✅ App booted • Python {sys.version.split()[0]} • {platform.system()}")

# -------- 1. Try to load webcam + pose stack (show real error) --------
try:
    import av
    import cv2
    import mediapipe as mp
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase
    HAVE_CAM = True
    st.info("Webcam stack: ✅ OK")
except Exception as e:
    HAVE_CAM = False
    st.error(f"Webcam stack failed to import: {e!r}")
    st.stop()  # stop the app here so you SEE the real reason

# -------- 2. Simple posture state + math --------
NECK_TILT_MAX = 35.0  # degrees
SMOOTH_WINDOW = 10

@dataclass
class PostureState:
    hist: Deque[int] = field(default_factory=lambda: collections.deque(maxlen=SMOOTH_WINDOW))

def angle_between(v1, v2):
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

# -------- 3. Video processor (robust, no fancy globals) --------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.state = PostureState()
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.draw = mp.solutions.drawing_utils

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")

            # Mirror for natural feel
            img = cv2.flip(img, 1)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)

            is_good = True

            if results.pose_landmarks:
                h, w, _ = img.shape
                lm = results.pose_landmarks.landmark

                # Left shoulder (11), left ear (7)
                shoulder = np.array([lm[11].x * w, lm[11].y * h])
                ear = np.array([lm[7].x * w, lm[7].y * h])

                vertical = np.array([0, -1])
                neck_vec = ear - shoulder
                neck_tilt = angle_between(neck_vec, vertical)

                if neck_tilt > NECK_TILT_MAX:
                    is_good = False

                # Smooth using history
                self.state.hist.append(1 if is_good else 0)
                avg = sum(self.state.hist) / max(1, len(self.state.hist))
                is_good = avg >= 0.5

                # Draw landmarks
                self.draw.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS,
                )

            # Label
            label = "GOOD" if is_good else "SLOUCH"
            color = (0, 200, 0) if is_good else (0, 0, 255)
            cv2.putText(
                img,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2,
                cv2.LINE_AA,
            )

            return av.VideoFrame.from_ndarray(img, format="bgr24")

        except Exception:
            # If anything goes wrong, NEVER kill the stream: just show raw frame
            return frame

# -------- 4. UI: just show camera + overlay --------
st.header("🧍 Live Posture Monitor")

webrtc_streamer(
    key="posture-monitor",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

