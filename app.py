from streamlit_webrtc import webrtc_streamer, WebRtcMode
with st.expander("Webcam", expanded=True):
    with st.spinner("Starting webcam..."):
        webrtc_streamer(
            key="posture-monitor",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
        )
