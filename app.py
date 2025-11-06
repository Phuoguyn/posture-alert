# Posture Alert – Phase 2 (imports + versions, webcam still disabled)
import sys, platform, importlib, textwrap
import streamlit as st

# --- UI header ---
st.set_page_config(page_title="Posture Alert – Phase 2", layout="wide")
st.success(f"✅ Base app running • Python {sys.version.split()[0]} • {platform.system()}")

# --- Try importing heavy deps safely (show any error instead of crashing) ---
pkgs = {
    "streamlit": "streamlit",
    "streamlit-webrtc": "streamlit_webrtc",
    "mediapipe": "mediapipe",
    "opencv-python-headless": "cv2",
    "numpy": "numpy",
    "pandas": "pandas",
    "av": "av",
    "pyarrow": "pyarrow",
    "protobuf": "google.protobuf"
}

rows = []
errors = []

for human, mod in pkgs.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "unknown")
        # special cases
        if mod == "google.protobuf":
            from google.protobuf import __version__ as pbv
            ver = pbv
        rows.append((human, ver, "✅ imported"))
    except Exception as e:
        rows.append((human, "-", "❌ failed"))
        errors.append(f"{human} ({mod}): {type(e).__name__}: {e}")

st.subheader("Dependency check")
st.dataframe(
    {"package": [r[0] for r in rows], "version": [r[1] for r in rows], "status": [r[2] for r in rows]},
    use_container_width=True
)

if errors:
    st.error("Some imports failed. See details below.")
    st.code("\n".join(errors))
else:
    st.success("All imports OK. Next step: enable the UI with webcam disabled, then turn webcam on.")

st.info(textwrap.dedent("""\
    Roadmap:
    1) Phase 2 (this page): verify imports.
    2) Phase 3: render full UI but keep webcam off (no WebRTC).
    3) Phase 4: turn WebRTC on with VideoProcessor.
"""))
