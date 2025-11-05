import sys, platform, streamlit as st
st.set_page_config(page_title="Posture Alert", layout="wide")
st.success(f"Smoke test OK • Python {sys.version.split()[0]} • {platform.system()}")
st.stop()
