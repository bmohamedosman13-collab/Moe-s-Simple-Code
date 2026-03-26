import streamlit as st
import streamlit.components.v1 as components
import os

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="Insyte",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── HIDE STREAMLIT CHROME ──
st.markdown("""
<style>
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    [data-testid="stAppViewContainer"] { padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD AND RENDER THE LANDING PAGE ──
# The HTML file should be in the same directory as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(BASE_DIR, "insyte_landing.html")

try:
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Render full-page HTML — height set large so it doesn't clip
    components.html(html_content, height=6000, scrolling=False)

except FileNotFoundError:
    st.error(
        f"Landing page file not found at: {HTML_PATH}\n\n"
        "Make sure insyte_landing.html is in the same folder as this script."
    )