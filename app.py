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

# ── STRIP STREAMLIT CHROME ──
st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; height: 0; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    [data-testid="stAppViewContainer"] { padding: 0 !important; }
    [data-testid="stVerticalBlock"] { gap: 0 !important; padding: 0 !important; }
    section[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── LOAD HTML ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(BASE_DIR, "insyte_landing.html")

try:
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Render the full HTML document inside an iframe.
    # This is the correct way to render a complete HTML page in Streamlit —
    # st.markdown() only injects fragments into Streamlit's own DOM and
    # cannot apply <style> blocks or <link> tags, which causes raw CSS to
    # appear as text (the bug you were seeing).
    components.html(html_content, height=6000, scrolling=True)

except FileNotFoundError:
    st.error(
        f"File not found: {HTML_PATH}\n\n"
        "Make sure insyte_landing.html is in the same folder as this script."
    )