import streamlit as st
import os
import re

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

    # Pull <head> (styles + font links) and <body> content separately
    # then inject both inline — this avoids the iframe sandbox that
    # blocks Google Fonts and custom CSS in components.html()
    head_match = re.search(r'<head[^>]*>(.*?)</head>', html_content, re.DOTALL)
    body_match = re.search(r'<body[^>]*>(.*?)</body>', html_content, re.DOTALL)

    head_content = head_match.group(1) if head_match else ""
    body_content = body_match.group(1) if body_match else html_content

    st.markdown(f"<head>{head_content}</head>{body_content}", unsafe_allow_html=True)

except FileNotFoundError:
    st.error(
        f"File not found: {HTML_PATH}\n\n"
        "Make sure insyte_landing.html is in the same folder as this script."
    )