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
    iframe { display: block; border: none; }
</style>
""", unsafe_allow_html=True)

# ── LOAD HTML ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_PATH = os.path.join(BASE_DIR, "insyte_landing.html")

try:
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        html_content = f.read()

    # ── PATCH 1: Force hero text visible after animation completes ──
    # Streamlit's iframe can cause CSS fadeUp animations to misfire and leave
    # elements stuck at opacity:0. This JS patch guarantees visibility after
    # the longest animation delay has passed.
    animation_fix = """
    <style>
      .hero-eyebrow, .hero-headline, .hero-sub, .hero-meta, .scroll-hint {
        animation-fill-mode: forwards !important;
      }
    </style>
    <script>
      window.addEventListener('load', function() {
        setTimeout(function() {
          var heroEls = document.querySelectorAll(
            '.hero-eyebrow, .hero-headline, .hero-sub, .hero-meta, .scroll-hint'
          );
          heroEls.forEach(function(el) {
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
          });
        }, 1500);
      });
    </script>
    """

    # ── PATCH 2: Re-initialise IntersectionObserver after load ──
    # Elements already in the initial iframe viewport won't trigger without
    # a scroll event. This re-attaches the observer after the DOM settles.
    observer_fix = """
    <script>
      window.addEventListener('load', function() {
        setTimeout(function() {
          var reveals = document.querySelectorAll('.reveal');
          var observer = new IntersectionObserver(function(entries) {
            entries.forEach(function(e) {
              if (e.isIntersecting) {
                e.target.classList.add('visible');
                observer.unobserve(e.target);
              }
            });
          }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });
          reveals.forEach(function(el) { observer.observe(el); });
        }, 200);
      });
    </script>
    """

    html_content = html_content.replace(
        '</body>',
        animation_fix + observer_fix + '</body>'
    )

    # Height must be set explicitly — adjust upward if page content grows.
    components.html(html_content, height=7000, scrolling=True)

except FileNotFoundError:
    st.error(
        f"File not found: {HTML_PATH}\n\n"
        "Make sure insyte_landing.html is in the same folder as this script.\n\n"
        "TIP: For production, host on Netlify, Vercel, or GitHub Pages. "
        "Plain HTML/CSS/JS renders perfectly there with no iframe sandboxing."
    )