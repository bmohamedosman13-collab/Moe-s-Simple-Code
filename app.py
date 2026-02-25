import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import re
import pickle
import os
import streamlit.components.v1 as components

# --- 0. CONFIGURATION ---
survey_url = "https://docs.google.com/forms/d/e/1FAIpQLScJZaX8ZrrIHv6XKB-sVPpIJcJlMUl7GUb5mbGCBQYzAj-WPQ/viewform"

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Insyte | Early Linguistic Examiner",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- 2. PREMIUM STYLING ---
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=Instrument+Serif:ital@0;1&display=swap');

    /* ── ANIMATED AURORA BACKGROUND ── */
    .stApp {
        background: #05030f;
        color: #ede9f4;
        font-family: 'DM Sans', sans-serif;
        position: relative;
        overflow-x: hidden;
    }

    /* Aurora layers rendered via pseudo-elements on a fixed overlay */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        background:
            radial-gradient(ellipse 80% 60% at 20% 30%, rgba(72, 40, 180, 0.38) 0%, transparent 70%),
            radial-gradient(ellipse 60% 50% at 80% 70%, rgba(120, 60, 200, 0.28) 0%, transparent 65%),
            radial-gradient(ellipse 50% 40% at 55% 10%, rgba(30, 10, 90, 0.45) 0%, transparent 60%);
        animation: auroraShift 14s ease-in-out infinite alternate;
    }

    .stApp::after {
        content: '';
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        background:
            radial-gradient(ellipse 70% 55% at 65% 80%, rgba(90, 20, 160, 0.25) 0%, transparent 60%),
            radial-gradient(ellipse 40% 35% at 10% 80%, rgba(40, 10, 120, 0.30) 0%, transparent 55%);
        animation: auroraShift2 18s ease-in-out infinite alternate;
    }

    @keyframes auroraShift {
        0%   { transform: scale(1)   translate(0px, 0px); }
        33%  { transform: scale(1.08) translate(-30px, 20px); }
        66%  { transform: scale(0.95) translate(25px, -15px); }
        100% { transform: scale(1.05) translate(-10px, 30px); }
    }

    @keyframes auroraShift2 {
        0%   { transform: scale(1.02) translate(0px, 0px); }
        40%  { transform: scale(0.96) translate(20px, -25px); }
        80%  { transform: scale(1.06) translate(-15px, 15px); }
        100% { transform: scale(1)   translate(10px, -10px); }
    }

    /* Noise grain overlay for depth */
    body::after {
        content: '';
        position: fixed;
        inset: 0;
        z-index: 1;
        pointer-events: none;
        opacity: 0.035;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
        background-size: 200px 200px;
    }

    /* Ensure all Streamlit content sits above aurora */
    .main .block-container, section[data-testid="stSidebar"] { position: relative; z-index: 2; }

    /* ── SIDEBAR ── */
    section[data-testid="stSidebar"] {
        background: rgba(8, 5, 20, 0.80) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(150, 100, 255, 0.12);
    }

    section[data-testid="stSidebar"] .stMarkdown p {
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
        color: #b0a8cc;
    }

    section[data-testid="stSidebar"] h1 {
        font-family: 'DM Serif Display', serif !important;
        font-size: 1.8rem !important;
        color: #c4b5f4 !important;
        letter-spacing: -0.02em;
    }

    section[data-testid="stSidebar"] h3 {
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #7a6f99 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600;
        margin-bottom: 8px !important;
    }

    .sidebar-nav-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 9px 14px;
        border-radius: 8px;
        color: #c0b8d8;
        font-size: 0.9rem;
        transition: background 0.2s;
        margin-bottom: 4px;
    }

    .sidebar-nav-item:hover { background: rgba(180, 150, 255, 0.1); }

    .sidebar-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: rgba(150, 120, 230, 0.5);
        flex-shrink: 0;
    }

    /* ── TYPOGRAPHY ── */
    h1, h2, h3 {
        font-family: 'DM Serif Display', serif !important;
        letter-spacing: -0.02em;
    }

    h1 { font-size: 3.6rem !important; font-weight: 400 !important; line-height: 1.08 !important; color: #f0eaf8 !important; }
    h2 { font-size: 2.2rem !important; font-weight: 400 !important; color: #e2d9f3 !important; }
    h3 { font-size: 1.3rem !important; color: #b89ef0 !important; font-weight: 400 !important; }

    p, li, label, .stMarkdown p {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 1.05rem !important;
        line-height: 1.75 !important;
        color: #c8c0dc;
    }

    /* ── MAIN CONTENT GLASS CARD ── */
    .glass-card {
        background: rgba(18, 12, 38, 0.55);
        border: 1px solid rgba(160, 130, 255, 0.15);
        border-radius: 16px;
        backdrop-filter: blur(24px);
        padding: 40px 44px;
        margin-bottom: 24px;
    }

    /* ── HERO BRAND ── */
    .hero-brand {
        font-family: 'DM Serif Display', serif;
        font-size: 5rem;
        font-weight: 400;
        letter-spacing: -0.04em;
        line-height: 1;
        background: linear-gradient(135deg, #e8e0ff 0%, #a78bfa 40%, #7c3aed 70%, #c4b5f4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        background-size: 300% 300%;
        animation: brandGradient 6s ease infinite;
    }

    @keyframes brandGradient {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .hero-sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #7a6f99;
        margin-top: 4px;
    }

    /* ── DIVIDER ── */
    .styled-divider {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, rgba(160, 130, 255, 0.3), transparent);
        margin: 28px 0;
    }

    /* ── TAG PILLS ── */
    .tag-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 20px; }
    .tag-pill {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.75rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #a78bfa;
        background: rgba(120, 80, 220, 0.12);
        border: 1px solid rgba(120, 80, 220, 0.25);
        padding: 5px 14px;
        border-radius: 100px;
    }

    /* ── STAT BLOCK ── */
    .stat-block {
        display: flex;
        align-items: baseline;
        gap: 14px;
        margin: 12px 0 8px;
    }
    .stat-number {
        font-family: 'DM Serif Display', serif;
        font-size: 4.5rem;
        line-height: 1;
        background: linear-gradient(135deg, #c4b5f4, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        color: #a09ac0;
        max-width: 260px;
        line-height: 1.4;
    }

    /* ── TRUST BADGES ── */
    .trust-row { display: flex; gap: 16px; margin-top: 16px; flex-wrap: wrap; }
    .trust-badge {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(30, 20, 60, 0.6);
        border: 1px solid rgba(120, 90, 200, 0.2);
        border-radius: 10px;
        padding: 10px 16px;
        font-size: 0.82rem;
        color: #b0a8cc;
        font-family: 'DM Sans', sans-serif;
    }
    .trust-icon { font-size: 1.1rem; }

    /* ── SURVEY BOX ── */
    .survey-box {
        background: rgba(80, 50, 160, 0.12);
        border: 1px solid rgba(160, 130, 255, 0.2);
        border-radius: 14px;
        padding: 32px 36px;
        text-align: center;
        margin-top: 8px;
    }
    .survey-box h2 { font-size: 1.7rem !important; margin: 0 0 8px !important; }
    .survey-box p { font-size: 0.95rem !important; color: #9890b8 !important; margin-bottom: 18px !important; }
    .survey-link {
        display: inline-block;
        color: #c4b5f4 !important;
        font-size: 0.85rem !important;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        text-decoration: none;
        padding: 12px 28px;
        border: 1px solid rgba(180, 150, 255, 0.35);
        border-radius: 8px;
        transition: all 0.25s;
        background: rgba(120, 80, 220, 0.1);
    }
    .survey-link:hover { background: rgba(120, 80, 220, 0.25); color: #e0d0ff !important; }

    /* ── BUTTON ── */
    .stButton > button {
        background: linear-gradient(135deg, #6d28d9, #7c3aed) !important;
        color: #f5f0ff !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        padding: 14px 32px !important;
        border-radius: 10px !important;
        border: none !important;
        width: 100%;
        transition: all 0.25s !important;
        box-shadow: 0 4px 24px rgba(109, 40, 217, 0.35) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 32px rgba(109, 40, 217, 0.5) !important;
        background: linear-gradient(135deg, #7c3aed, #8b5cf6) !important;
    }

    /* ── TEXT AREA ── */
    .stTextArea textarea {
        background: rgba(15, 10, 35, 0.7) !important;
        border: 1px solid rgba(120, 90, 200, 0.25) !important;
        border-radius: 10px !important;
        color: #e0d8f0 !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        line-height: 1.65 !important;
        transition: border-color 0.2s !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(160, 130, 255, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(120, 80, 220, 0.12) !important;
    }

    /* ── RADIO ── */
    .stRadio > div { gap: 12px !important; }
    .stRadio label {
        font-size: 0.9rem !important;
        color: #c0b8d8 !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ── METRICS ── */
    [data-testid="stMetric"] {
        background: rgba(20, 12, 45, 0.6);
        border: 1px solid rgba(120, 90, 200, 0.2);
        border-radius: 12px;
        padding: 16px 18px !important;
    }
    [data-testid="stMetricLabel"] { font-size: 0.75rem !important; color: #7a6f99 !important; letter-spacing: 0.08em; text-transform: uppercase; }
    [data-testid="stMetricValue"] { font-family: 'DM Serif Display', serif !important; font-size: 2rem !important; color: #c4b5f4 !important; }

    /* ── PROGRESS ── */
    .stProgress > div > div { background: linear-gradient(to right, #6d28d9, #a78bfa) !important; border-radius: 4px; }
    .stProgress > div { background: rgba(40, 25, 80, 0.5) !important; border-radius: 4px; }

    /* ── EVIDENCE SENTENCES ── */
    .evidence-item {
        background: rgba(20, 12, 45, 0.5);
        border-left: 3px solid #7c3aed;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .evidence-text { font-size: 1rem !important; color: #d8d0ee !important; line-height: 1.6 !important; }
    .evidence-score { font-size: 0.78rem; color: #9070c8; letter-spacing: 0.06em; text-transform: uppercase; margin-top: 4px; }

    /* ── CAPTION / SMALL TEXT ── */
    .stCaption, small, .caption-text {
        font-size: 0.75rem !important;
        color: #5a5278 !important;
        font-family: 'DM Sans', sans-serif !important;
        letter-spacing: 0.04em;
    }

    /* ── WARNING / ERROR ── */
    .stAlert { border-radius: 10px !important; }

    /* ── SECTION LABEL ── */
    .section-eyebrow {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.7rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #6d5fa0;
        margin-bottom: 6px;
    }

    /* ── ASSESSMENT RESULT ── */
    .result-label {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem;
        color: #c4b5f4;
        line-height: 1;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2.5rem !important; padding-bottom: 3rem !important; max-width: 820px !important; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if 'show_demo' not in st.session_state:
    st.session_state.show_demo = False

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("<h1>Insyte</h1>", unsafe_allow_html=True)
    st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h3>Navigation</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class='sidebar-nav-item'><div class='sidebar-dot'></div>Patient assessment tracking</div>
    <div class='sidebar-nav-item'><div class='sidebar-dot'></div>Treatment progress monitoring</div>
    <div class='sidebar-nav-item'><div class='sidebar-dot'></div>Structured clinical check-ins</div>
    <div class='sidebar-nav-item'><div class='sidebar-dot'></div>Documentation support</div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)
    if st.button("← Home"):
        st.session_state.show_demo = False
        st.rerun()
    st.markdown("<p class='caption-text' style='margin-top:16px;'>Pilot v1.2 · Clinical Validation Phase</p>", unsafe_allow_html=True)

# --- 5. CORE ENGINE SETUP (UNCHANGED) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=1)
        return (x * weights).sum(dim=1), weights

class ReHAN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.word_rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.word_att = Attention(hidden_dim)
        self.sent_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.sent_att = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, docs):
        B, S, W = docs.size()
        docs_flat = docs.view(B * S, W)
        emb = self.embedding(docs_flat)
        h_word, _ = self.word_rnn(emb)
        s_word, _ = self.word_att(h_word)
        sent_vecs = s_word.view(B, S, -1)
        h_sent, _ = self.sent_rnn(sent_vecs)
        doc_vec, sent_weights = self.sent_att(h_sent)
        return self.fc(doc_vec), sent_weights

@st.cache_resource
def load_resources():
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(base_path, "rehan_severity_vocab.pkl"), "rb") as f: v_sev = pickle.load(f)
        with open(os.path.join(base_path, "rehan_cause_vocab.pkl"), "rb") as f: v_cau = pickle.load(f)
        r_sev = ReHAN(len(v_sev), 200, 128, 4).to(device)
        r_sev.load_state_dict(torch.load(os.path.join(base_path, "rehan_severity.pt"), map_location=device))
        r_sev.eval()
        r_cau = ReHAN(len(v_cau), 200, 128, 6).to(device)
        r_cau.load_state_dict(torch.load(os.path.join(base_path, "rehan_cause.pt"), map_location=device))
        r_cau.eval()
        rob_sev = RobertaForSequenceClassification.from_pretrained("roberta-base").to(device)
        rob_sev.eval()
        rob_cau = RobertaForSequenceClassification.from_pretrained("roberta-base").to(device)
        rob_cau.eval()
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        return r_sev, r_cau, rob_sev, rob_cau, tokenizer, v_sev, v_cau, None
    except Exception as e:
        return None, None, None, None, None, None, None, str(e)

r_sev, r_cau, rob_sev, rob_cau, tokenizer, v_sev, v_cau, error_msg = load_resources()

# --- 6. MAIN VIEW LOGIC ---
if not st.session_state.show_demo:

    # ── HERO ──
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 8px;'>
        <div class='hero-brand'>Insyte</div>
        <div class='hero-sub'>Early Linguistic Examiner · Clinical Pilot</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)

    # ── ANIMATED COUNT-UP STAT ──
    count_up_html = """
    <div style="text-align:center; padding: 28px 0 20px;">
        <div style="font-size:0.72rem; letter-spacing:0.2em; text-transform:uppercase; color:#5a5278; font-family:'DM Sans',sans-serif; margin-bottom:10px;">The Problem</div>
        <div style="display:flex; align-items:baseline; justify-content:center; gap:16px;">
            <div id="counter" style="font-family:'DM Serif Display',serif; font-size:5.5rem; line-height:1; background:linear-gradient(135deg,#c4b5f4,#7c3aed); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">0</div>
            <div style="font-family:'DM Sans',sans-serif; font-size:1rem; color:#9890b8; max-width:220px; text-align:left; line-height:1.4;">people die by suicide annually in Canada<br><span style='color:#5a5278; font-size:0.8rem;'>Statistics Canada</span></div>
        </div>
    </div>
    <script>
        let startTimestamp = null;
        const duration = 2800;
        const finalValue = 4500;
        const element = document.getElementById('counter');
        const easeOutExpo = t => t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const eased = easeOutExpo(progress);
            element.innerHTML = Math.floor(eased * finalValue).toLocaleString();
            if (progress < 1) window.requestAnimationFrame(step);
        };
        window.requestAnimationFrame(step);
    </script>
    """
    components.html(count_up_html, height=190)

    # ── MISSION + WHO IT'S FOR ──
    st.markdown("""
    <div class='glass-card'>
        <div class='tag-row'>
            <span class='tag-pill'>For Psychologists</span>
            <span class='tag-pill'>For Mental Health Clinicians</span>
            <span class='tag-pill'>Written Intake Analysis</span>
        </div>
        <div class='section-eyebrow'>The Mission</div>
        <p style='font-size:1.1rem !important; color:#d8d0ee !important; margin-bottom:18px;'>
            Clinicians face relentless caseloads and documentation pressure. Early warning signals buried in patient language go undetected — not from lack of skill, but lack of time.
        </p>
        <p style='font-size:1.05rem !important; color:#b0a8cc !important; margin-bottom:0;'>
            Insyte builds AI-assisted tools that surface structured clinical insight from written intake and assessment materials — giving you a sharper lens, faster.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── WHY USE IT ──
    st.markdown("""
    <div class='glass-card'>
        <div class='section-eyebrow'>Why Insyte</div>
        <div class='trust-row'>
            <div class='trust-badge'><span class='trust-icon'>🔍</span> Linguistic signal detection</div>
            <div class='trust-badge'><span class='trust-icon'>🤖</span> RoBERTa + ReHAN hybrid model</div>
            <div class='trust-badge'><span class='trust-icon'>🔒</span> Anonymized inputs only</div>
            <div class='trust-badge'><span class='trust-icon'>📋</span> Structured for clinical review</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CTA BUTTON ──
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        if st.button("Try the demo →"):
            st.session_state.show_demo = True
            st.rerun()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── SURVEY BOX ──
    st.markdown(f"""
    <div class="survey-box">
        <div class='section-eyebrow' style='text-align:center;'>Shape this tool</div>
        <h2>Help shape what this becomes.</h2>
        <p>This is a clinical pilot. Your feedback — from clinicians, for clinicians — defines the roadmap.</p>
        <a href="{survey_url}" class="survey-link" target="_blank">Share Your Feedback →</a>
    </div>
    """, unsafe_allow_html=True)

else:
    # --- TOOL VIEW ---
    st.markdown("""
    <div style='margin-bottom: 4px;'>
        <div class='section-eyebrow'>Insyte · Early Linguistic Examiner</div>
        <h1 style='font-size:2.6rem !important; margin-bottom:4px !important;'>Patient Discourse Analysis</h1>
        <p style='color:#6d5fa0; font-size:0.9rem !important; margin-top:0;'>Clinical Pilot · Semantic Analysis Engine · Handle all inputs with care</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)

    if error_msg:
        st.error(f"Model Load Error: {error_msg}")
        st.stop()

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    user_input = st.text_area(
        "Anonymized Patient Discourse",
        height=230,
        placeholder="Paste anonymized intake text, session notes, or patient-authored content here…"
    )
    analysis_type = st.radio("Analysis Mode", ["Severity", "Causality"], horizontal=True)

    run = st.button("Generate Clinical Insight →")
    st.markdown("</div>", unsafe_allow_html=True)

    if run:
        if user_input.strip():
            raw_sentences = [s.strip() for s in re.split(r'[.!?]', user_input) if s.strip()][:15]
            inputs = tokenizer([user_input], return_tensors="pt", truncation=True, padding=True).to(device)

            with torch.no_grad():
                sev_out = rob_sev(**inputs)
                sev_probs = F.softmax(sev_out.logits, dim=1)[0]
                p_non_severe, p_severe = sev_probs.tolist()

            if analysis_type == "Severity":
                with torch.no_grad():
                    tensor = torch.zeros(15, 20, dtype=torch.long).to(device)
                    for i, sent in enumerate(raw_sentences):
                        words = sent.lower().split()[:20]
                        for j, word in enumerate(words):
                            tensor[i, j] = v_sev.get(word, v_sev.get("<UNK>", 1))
                    _, weights_re = r_sev(tensor.unsqueeze(0))
                    importance = weights_re.squeeze().cpu().tolist()
                    if isinstance(importance, float): importance = [importance]

                rehan_signal = min(sum(w for w in importance if w > 0.15), 1.0)
                raw_hybrid = (p_severe * 0.6) + (rehan_signal * 0.4)

                if p_non_severe > 0.85 and len(user_input.split()) < 12: final_label = "Healthy Range"
                elif p_severe > 0.75: final_label = "Severe"
                else:
                    if raw_hybrid < 0.25: final_label = "Minimum"
                    elif raw_hybrid < 0.45: final_label = "Mild"
                    elif raw_hybrid < 0.70: final_label = "Moderate"
                    else: final_label = "Severe"

                st.markdown(f"""
                <div class='glass-card' style='margin-top:20px;'>
                    <div class='section-eyebrow'>Severity Assessment</div>
                    <div class='result-label'>{final_label}</div>
                </div>
                """, unsafe_allow_html=True)

                cols = st.columns(3)
                cols[0].metric("RoBERTa Confidence", f"{p_severe:.2f}")
                cols[1].metric("ReHAN Attribution", f"{rehan_signal:.2f}")
                cols[2].metric("Composite Index", f"{raw_hybrid:.2f}")

            else:
                with torch.no_grad():
                    cau_out = rob_cau(**inputs)
                    cau_probs = F.softmax(cau_out.logits, dim=1)[0]
                    tensor_cau = torch.zeros(15, 20, dtype=torch.long).to(device)
                    for i, sent in enumerate(raw_sentences):
                        words = sent.lower().split()[:20]
                        for j, word in enumerate(words):
                            tensor_cau[i, j] = v_cau.get(word, v_cau.get("<UNK>", 1))
                    _, weights_cau = r_cau(tensor_cau.unsqueeze(0))
                    importance = weights_cau.squeeze().cpu().tolist()

                causes = ["No Reason", "Bias", "Job / Career", "Medication", "Relationship", "Alienation"]
                rob_max, rob_idx = torch.max(cau_probs, dim=0)
                rehan_cau_sig = min(sum(w for w in (importance if isinstance(importance, list) else [importance]) if w > 0.18), 1.0)
                hybrid_cau = (rob_max.item() * 0.5) + (rehan_cau_sig * 0.5)
                result_cause = causes[rob_idx.item()] if hybrid_cau >= 0.40 else "Inconclusive"

                st.markdown(f"""
                <div class='glass-card' style='margin-top:20px;'>
                    <div class='section-eyebrow'>Primary Thematic Determinant</div>
                    <div class='result-label'>{result_cause}</div>
                    <div style='margin-top:16px;'>
                """, unsafe_allow_html=True)
                st.progress(hybrid_cau)
                st.markdown("</div></div>", unsafe_allow_html=True)

            # ── LINGUISTIC EVIDENCE ──
            st.markdown("<div class='styled-divider'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-eyebrow'>Linguistic Evidence · High-Signal Passages</div>", unsafe_allow_html=True)

            imp_list = importance if isinstance(importance, list) else [importance]
            found = False
            for score, sent in zip(imp_list, raw_sentences):
                if score > 0.15:
                    found = True
                    st.markdown(f"""
                    <div class='evidence-item'>
                        <div class='evidence-text'>{sent}</div>
                        <div class='evidence-score'>Signal Intensity · {score:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            if not found:
                st.markdown("<p style='color:#5a5278; font-size:0.9rem !important;'>No high-signal passages detected above threshold.</p>", unsafe_allow_html=True)

            st.markdown("""
            <p style='font-size:0.78rem !important; color:#3d3558 !important; margin-top:24px; line-height:1.5 !important;'>
            This tool is for clinical decision support only. It does not replace clinical judgment, diagnosis, or risk assessment protocols.
            </p>
            """, unsafe_allow_html=True)

        else:
            st.warning("Please enter patient discourse text to analyze.")