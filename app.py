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

# --- 2. AURORA BACKGROUND — injected into parent window via JS ---
# Streamlit iframes can reach window.parent to inject into the real page DOM
aurora_injector = """
<script>
(function() {
    var p = window.parent.document;
    if (p.getElementById('insyte-aurora')) return; // already injected

    // Inject Google Fonts
    var font = p.createElement('link');
    font.rel = 'stylesheet';
    font.href = 'https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap';
    p.head.appendChild(font);

    // Build aurora container
    var aurora = p.createElement('div');
    aurora.id = 'insyte-aurora';
    aurora.style.cssText = [
        'position:fixed','top:0','left:0','width:100vw','height:100vh',
        'z-index:0','pointer-events:none','overflow:hidden',
        'background:#06030f'
    ].join(';');

    // Four animated orbs
    var orbs = [
        { w:750, h:520, bg:'#2e17a0', top:'-120px', left:'-180px', anim:'drift1' },
        { w:620, h:480, bg:'#5e14b8', bottom:'-100px', right:'-120px', anim:'drift2' },
        { w:520, h:420, bg:'#130847', top:'35%', left:'42%', anim:'drift3' },
        { w:420, h:360, bg:'#3d0d90', top:'58%', left:'8%', anim:'drift4' },
    ];
    orbs.forEach(function(o) {
        var d = p.createElement('div');
        var pos = '';
        if (o.top !== undefined)    pos += 'top:' + o.top + ';';
        if (o.bottom !== undefined) pos += 'bottom:' + o.bottom + ';';
        if (o.left !== undefined)   pos += 'left:' + o.left + ';';
        if (o.right !== undefined)  pos += 'right:' + o.right + ';';
        d.style.cssText = [
            'position:absolute',
            'width:' + o.w + 'px',
            'height:' + o.h + 'px',
            'border-radius:50%',
            'background:radial-gradient(ellipse,' + o.bg + ' 0%,transparent 70%)',
            'filter:blur(85px)',
            'opacity:0.60',
            'animation:' + o.anim + ' 18s ease-in-out infinite alternate',
            pos
        ].join(';');
        aurora.appendChild(d);
    });

    // Keyframe CSS
    var style = p.createElement('style');
    style.textContent = `
        @keyframes drift1 {
            0%   { transform: translate(0,0) scale(1); }
            50%  { transform: translate(90px,70px) scale(1.14); }
            100% { transform: translate(35px,130px) scale(0.94); }
        }
        @keyframes drift2 {
            0%   { transform: translate(0,0) scale(1); }
            40%  { transform: translate(-110px,-90px) scale(1.12); }
            100% { transform: translate(-55px,-150px) scale(1.06); }
        }
        @keyframes drift3 {
            0%   { transform: translate(0,0) scale(1); }
            60%  { transform: translate(70px,-80px) scale(1.18); }
            100% { transform: translate(-45px,55px) scale(0.88); }
        }
        @keyframes drift4 {
            0%   { transform: translate(0,0) scale(1); }
            50%  { transform: translate(100px,-60px) scale(1.1); }
            100% { transform: translate(25px,-110px) scale(1.12); }
        }
    `;
    p.head.appendChild(style);

    // Insert aurora as the very first child of body so it sits behind everything
    p.body.insertBefore(aurora, p.body.firstChild);

    // Make sure Streamlit's own containers are transparent
    var patch = p.createElement('style');
    patch.textContent = `
        html, body { background: #06030f !important; }
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background: transparent !important;
        }
        /* Keep sidebar opaque */
        section[data-testid="stSidebar"] {
            background: rgba(7,3,20,0.88) !important;
            backdrop-filter: blur(22px) !important;
            border-right: 1px solid rgba(120,80,220,0.14) !important;
        }
    `;
    p.head.appendChild(patch);
})();
</script>
"""
components.html(aurora_injector, height=0)

# --- 3. MAIN CSS (content styling only — background handled above) ---
custom_css = """
<style>
    /* ── FONTS & BASE ── */
    * { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3, .hero-brand, .result-label {
        font-family: 'DM Serif Display', serif !important;
    }

    /* ── LAYOUT ── */
    .block-container {
        max-width: 800px !important;
        padding-top: 2.5rem !important;
        padding-bottom: 4rem !important;
    }

    /* ── HEADINGS ── */
    h1 {
        font-size: 3.2rem !important; font-weight: 400 !important;
        letter-spacing: -0.03em !important; line-height: 1.08 !important;
        color: #f0eaff !important; margin-bottom: 6px !important;
    }
    h2 {
        font-size: 1.9rem !important; font-weight: 400 !important;
        color: #e0d4f8 !important; letter-spacing: -0.02em !important;
    }
    h3 { font-size: 1.15rem !important; color: #b89ef0 !important; font-weight: 400 !important; }
    p, li, label { font-size: 0.98rem !important; line-height: 1.72 !important; color: #c0b8d8 !important; }

    /* ── HERO ── */
    .hero-brand {
        font-size: 5.8rem !important; font-weight: 400;
        letter-spacing: -0.04em; line-height: 1;
        background: linear-gradient(135deg, #e8e0ff 0%, #a78bfa 45%, #7c3aed 75%, #c4b5f4 100%);
        background-size: 300% 300%;
        animation: brandGrad 6s ease infinite;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        display: block; text-align: center;
    }
    @keyframes brandGrad {
        0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; }
    }
    .hero-sub {
        font-size: 0.76rem; letter-spacing: 0.2em; text-transform: uppercase;
        color: #4e4470; text-align: center; margin-top: 7px; display: block;
    }

    /* ── DIVIDER ── */
    .divider {
        border: none; height: 1px; margin: 26px 0;
        background: linear-gradient(to right, transparent, rgba(140,100,255,0.22), transparent);
    }

    /* ── GLASS CARDS ── */
    .glass-card {
        background: rgba(12, 6, 28, 0.58);
        border: 1px solid rgba(130, 90, 240, 0.15);
        border-radius: 16px;
        backdrop-filter: blur(28px); -webkit-backdrop-filter: blur(28px);
        padding: 34px 38px; margin-bottom: 18px;
    }

    /* ── EYEBROW LABELS ── */
    .eyebrow {
        font-size: 0.66rem !important; letter-spacing: 0.2em !important;
        text-transform: uppercase !important; color: #4e4470 !important;
        margin-bottom: 7px !important; display: block;
    }

    /* ── TAG PILLS ── */
    .tag-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
    .tag-pill {
        font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
        color: #a78bfa; background: rgba(100,60,200,0.10);
        border: 1px solid rgba(100,60,200,0.22); padding: 4px 13px; border-radius: 100px;
    }

    /* ── TRUST BADGES ── */
    .trust-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
    .trust-badge {
        display: flex; align-items: center; gap: 8px;
        background: rgba(16,8,40,0.72); border: 1px solid rgba(100,70,180,0.18);
        border-radius: 9px; padding: 8px 13px;
        font-size: 0.78rem; color: #9e96be;
    }

    /* ── SURVEY BOX ── */
    .survey-box {
        background: rgba(60,30,120,0.09); border: 1px solid rgba(140,100,255,0.17);
        border-radius: 14px; padding: 32px 36px; text-align: center; margin-top: 6px;
    }
    .survey-link {
        display: inline-block; color: #c4b5f4 !important;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.12em;
        text-transform: uppercase; text-decoration: none;
        padding: 11px 26px; border: 1px solid rgba(160,130,255,0.28);
        border-radius: 8px; background: rgba(100,60,200,0.10); transition: all 0.2s;
    }
    .survey-link:hover { background: rgba(100,60,200,0.26); }

    /* ── BUTTONS ── */
    .stButton > button {
        background: linear-gradient(135deg, #4e1d9e, #7c3aed) !important;
        color: #ede8ff !important; font-size: 0.82rem !important;
        font-weight: 600 !important; letter-spacing: 0.1em !important;
        text-transform: uppercase !important; padding: 13px 28px !important;
        border-radius: 10px !important; border: none !important; width: 100%;
        box-shadow: 0 4px 18px rgba(90,30,180,0.35) !important; transition: all 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 26px rgba(90,30,180,0.52) !important;
    }

    /* ── TEXTAREA ── */
    .stTextArea > div > div > textarea {
        background: rgba(10,4,24,0.78) !important;
        border: 1px solid rgba(100,70,180,0.24) !important;
        border-radius: 10px !important; color: #dcd4f0 !important;
        font-size: 0.94rem !important; line-height: 1.65 !important;
    }
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(140,100,255,0.48) !important;
        box-shadow: 0 0 0 3px rgba(90,50,190,0.12) !important;
    }

    /* ── RADIO ── */
    .stRadio label { font-size: 0.88rem !important; color: #b0a8cc !important; }

    /* ── METRICS ── */
    [data-testid="stMetric"] {
        background: rgba(14,7,34,0.68) !important;
        border: 1px solid rgba(100,70,180,0.17) !important;
        border-radius: 12px !important; padding: 15px !important;
    }
    [data-testid="stMetricLabel"] > div {
        font-size: 0.68rem !important; text-transform: uppercase !important;
        letter-spacing: 0.1em !important; color: #5e5280 !important;
    }
    [data-testid="stMetricValue"] > div {
        font-family: 'DM Serif Display', serif !important;
        font-size: 1.8rem !important; color: #c4b5f4 !important;
    }

    /* ── PROGRESS ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #4e1d9e, #a78bfa) !important;
    }

    /* ── RESULT LABEL ── */
    .result-label {
        font-size: 2.5rem; color: #c4b5f4; line-height: 1.1; margin: 6px 0 0;
    }

    /* ── EVIDENCE ── */
    .evidence-item {
        background: rgba(14,7,34,0.55); border-left: 3px solid #7c3aed;
        border-radius: 0 10px 10px 0; padding: 13px 17px; margin-bottom: 11px;
    }
    .evidence-text { font-size: 0.97rem !important; color: #d0c8e8 !important; line-height: 1.6 !important; }
    .evidence-score { font-size: 0.7rem; color: #6e54a0; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 4px; }

    /* ── SIDEBAR ── */
    section[data-testid="stSidebar"] h1 {
        font-family: 'DM Serif Display', serif !important;
        font-size: 1.85rem !important; color: #c4b5f4 !important; letter-spacing: -0.02em !important;
    }
    section[data-testid="stSidebar"] p { font-size: 0.86rem !important; color: #8880a8 !important; }
    section[data-testid="stSidebar"] h3 {
        font-size: 0.66rem !important; text-transform: uppercase !important;
        letter-spacing: 0.16em !important; color: #4e4470 !important; font-weight: 600 !important;
    }
    .nav-item {
        display: flex; align-items: center; gap: 9px;
        padding: 7px 11px; border-radius: 7px; color: #9e96be;
        font-size: 0.84rem; margin-bottom: 3px;
    }
    .nav-dot { width: 5px; height: 5px; border-radius: 50%; background: rgba(130,100,220,0.45); flex-shrink: 0; }

    /* ── HIDE STREAMLIT CHROME ── */
    #MainMenu, footer { visibility: hidden; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if 'show_demo' not in st.session_state:
    st.session_state.show_demo = False

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("<h1>Insyte</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border:none;height:1px;background:linear-gradient(to right,transparent,rgba(120,80,210,0.2),transparent);margin:10px 0 16px;'>", unsafe_allow_html=True)
    st.markdown("<h3>Clinical Workflow</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class='nav-item'><div class='nav-dot'></div>Patient assessment tracking</div>
    <div class='nav-item'><div class='nav-dot'></div>Treatment progress monitoring</div>
    <div class='nav-item'><div class='nav-dot'></div>Structured clinical check-ins</div>
    <div class='nav-item'><div class='nav-dot'></div>Documentation support</div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border:none;height:1px;background:linear-gradient(to right,transparent,rgba(120,80,210,0.2),transparent);margin:16px 0;'>", unsafe_allow_html=True)
    if st.button("← Home"):
        st.session_state.show_demo = False
        st.rerun()
    st.markdown("<p style='font-size:0.7rem !important; color:#332d50 !important; margin-top:12px; letter-spacing:0.06em;'>Pilot v1.2 · Clinical Validation Phase</p>", unsafe_allow_html=True)

# --- 5. CORE ENGINE (UNCHANGED) ---
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

# --- 6. LANDING PAGE ---
if not st.session_state.show_demo:

    # HERO
    st.markdown("""
    <div style="text-align:center; padding: 28px 0 12px;">
        <span class="hero-brand">Insyte</span>
        <span class="hero-sub">Early Linguistic Examiner &nbsp;·&nbsp; Clinical Pilot</span>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    # ANIMATED STAT COUNTER
    stat_html = """
    <div style="text-align:center; padding:28px 20px 22px; font-family:'DM Sans',sans-serif;">
        <div style="font-size:0.68rem; letter-spacing:0.2em; text-transform:uppercase; color:#4a3f68; margin-bottom:16px;">The Scale of the Problem</div>
        <div style="display:flex; align-items:center; justify-content:center; gap:22px; flex-wrap:wrap;">
            <div id="stat-num" style="
                font-family:'DM Serif Display',serif;
                font-size:6rem; line-height:1;
                background:linear-gradient(135deg,#c4b5f4 0%,#7c3aed 60%);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
            ">0</div>
            <div style="font-size:0.92rem; color:#7870a0; max-width:190px; text-align:left; line-height:1.55;">
                people die by suicide annually in Canada
                <div style="font-size:0.7rem; color:#3e3560; margin-top:5px; letter-spacing:0.07em;">— Statistics Canada</div>
            </div>
        </div>
    </div>
    <script>
    (function(){
        var start=null, dur=2800, target=4500;
        var el=document.getElementById('stat-num');
        function ease(t){ return 1-Math.pow(1-t,4); }
        function step(ts){
            if(!start) start=ts;
            var p=Math.min((ts-start)/dur,1);
            el.textContent=Math.floor(ease(p)*target).toLocaleString();
            if(p<1) requestAnimationFrame(step);
            else el.textContent=target.toLocaleString();
        }
        requestAnimationFrame(step);
    })();
    </script>
    """
    components.html(stat_html, height=185)

    # MISSION CARD
    st.markdown("""
    <div class="glass-card">
        <div class="tag-row">
            <span class="tag-pill">For Psychologists</span>
            <span class="tag-pill">Mental Health Clinicians</span>
            <span class="tag-pill">Written Intake Analysis</span>
        </div>
        <span class="eyebrow">The Mission</span>
        <p style="font-size:1.05rem !important; color:#d4cce8 !important; margin-bottom:13px;">
            Clinicians face relentless caseloads and documentation pressure. Early warning signals buried in patient language go undetected — not from lack of skill, but lack of time.
        </p>
        <p style="color:#9890b8 !important; margin-bottom:0;">
            Insyte builds AI-assisted tools that surface structured clinical insight from written intake and assessment materials — giving you a sharper lens, faster.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # TRUST CARD
    st.markdown("""
    <div class="glass-card">
        <span class="eyebrow">Why Insyte</span>
        <div class="trust-row">
            <div class="trust-badge">🔍&nbsp; Linguistic signal detection</div>
            <div class="trust-badge">🤖&nbsp; RoBERTa + ReHAN hybrid</div>
            <div class="trust-badge">🔒&nbsp; Anonymized inputs only</div>
            <div class="trust-badge">📋&nbsp; Structured for clinical review</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # CTA BUTTON
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        if st.button("Try the Demo →"):
            st.session_state.show_demo = True
            st.rerun()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # SURVEY BOX
    st.markdown(f"""
    <div class="survey-box">
        <span class="eyebrow" style="text-align:center; display:block;">Shape This Tool</span>
        <h2 style="margin:8px 0 10px !important;">Help shape what this becomes.</h2>
        <p style="color:#6a608a !important; font-size:0.9rem !important; margin-bottom:22px;">
            This is a clinical pilot. Your feedback — from clinicians, for clinicians — defines the roadmap.
        </p>
        <a href="{survey_url}" class="survey-link" target="_blank">Share Your Feedback →</a>
    </div>
    """, unsafe_allow_html=True)

# --- 7. DEMO TOOL ---
else:
    st.markdown("""
    <div style="margin-bottom:8px;">
        <span class="eyebrow">Insyte · Early Linguistic Examiner</span>
        <h1 style="font-size:2.5rem !important; margin-bottom:4px !important;">Patient Discourse Analysis</h1>
        <p style="color:#4a4068 !important; font-size:0.83rem !important; margin-top:0 !important;">
            Clinical Pilot · Semantic Analysis Engine · Handle all inputs with care
        </p>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

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
                <div class="glass-card" style="margin-top:20px;">
                    <span class="eyebrow">Severity Assessment</span>
                    <div class="result-label">{final_label}</div>
                </div>
                """, unsafe_allow_html=True)
                cols = st.columns(3)
                cols[0].metric("RoBERTa Score", f"{p_severe:.2f}")
                cols[1].metric("ReHAN Signal", f"{rehan_signal:.2f}")
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
                <div class="glass-card" style="margin-top:20px;">
                    <span class="eyebrow">Primary Thematic Determinant</span>
                    <div class="result-label">{result_cause}</div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(hybrid_cau)

            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("<span class='eyebrow'>Linguistic Evidence · High-Signal Passages</span>", unsafe_allow_html=True)

            imp_list = importance if isinstance(importance, list) else [importance]
            found = False
            for score, sent in zip(imp_list, raw_sentences):
                if score > 0.15:
                    found = True
                    st.markdown(f"""
                    <div class="evidence-item">
                        <div class="evidence-text">{sent}</div>
                        <div class="evidence-score">Signal Intensity · {score:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
            if not found:
                st.markdown("<p style='color:#3e3560 !important; font-size:0.86rem !important;'>No passages exceeded signal threshold.</p>", unsafe_allow_html=True)

            st.markdown("""
            <p style="font-size:0.73rem !important; color:#2e2848 !important; margin-top:22px; line-height:1.5 !important;">
            ⚠ This tool supports clinical decision-making only. It does not replace clinical judgment, diagnosis, or risk assessment protocols. All inputs must be anonymized.
            </p>
            """, unsafe_allow_html=True)

        else:
            st.warning("Please enter patient discourse text to analyze.")