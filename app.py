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
aurora_injector = """
<script>
(function() {
    var p = window.parent.document;
    if (p.getElementById('insyte-aurora')) return;

    var font = p.createElement('link');
    font.rel = 'stylesheet';
    font.href = 'https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap';
    p.head.appendChild(font);

    var aurora = p.createElement('div');
    aurora.id = 'insyte-aurora';
    aurora.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:0;pointer-events:none;overflow:hidden;background:#06030f';

    var orbs = [
        { w:900, h:650, bg:'#2e17a0', top:'-150px', left:'-200px', anim:'drift1' },
        { w:750, h:580, bg:'#5e14b8', bottom:'-120px', right:'-140px', anim:'drift2' },
        { w:600, h:500, bg:'#130847', top:'30%', left:'38%', anim:'drift3' },
        { w:500, h:420, bg:'#3d0d90', top:'55%', left:'5%', anim:'drift4' },
    ];
    orbs.forEach(function(o) {
        var d = p.createElement('div');
        var pos = '';
        if (o.top    !== undefined) pos += 'top:'    + o.top    + ';';
        if (o.bottom !== undefined) pos += 'bottom:' + o.bottom + ';';
        if (o.left   !== undefined) pos += 'left:'   + o.left   + ';';
        if (o.right  !== undefined) pos += 'right:'  + o.right  + ';';
        d.style.cssText = 'position:absolute;width:' + o.w + 'px;height:' + o.h + 'px;border-radius:50%;background:radial-gradient(ellipse,' + o.bg + ' 0%,transparent 70%);filter:blur(90px);opacity:0.65;animation:' + o.anim + ' 20s ease-in-out infinite alternate;' + pos;
        aurora.appendChild(d);
    });

    var style = p.createElement('style');
    style.textContent = `
        @keyframes drift1 {
            0%   { transform: translate(0px,0px) scale(1); }
            33%  { transform: translate(100px,80px) scale(1.16); }
            66%  { transform: translate(50px,160px) scale(0.92); }
            100% { transform: translate(120px,60px) scale(1.08); }
        }
        @keyframes drift2 {
            0%   { transform: translate(0px,0px) scale(1); }
            33%  { transform: translate(-120px,-100px) scale(1.14); }
            66%  { transform: translate(-60px,-170px) scale(1.04); }
            100% { transform: translate(-90px,-80px) scale(1.1); }
        }
        @keyframes drift3 {
            0%   { transform: translate(0px,0px) scale(1); }
            40%  { transform: translate(80px,-90px) scale(1.2); }
            80%  { transform: translate(-50px,65px) scale(0.86); }
            100% { transform: translate(30px,-40px) scale(1.05); }
        }
        @keyframes drift4 {
            0%   { transform: translate(0px,0px) scale(1); }
            50%  { transform: translate(110px,-70px) scale(1.12); }
            100% { transform: translate(30px,-130px) scale(1.15); }
        }
    `;
    p.head.appendChild(style);
    p.body.insertBefore(aurora, p.body.firstChild);

    var patch = p.createElement('style');
    patch.textContent = `
        html, body { background: #06030f !important; }
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background: transparent !important;
        }
        section[data-testid="stSidebar"] {
            background: rgba(7,3,20,0.90) !important;
            backdrop-filter: blur(24px) !important;
            border-right: 1px solid rgba(120,80,220,0.14) !important;
        }
    `;
    p.head.appendChild(patch);
})();
</script>
"""
components.html(aurora_injector, height=0)

# --- 3. MAIN CSS ---
custom_css = """
<style>
    * { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3, .hero-brand, .result-label {
        font-family: 'DM Serif Display', serif !important;
    }

    .block-container {
        max-width: 820px !important;
        padding-top: 3rem !important;
        padding-bottom: 5rem !important;
    }

    /* ── HEADINGS ── */
    h1 {
        font-size: 3.4rem !important; font-weight: 400 !important;
        letter-spacing: -0.03em !important; line-height: 1.08 !important;
        color: #f0eaff !important; margin-bottom: 6px !important;
    }
    h2 {
        font-size: 2rem !important; font-weight: 400 !important;
        color: #e0d4f8 !important; letter-spacing: -0.02em !important;
        margin-bottom: 10px !important;
    }
    h3 { font-size: 1.2rem !important; color: #b89ef0 !important; font-weight: 400 !important; }

    /* ── BODY TEXT — readable size ── */
    p, li, label {
        font-size: 1.1rem !important;
        line-height: 1.75 !important;
        color: #cdc5e2 !important;
    }

    /* ── HERO ── */
    .hero-wrap {
        text-align: center;
        padding: 32px 0 14px;
    }
    .hero-brand {
        font-family: 'DM Serif Display', serif !important;
        font-size: 7rem !important;
        font-weight: 400;
        letter-spacing: -0.04em;
        line-height: 1;
        /* Full word gradient — wide enough to cover every letter including Y */
        background: linear-gradient(110deg, #f0ebff 0%, #c4b5f4 30%, #8b5cf6 60%, #a78bfa 80%, #e0d4ff 100%);
        background-size: 200% 100%;
        animation: brandGrad 5s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        display: inline-block;
        /* padding prevents the descender/ascender clip */
        padding: 0 4px 6px 4px;
    }
    @keyframes brandGrad {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .hero-sub {
        font-size: 0.8rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: #4e4470;
        margin-top: 8px;
        display: block;
    }

    /* ── DIVIDER ── */
    .divider {
        border: none; height: 1px; margin: 28px 0;
        background: linear-gradient(to right, transparent, rgba(140,100,255,0.22), transparent);
    }

    /* ── GLASS CARD ── */
    .glass-card {
        background: rgba(12, 6, 28, 0.60);
        border: 1px solid rgba(130, 90, 240, 0.15);
        border-radius: 16px;
        backdrop-filter: blur(28px); -webkit-backdrop-filter: blur(28px);
        padding: 36px 42px;
        margin-bottom: 20px;
    }

    /* ── EYEBROW ── */
    .eyebrow {
        font-size: 0.7rem !important; letter-spacing: 0.2em !important;
        text-transform: uppercase !important; color: #5a5080 !important;
        margin-bottom: 8px !important; display: block;
    }

    /* ── BULLET LIST — Why Insyte ── */
    .why-list {
        list-style: none;
        padding: 0; margin: 0;
    }
    .why-list li {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        padding: 10px 0;
        border-bottom: 1px solid rgba(130,90,240,0.08);
        font-size: 1.05rem !important;
        color: #cdc5e2 !important;
        line-height: 1.6 !important;
    }
    .why-list li:last-child { border-bottom: none; }
    .why-list .bullet {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #7c3aed;
        flex-shrink: 0;
        margin-top: 9px;
    }

    /* ── SURVEY BOX ── */
    .survey-box {
        background: rgba(60,30,120,0.10);
        border: 1px solid rgba(140,100,255,0.16);
        border-radius: 14px;
        padding: 36px 42px;
        text-align: center;
        margin-top: 6px;
    }
    .survey-box p {
        font-size: 1.1rem !important;
        color: #b8b0d0 !important;
        margin-bottom: 20px !important;
        line-height: 1.65 !important;
    }
    .survey-link {
        display: inline-block;
        color: #c4b5f4 !important;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        text-decoration: none;
        padding: 12px 28px;
        border: 1px solid rgba(160,130,255,0.30);
        border-radius: 8px;
        background: rgba(100,60,200,0.12);
        transition: all 0.2s;
    }
    .survey-link:hover { background: rgba(100,60,200,0.28); }

    /* ── DEMO BUTTON — centred via HTML, not st.columns ── */
    .demo-btn-wrap {
        display: flex;
        justify-content: center;
        margin: 8px 0 20px;
    }

    /* ── BUTTONS ── */
    .stButton > button {
        background: linear-gradient(135deg, #4e1d9e, #7c3aed) !important;
        color: #ede8ff !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        padding: 14px 36px !important;
        border-radius: 10px !important;
        border: none !important;
        width: auto !important;
        min-width: 220px;
        box-shadow: 0 4px 20px rgba(90,30,180,0.38) !important;
        transition: all 0.2s !important;
        display: block !important;
        margin: 0 auto !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(90,30,180,0.55) !important;
    }

    /* ── TEXTAREA ── */
    .stTextArea > div > div > textarea {
        background: rgba(10,4,24,0.80) !important;
        border: 1px solid rgba(100,70,180,0.25) !important;
        border-radius: 10px !important;
        color: #dcd4f0 !important;
        font-size: 1rem !important;
        line-height: 1.7 !important;
    }
    .stTextArea > div > div > textarea:focus {
        border-color: rgba(140,100,255,0.50) !important;
        box-shadow: 0 0 0 3px rgba(90,50,190,0.12) !important;
    }
    /* Textarea label */
    .stTextArea label p { font-size: 1rem !important; color: #9e96be !important; }

    /* ── RADIO ── */
    .stRadio label { font-size: 1rem !important; color: #b0a8cc !important; }

    /* ── METRICS ── */
    [data-testid="stMetric"] {
        background: rgba(14,7,34,0.70) !important;
        border: 1px solid rgba(100,70,180,0.18) !important;
        border-radius: 12px !important;
        padding: 18px !important;
    }
    [data-testid="stMetricLabel"] > div {
        font-size: 0.72rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
        color: #6a5f8a !important;
    }
    [data-testid="stMetricValue"] > div {
        font-family: 'DM Serif Display', serif !important;
        font-size: 2rem !important;
        color: #c4b5f4 !important;
    }

    /* ── PROGRESS ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #4e1d9e, #a78bfa) !important;
    }

    /* ── RESULT LABEL ── */
    .result-label {
        font-size: 2.8rem;
        color: #c4b5f4;
        line-height: 1.1;
        margin: 8px 0 0;
    }

    /* ── EVIDENCE ── */
    .evidence-item {
        background: rgba(14,7,34,0.58);
        border-left: 3px solid #7c3aed;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .evidence-text { font-size: 1rem !important; color: #d0c8e8 !important; line-height: 1.65 !important; }
    .evidence-score { font-size: 0.72rem; color: #7054a8; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 5px; }

    /* ── SIDEBAR ── */
    section[data-testid="stSidebar"] h1 {
        font-family: 'DM Serif Display', serif !important;
        font-size: 2rem !important; color: #c4b5f4 !important; letter-spacing: -0.02em !important;
    }
    section[data-testid="stSidebar"] p { font-size: 0.9rem !important; color: #8880a8 !important; }
    section[data-testid="stSidebar"] h3 {
        font-size: 0.68rem !important; text-transform: uppercase !important;
        letter-spacing: 0.16em !important; color: #4e4470 !important; font-weight: 600 !important;
    }
    .nav-item {
        display: flex; align-items: center; gap: 10px;
        padding: 8px 12px; border-radius: 7px; color: #a098c0;
        font-size: 0.9rem; margin-bottom: 4px;
    }
    .nav-dot {
        width: 5px; height: 5px; border-radius: 50%;
        background: rgba(130,100,220,0.5); flex-shrink: 0;
    }

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
    st.markdown("<hr style='border:none;height:1px;background:linear-gradient(to right,transparent,rgba(120,80,210,0.2),transparent);margin:10px 0 18px;'>", unsafe_allow_html=True)
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
    st.markdown("<p style='font-size:0.72rem !important; color:#332d50 !important; margin-top:14px; letter-spacing:0.06em;'>Pilot v1.2 · Clinical Validation Phase</p>", unsafe_allow_html=True)

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

# ============================================================
# --- 6. LANDING PAGE ---
# ============================================================
if not st.session_state.show_demo:

    # HERO — inline-block with padding so gradient covers full "Insyte" incl. Y
    st.markdown("""
    <div class="hero-wrap">
        <span class="hero-brand">Insyte</span>
        <span class="hero-sub">Early Linguistic Examiner &nbsp;·&nbsp; Clinical Pilot</span>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    # STAT COUNTER
    stat_html = """
    <div style="text-align:center; padding:30px 20px 24px; font-family:'DM Sans',sans-serif;">
        <div style="font-size:0.7rem; letter-spacing:0.2em; text-transform:uppercase; color:#4a3f68; margin-bottom:18px;">The Scale of the Problem</div>
        <div style="display:flex; align-items:center; justify-content:center; gap:24px; flex-wrap:wrap;">
            <div id="stat-num" style="
                font-family:'DM Serif Display',serif;
                font-size:6.5rem; line-height:1;
                background:linear-gradient(135deg,#c4b5f4 0%,#7c3aed 60%);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
            ">0</div>
            <div style="font-size:1rem; color:#8880a8; max-width:210px; text-align:left; line-height:1.6;">
                people die by suicide annually in Canada
                <div style="font-size:0.75rem; color:#3e3560; margin-top:6px; letter-spacing:0.07em;">— Statistics Canada</div>
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
    components.html(stat_html, height=200)

    # MISSION CARD — exact copy text, no keyword bubbles
    st.markdown("""
    <div class="glass-card">
        <span class="eyebrow">About Insyte</span>
        <p style="color:#cdc5e2 !important; margin-bottom:14px;">
            Clinicians face increasing caseloads and documentation burdens, making early signal detection difficult.
        </p>
        <p style="color:#cdc5e2 !important; margin-bottom:14px;">
            Insyte builds AI-assisted tools to support structured clinical insight for psychologists and mental health clinicians reviewing written intake and assessment materials.
        </p>
        <p style="color:#cdc5e2 !important; margin-bottom:0;">
            Analyze patient discourse to identify symptom severity and potential contributing factors in seconds.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # WHY INSYTE — clean bullet list, no emojis
    st.markdown("""
    <div class="glass-card">
        <span class="eyebrow">Why Insyte</span>
        <ul class="why-list">
            <li><span class="bullet"></span>Linguistic signal detection from written patient discourse</li>
            <li><span class="bullet"></span>Hybrid RoBERTa + ReHAN model architecture</li>
            <li><span class="bullet"></span>Anonymized inputs — no identifiable data stored</li>
            <li><span class="bullet"></span>Structured output designed for clinical review workflows</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # DEMO BUTTON — centred using HTML flex wrapper + single st.button
    st.markdown('<div class="demo-btn-wrap">', unsafe_allow_html=True)
    if st.button("Try the Demo →"):
        st.session_state.show_demo = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # SURVEY BOX
    st.markdown(f"""
    <div class="survey-box">
        <p>Your feedback is important, help shape the future.</p>
        <a href="{survey_url}" class="survey-link" target="_blank">Share Your Feedback →</a>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# --- 7. DEMO TOOL VIEW ---
# ============================================================
else:
    st.markdown("""
    <div style="margin-bottom:10px;">
        <span class="eyebrow">Insyte · Early Linguistic Examiner</span>
        <h1 style="font-size:2.6rem !important; margin-bottom:6px !important;">Patient Discourse Analysis</h1>
        <p style="color:#5a5080 !important; font-size:0.88rem !important; margin-top:0 !important;">
            Clinical Pilot · Semantic Analysis Engine · All inputs must be anonymized
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
        height=240,
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
                <div class="glass-card" style="margin-top:22px;">
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
                <div class="glass-card" style="margin-top:22px;">
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
                st.markdown("<p style='color:#4a4068 !important; font-size:0.9rem !important;'>No passages exceeded signal threshold.</p>", unsafe_allow_html=True)

            st.markdown("""
            <p style="font-size:0.82rem !important; color:#4a4068 !important; margin-top:24px; line-height:1.6 !important;">
            This tool supports clinical decision-making only. It does not replace clinical judgment, diagnosis, or risk assessment protocols. All inputs must be anonymized prior to entry.
            </p>
            """, unsafe_allow_html=True)

        else:
            st.warning("Please enter patient discourse text to analyze.")