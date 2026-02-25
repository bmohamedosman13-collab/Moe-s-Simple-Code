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

# --- 2. STYLING ---
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500&display=swap');

    .stApp {
        background: linear-gradient(-45deg, #141018, #1B1622, #221A2B, #2A2034);
        background-size: 400% 400%;
        animation: gradientMove 32s ease infinite;
        color: #f5f5f5;
        font-family: 'Playfair Display', serif;
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Sidebar smaller font */
    section[data-testid="stSidebar"] * {
        font-size: 0.95rem !important;
    }

    h1 { font-size: 4.5rem !important; font-weight: 700 !important; }
    h2 { font-size: 3rem !important; font-weight: 400 !important; }
    h3 { font-size: 2.2rem !important; font-weight: 400 !important; color: #9C8ACF !important; }

    p, li, label {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem !important;
        line-height: 1.8 !important;
        color: #D6D3DD;
    }

    .survey-box {
        background-color: rgba(156, 138, 207, 0.12);
        border: 1px solid #9C8ACF;
        padding: 30px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 40px;
    }

    .survey-link {
        color: #9C8ACF !important;
        font-size: 1.8rem !important;
        font-weight: 700;
        text-decoration: underline !important;
    }

    .stButton>button {
        background-color: #9C8ACF !important;
        color: #141018 !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        padding: 12px 30px !important;
        border-radius: 8px !important;
        border: none !important;
        width: 100%;
    }

    hr { border-top: 1px solid rgba(255,255,255,0.08); margin: 3rem 0; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'show_demo' not in st.session_state:
    st.session_state.show_demo = False

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='font-size: 2rem !important;'>Insyte</h1>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Clinical Workflow Roadmap")
    st.markdown("""
    * Patient assessment tracking
    * Treatment progress monitoring
    * Structured check-ins
    * Workflow and documentation support
    """)
    st.markdown("---")
    if st.button("Home"):
        st.session_state.show_demo = False
        st.rerun()
    st.markdown("---")
    st.markdown("<p style='font-size:0.8rem; opacity:0.7;'>Pilot v1.2 | Clinical Validation Phase</p>", unsafe_allow_html=True)

# ==========================================
# CORE ENGINE SETUP
# ==========================================
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
    base_path = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.join(base_path, "rehan_severity_vocab.pkl"), "rb") as f: v_sev = pickle.load(f)
        with open(os.path.join(base_path, "rehan_cause_vocab.pkl"), "rb") as f: v_cau = pickle.load(f)

        r_sev = ReHAN(len(v_sev), 200, 128, 4).to(device)
        r_sev.load_state_dict(torch.load(os.path.join(base_path, "rehan_severity.pt"), map_location=device))
        r_sev.eval()

        r_cau = ReHAN(len(v_cau), 200, 128, 6).to(device)
        r_cau.load_state_dict(torch.load(os.path.join(base_path, "rehan_cause.pt"), map_location=device))
        r_cau.eval()

        rob_sev = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)
        rob_sev.eval()

        rob_cau = RobertaForSequenceClassification.from_pretrained("roberta-base").to(device)
        rob_cau.eval()

        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        return r_sev, r_cau, rob_sev, rob_cau, tokenizer, v_sev, v_cau, None
    except Exception as e:
        return None, None, None, None, None, None, None, str(e)

r_sev, r_cau, rob_sev, rob_cau, tokenizer, v_sev, v_cau, error_msg = load_resources()

# ==========================================
# MAIN VIEW
# ==========================================

if not st.session_state.show_demo:

    st.markdown("<h1 style='text-align:center;'>Insyte</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1.6rem; opacity:0.8;'>Early Linguistic Examiner</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    _, col_btn, _ = st.columns([1,2,1])
    with col_btn:
        if st.button("Demo the tool →", use_container_width=True):
            st.session_state.show_demo = True
            st.rerun()

    st.markdown(f"""
    <div class="survey-box">
        <h2>Help shape the future.</h2>
        <p>This is a clinical pilot. Your feedback is essential.</p>
        <a href="{survey_url}" class="survey-link" target="_blank">Feedback Survey →</a>
    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown("<h1 style='font-size:3rem;'>Early Linguistic Examiner</h1>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    user_input = st.text_area("Patient Discourse Input:", height=250)

    if st.button("Generate Insights") and user_input.strip():

        inputs = tokenizer([user_input], return_tensors="pt", truncation=True, padding=True).to(device)

        with torch.no_grad():
            sev_out = rob_sev(**inputs)
            sev_probs = F.softmax(sev_out.logits, dim=1)[0]
            p_non_severe, p_severe = sev_probs.tolist()

        rehan_signal = 0.3
        raw_hybrid = (p_severe * 0.6) + (rehan_signal * 0.4)

        if raw_hybrid < 0.3:
            final_label = "Minimum"
        elif raw_hybrid < 0.5:
            final_label = "Mild"
        elif raw_hybrid < 0.75:
            final_label = "Moderate"
        else:
            final_label = "Severe"

        st.markdown(f"### Assessment: <span style='color:#9C8ACF;'>{final_label}</span>", unsafe_allow_html=True)
        st.metric("RoBERTa Severe Probability", f"{p_severe:.2f}")
        st.metric("Final Hybrid Index", f"{raw_hybrid:.2f}")