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
# PASTE YOUR SURVEY LINK HERE
survey_url = "https://docs.google.com/forms/d/e/1FAIpQLScJZaX8ZrrIHv6XKB-sVPpIJcJlMUl7GUb5mbGCBQYzAj-WPQ/viewform" 

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Insyte | Early Linguistic Examiner",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS (PLAYFAIR DISPLAY & LARGE TEXT) ---
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400&display=swap');

    /* Global Typography */
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #212121 40%, #3a3258 100%);
        color: #f5f5f5;
        font-family: 'Inter', sans-serif;
        font-size: 1.15rem; /* Larger base text */
    }

    /* Headings in Playfair Display */
    h1, h2, h3, h4 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 700 !important;
        color: #f5f5f5;
    }

    /* Larger body text for better readability */
    p, li, label, div {
        font-size: 1.15rem !important;
        line-height: 1.6 !important;
    }

    /* Thin Lines and Spacing */
    hr {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.1);
        margin: 2.5rem 0;
    }

    /* Premium Button Style */
    .stButton>button {
        background-color: transparent;
        color: #8d7dca;
        border: 1px solid #8d7dca;
        border-radius: 4px;
        transition: all 0.3s ease;
        padding: 0.75rem 2.5rem;
        font-family: 'Playfair Display', serif;
        font-size: 1.2rem !important;
    }
    .stButton>button:hover {
        background-color: #8d7dca;
        color: #1a1a1a;
        border: 1px solid #8d7dca;
    }

    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #1a1a1a;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if 'show_demo' not in st.session_state:
    st.session_state.show_demo = False

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("## Insyte")
    st.markdown("### Upcoming Features")
    st.markdown("""
    * Patient assessment tracking
    * Treatment progress monitoring
    * Structured check-ins
    * Workflow and documentation support
    """)
    st.markdown("---")
    if st.button("🏠 Home / Info"):
        st.session_state.show_demo = False
        st.rerun()
    st.markdown("---")
    st.info("Early Pilot: Clinical feedback version.")

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
    try:
        current_dir = os.getcwd() # Explicit pathing to root
        with open("rehan_severity_vocab.pkl", "rb") as f: v_sev = pickle.load(f)
        with open("rehan_cause_vocab.pkl", "rb") as f: v_cau = pickle.load(f)
        
        r_sev = ReHAN(len(v_sev), 200, 128, 4).to(device)
        r_sev.load_state_dict(torch.load("rehan_severity.pt", map_location=device))
        r_sev.eval()
        
        r_cau = ReHAN(len(v_cau), 200, 128, 6).to(device)
        r_cau.load_state_dict(torch.load("rehan_cause.pt", map_location=device))
        r_cau.eval()
        
        # FIX: Explicit directory path
        rob_sev = RobertaForSequenceClassification.from_pretrained(current_dir, local_files_only=True).to(device)
        rob_sev.eval()
        
        rob_cau = RobertaForSequenceClassification.from_pretrained(current_dir, local_files_only=True).to(device)
        rob_cau.eval()
        
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        return r_sev, r_cau, rob_sev, rob_cau, tokenizer, v_sev, v_cau, None
    except Exception as e:
        return None, None, None, None, None, None, None, str(e)

r_sev, r_cau, rob_sev, rob_cau, tokenizer, v_sev, v_cau, error_msg = load_resources()

# ==========================================
#         MAIN VIEW LOGIC
# ==========================================

if not st.session_state.show_demo:
    # --- LANDING PAGE ---
    st.markdown("<h1 style='text-align: center; color: #8d7dca; font-size: 3.5rem; margin-bottom: 0;'>Insyte</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.5rem; color: #a0a0a0;'>Early Linguistic Examiner</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    count_up_html = """
    <div style="text-align: center; font-family: 'Playfair Display', serif; padding: 20px;">
        <div id="counter" style="font-size: 5rem; font-weight: 700; color: #8d7dca; margin-bottom: 5px; line-height: 1;">0</div>
        <div style="font-size: 1.3rem; color: #d0d0d0; font-weight: 300; font-family: 'Inter', sans-serif;">people die by suicide annually in Canada.</div>
    </div>
    <script>
        let startTimestamp = null;
        const duration = 2500;
        const finalValue = 4500;
        const element = document.getElementById('counter');
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const easeOutQuart = 1 - Math.pow(1 - progress, 4);
            element.innerHTML = "> " + Math.floor(easeOutQuart * finalValue).toLocaleString();
            if (progress < 1) { window.requestAnimationFrame(step); }
        };
        window.requestAnimationFrame(step);
    </script>
    """
    components.html(count_up_html, height=200)

    st.markdown("""
    Clinicians face increasing caseloads, time constraints, and documentation burdens, making early signal detection more difficult. 
    Insyte is building AI-assisted tools to support structured, time-efficient clinical insight. Analyze patient-written responses to identify symptom severity and potential contributing factors in seconds.
    """)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### About Insyte")
        st.markdown("Insyte is an early-stage clinical support tool designed to assist psychologists in reviewing written intake and assessment materials more efficiently. This prototype is currently in pilot testing and is not clinically validated.")
    with c2:
        st.markdown("### How it helps")
        st.markdown("* Highlights symptom-related language patterns\n* Generates structured summaries\n* Supports, not replaces, clinical decision-making")
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # SALIENT SURVEY AND BUTTON
    st.markdown(f"<div style='text-align: center; margin-bottom: 20px;'><h3 style='font-weight: 400;'>Help shape the future of this tool.</h3><a href='{survey_url}' style='color: #8d7dca; font-size: 1.2rem; text-decoration: underline;'>Take the Clinical Feedback Survey</a></div>", unsafe_allow_html=True)
    
    _, col_btn, _ = st.columns([1, 1.5, 1])
    with col_btn:
        if st.button("Demo the tool →", use_container_width=True):
            st.session_state.show_demo = True
            st.rerun()

else:
    # --- TOOL / DEMO VIEW ---
    st.markdown("<h2 style='color: #8d7dca; margin-bottom: 0;'>Early Linguistic Examiner</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #a0a0a0;'>Clinical Pilot &mdash; Diagnostic Support</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    if error_msg:
        st.error(f"Critical Error Loading Models: {error_msg}")
        st.info("Check the GitHub logs. Ensure 'config.json' and 'model.safetensors' are in the root directory.")
        st.stop()

    user_input = st.text_area("Patient Discourse Input:", height=200)
    analysis_type = st.radio("Analysis Mode:", ["Severity", "Causality"], horizontal=True)

    if st.button("Generate Insights"):
        if user_input.strip():
            raw_sentences = [s.strip() for s in re.split(r'[.!?]', user_input) if s.strip()][:15]
            inputs = tokenizer([user_input], return_tensors="pt", truncation=True, padding=True).to(device)

            with torch.no_grad():
                sev_out = rob_sev(**inputs)
                sev_probs = F.softmax(sev_out.logits, dim=1)[0]
                p_min, p_mild, p_mod, p_sev_val = [p.item() for p in sev_probs]

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
                raw_hybrid = (p_sev_val * 0.6) + (rehan_signal * 0.4)
            
                if p_min > 0.85 and len(user_input.split()) < 12:
                    hybrid_score, final_label, status_color = 0.05, "No Depression / Healthy Range", "success"
                elif p_sev_val > 0.75:
                    hybrid_score, final_label, status_color = max(p_sev_val, raw_hybrid), "Severe", "error"
                else:
                    hybrid_score = raw_hybrid
                    if hybrid_score < 0.25: final_label, status_color = "Minimum", "success"
                    elif hybrid_score < 0.45: final_label, status_color = "Mild", "info"
                    elif hybrid_score < 0.70: final_label, status_color = "Moderate", "warning"
                    else: final_label, status_color = "Severe", "error"

                st.markdown(f"### Assessment: **{final_label}**")

                cols = st.columns(3)
                cols[0].metric("RoBERTa Confidence", f"{p_sev_val:.2f}")
                cols[1].metric("ReHAN Attribution", f"{rehan_signal:.2f}")
                cols[2].metric("Final Severity Index", f"{hybrid_score:.2f}")

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
                    if isinstance(importance, float): importance = [importance]

                causes = ["No Reason", "Bias", "Job/Career", "Medication", "Relationship", "Alienation"]
                rob_max, rob_idx = torch.max(cau_probs, dim=0)
                rehan_cau_sig = min(sum(w for w in importance if w > 0.18), 1.0)
                hybrid_cau = (rob_max.item() * 0.5) + (rehan_cau_sig * 0.5)
                result_cause = causes[rob_idx.item()] if hybrid_cau >= 0.40 else "Inconclusive / Other"

                st.markdown(f"### Thematic Determinant: **{result_cause}**")
                st.progress(hybrid_cau)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("### Linguistic Evidence Analysis")
            for score, sent in zip(importance, raw_sentences):
                if score > 0.15:
                    st.markdown(f"> **[Signal Intensity: {score:.2f}]** {sent}")
        else:
            st.warning("Input required.")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### Important Notice")
    st.markdown("""
    <div style='background-color: rgba(141, 125, 202, 0.1); border-left: 3px solid #8d7dca; padding: 15px; color: #d0d0d0; font-size: 1rem;'>
        This is an early pilot version designed for clinical feedback and validation.
        <ul style='margin-top: 10px;'>
            <li><strong>Privacy:</strong> Do not enter identifying patient information.</li>
            <li><strong>Validation:</strong> Not clinically validated.</li>
            <li><strong>Usage:</strong> For feedback only, not for diagnostic decisions.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)