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
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500&display=swap');

    /* Global Typography + Animated Deep Plum/Grey/Black Gradient */
    .stApp {
        background: linear-gradient(-45deg, #0a0a0a, #141018, #1B1622, #000000);
        background-size: 400% 400%;
        animation: gradientMove 15s ease infinite;
        color: #f5f5f5;
        font-family: 'Playfair Display', serif;
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Aesthetic "I" enlargement for Insyte branding */
    .insyte-brand { font-family: 'Playfair Display', serif; font-weight: 700; }
    .insyte-brand::first-letter { font-size: 1.3em; color: #9C8ACF; }

    /* Massive Headings */
    h1 { font-size: 4.5rem !important; font-weight: 700 !important; margin-bottom: 0px !important; }
    h2 { font-size: 3rem !important; font-weight: 400 !important; }
    h3 { font-size: 2.2rem !important; font-weight: 400 !important; color: #9C8ACF !important; }

    /* Sidebar - Smaller Text */
    section[data-testid="stSidebar"] .stMarkdown p {
        font-size: 1rem !important;
        line-height: 1.5 !important;
    }
    section[data-testid="stSidebar"] h1 { font-size: 2rem !important; }

    /* Large, Readable Body Text */
    p, li, label {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem !important;
        line-height: 1.7 !important;
        color: #D6D3DD;
    }

    /* Survey Box Styling */
    .survey-box {
        background-color: rgba(156, 138, 207, 0.08);
        border: 1px solid rgba(156, 138, 207, 0.3);
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-top: 20px;
    }
    .survey-link { color: #9C8ACF !important; font-size: 1.5rem !important; font-weight: 700; }

    /* Premium Button */
    .stButton>button {
        background-color: #9C8ACF !important;
        color: #141018 !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        padding: 12px 30px !important;
        border-radius: 8px !important;
        border: none !important;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 10px 25px rgba(156, 138, 207, 0.2); }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- 3. SESSION STATE ---
if 'show_demo' not in st.session_state:
    st.session_state.show_demo = False

# --- 4. SIDEBAR (Smaller Text) ---
with st.sidebar:
    st.markdown("<h1 class='insyte-brand'>Insyte</h1>", unsafe_allow_html=True)
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
    st.caption("Pilot v1.2 | Clinical Validation Phase")

# --- 5. CORE ENGINE SETUP ---
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
    st.markdown("<h1 style='text-align: center; color: #8d7dca;' class='insyte-brand'>Insyte</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.8rem !important; color: #a0a0a0; margin-top: -20px;'>Early Linguistic Examiner</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    count_up_html = """
    <div style="text-align: center; padding: 20px;">
        <div id="counter" style="font-size: 6rem; font-weight: 700; color: #8d7dca; margin-bottom: 10px; line-height: 1; font-family: 'Playfair Display', serif;">0</div>
        <div style="font-size: 1.6rem; color: #d0d0d0; font-family: 'Inter', sans-serif;">people die by suicide annually in Canada (Statistics Canada).</div>
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
            element.innerHTML = "" + Math.floor(easeOutQuart * finalValue).toLocaleString();
            if (progress < 1) { window.requestAnimationFrame(step); }
        };
        window.requestAnimationFrame(step);
    </script>
    """
    components.html(count_up_html, height=250)

    st.markdown("Clinicians face increasing caseloads and documentation burdens, making early signal detection difficult. Insyte builds AI-assisted tools that support structured clinical insight for psychologists and mental health clinicians reviewing written intake and assessment materials.")
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # DEMO BUTTON (NOW ABOVE SURVEY BOX)
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        if st.button("Demo the tool →", use_container_width=True):
            st.session_state.show_demo = True
            st.rerun()

    # SURVEY BOX (BELOW BUTTON)
    st.markdown(f"""
    <div class="survey-box">
        <h2 style="margin-top: 0;">Help shape the future.</h2>
        <p>This is a clinical pilot. Your feedback is essential.</p>
        <a href="{survey_url}" class="survey-link" target="_blank">Feedback Survey →</a>
    </div>
    """, unsafe_allow_html=True)

else:
    # --- TOOL VIEW ---
    st.markdown("<h1 style='font-size: 3rem !important;' class='insyte-brand'>Early Linguistic Examiner</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #a0a0a0; margin-top: -20px;'>Clinical Pilot | Semantic Analysis Engine</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    if error_msg:
        st.error(f"Critical Error: {error_msg}")
        st.stop()

    user_input = st.text_area("Patient Discourse Input:", height=250, placeholder="Enter anonymized patient text here...")
    analysis_type = st.radio("Analysis Mode:", ["Severity", "Causality"], horizontal=True)

    if st.button("Generate Insights"):
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

                st.markdown(f"### Assessment: <span style='color:#8d7dca;'>{final_label}</span>", unsafe_allow_html=True)
                cols = st.columns(3)
                cols[0].metric("RoBERTa Confidence", f"{p_severe:.2f}")
                cols[1].metric("ReHAN Attribution", f"{rehan_signal:.2f}")
                cols[2].metric("Final Index", f"{raw_hybrid:.2f}")

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
                causes = ["No Reason", "Bias", "Job/Career", "Medication", "Relationship", "Alienation"]
                rob_max, rob_idx = torch.max(cau_probs, dim=0)
                rehan_cau_sig = min(sum(w for w in (importance if isinstance(importance, list) else [importance]) if w > 0.18), 1.0)
                hybrid_cau = (rob_max.item() * 0.5) + (rehan_cau_sig * 0.5)
                result_cause = causes[rob_idx.item()] if hybrid_cau >= 0.40 else "Inconclusive"
                st.markdown(f"### Primary Thematic Determinant: <span style='color:#8d7dca;'>{result_cause}</span>", unsafe_allow_html=True)
                st.progress(hybrid_cau)

            st.markdown("---")
            st.markdown("### Linguistic Evidence Analysis")
            for score, sent in zip((importance if isinstance(importance, list) else [importance]), raw_sentences):
                if score > 0.15:
                    st.markdown(f"<p style='font-size: 1.1rem !important; border-left: 3px solid #8d7dca; padding-left: 15px;'>{sent} <br><span style='color:#8d7dca; font-size: 0.9rem;'>Signal Intensity: {score:.2f}</span></p>", unsafe_allow_html=True)
        else:
            st.warning("Input required.")