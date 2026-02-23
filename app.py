import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import re
import pickle
import os


# ==========================================
# 1. CUSTOM AESTHETICS (Midnight Plum & Silver Ash)
# ==========================================
st.set_page_config(page_title="Insyte | Clarity in every word", layout="wide")

st.markdown(f"""
    <style>
    /* Main background */
    .stApp {{
        background-color: #2D1B33; /* Midnight Plum */
        color: #B2B2B2; /* Silver Ash text */
    }}
    
    /* Input box styling */
    .stTextArea textarea {{
        background-color: #3D2B43 !important;
        color: #E0E0E0 !important;
        border: 1px solid #7D6B83 !important;
    }}

    /* Sidebar and Headers */
    h1, h2, h3 {{
        color: #D8D8D8 !important; /* Brighter Silver for headers */
    }}

    /* Radio and Selectbox text */
    .stRadio label, .stSelectbox label {{
        color: #B2B2B2 !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. BRANDING & LOGO SECTION
# ==========================================
# Create two columns: Left for text, Right for the logo
header_left, header_right = st.columns([4, 1])

with header_left:
    st.title("Insyte")
    st.markdown("*\"Clarity in every word\"*")

with header_right:
    # Ensure 'Insytelogo.png' is in the same folder as your script
    try:
        st.image("Insytelogo.png", width=200)
    except:
        st.caption("[Logo Missing: Insytelogo.png]")

st.markdown("---")

# ==========================================
# 3. WELCOME MESSAGE
# ==========================================

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    pwd = st.text_input("Enter Pilot Access Code:", type="password")
    if pwd == "0117": # Change this to your desired password
        st.session_state["authenticated"] = True
        st.rerun()
    else:
        st.stop()
        
st.write("### Welcome to Insyte, a new standard for precision in mental health support.")
st.write("""
You are invited to test our early pilot feature: a **Mental Health Text Analyzer** that reviews 
written responses and provides an estimate of depression severity and potential contributing factors.
""")

with st.expander("ℹ️ Pilot Information & Survey (Click to expand)", expanded=True):
    st.warning("""
    **This is a very early prototype built for feedback purposes only.** After trying the demo, please complete the short survey below to let us know what 
    could be improved and whether you would consider using a refined version in your clinic.
    """)
st.info("""
**We value your feedback!** Please let us know your thoughts:
👉 **[Complete the Pilot Survey Here](https://forms.gle/Undb6yxKKC53b3Zd6)**
    """)

# --- Sidebar for Roadmap ---
with st.sidebar:
    st.header("Coming Soon")
    st.write("""
    * Patient assessment tracking
    * Treatment progress monitoring
    * Structured check-ins
    * Workflow and documentation support
    """)
    st.info("Join the pilot to help shape these features.")

# ==========================================
# CORE ENGINE SETUP
# ==========================================
# Ensuring consistent device usage across all models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        # Dim=1 ensures softmax happens across the sequence length
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

# ==========================================
# 2. RESOURCE LOADING (With Error Catching)
# ==========================================
@st.cache_resource
def load_resources():
    try:
        # Load Vocabularies
        with open("rehan_severity_vocab.pkl", "rb") as f: v_sev = pickle.load(f)
        with open("rehan_cause_vocab.pkl", "rb") as f: v_cau = pickle.load(f)
        
        # Initialize and Load ReHAN Severity
        r_sev = ReHAN(len(v_sev), 200, 128, 4).to(device)
        r_sev.load_state_dict(torch.load("rehan_severity.pt", map_location=device))
        r_sev.eval()
        
        # Initialize and Load ReHAN Causality
        r_cau = ReHAN(len(v_cau), 200, 128, 6).to(device)
        r_cau.load_state_dict(torch.load("rehan_cause.pt", map_location=device))
        r_cau.eval()
        
        # Load RoBERTa Models (Ensure these paths exist!)
        rob_sev = RobertaForSequenceClassification.from_pretrained("./results/checkpoint-1068").to(device)
        rob_sev.eval()
        
        rob_cau = RobertaForSequenceClassification.from_pretrained("./cause_model_results/checkpoint-1512").to(device)
        rob_cau.eval()
        
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        return r_sev, r_cau, rob_sev, rob_cau, tokenizer, v_sev, v_cau, None
    except Exception as e:
        return None, None, None, None, None, None, None, str(e)

# Attempt to load
r_sev, r_cau, rob_sev, rob_cau, tokenizer, v_sev, v_cau, error_msg = load_resources()

# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Insyte", layout="wide")

if error_msg:
    st.error(f"Critical Error Loading Models: {error_msg}")
    st.info("Ensure all .pt, .pkl, and checkpoint folders are in: " + os.getcwd())
    st.stop()

st.title("🧠 Mental Health Analysis Conduit")
st.caption("Diagnostic Support Tool")

user_input = st.text_area("Patient Discourse Input:", height=150)
analysis_type = st.radio("Analysis Mode:", ["Severity", "Causality"], horizontal=True)

if st.button("Generate Insights"):
    if user_input.strip():
        # 1. PRE-PROCESSING
        raw_sentences = [s.strip() for s in re.split(r'[.!?]', user_input) if s.strip()][:15]
        inputs = tokenizer([user_input], return_tensors="pt", truncation=True, padding=True).to(device)

        with torch.no_grad():
            # RoBERTa Severity Pass
            sev_out = rob_sev(**inputs)
            sev_probs = F.softmax(sev_out.logits, dim=1)[0]
            p_min, p_mild, p_mod, p_sev_val = [p.item() for p in sev_probs]

        if analysis_type == "Severity":
            # 2. ReHAN SEVERITY PASS
            with torch.no_grad():
                tensor = torch.zeros(15, 20, dtype=torch.long).to(device)
                for i, sent in enumerate(raw_sentences):
                    words = sent.lower().split()[:20]
                    for j, word in enumerate(words):
                        tensor[i, j] = v_sev.get(word, v_sev.get("<UNK>", 1))
                
                _, weights_re = r_sev(tensor.unsqueeze(0))
                importance = weights_re.squeeze().cpu().tolist()
                if isinstance(importance, float): importance = [importance]

            # Calculate ReHAN signal strength
            rehan_signal = min(sum(w for w in importance if w > 0.15), 1.0)
            
            # --- THE FIX: HYBRID LOGIC ADJUSTMENT ---
            
            # Use 60/40 weighted average as a baseline
            raw_hybrid = (p_sev_val * 0.6) + (rehan_signal * 0.4)
            
            if p_min > 0.85 and len(user_input.split()) < 12:
                # OVERRIDE 1: If RoBERTa is 85%+ sure of 'Minimum' and it's short, force "No Depression"
                hybrid_score = 0.05
                final_label = "No Depression / Healthy Range"
                status_color = "success"
            elif p_sev_val > 0.75:
                # OVERRIDE 2: If RoBERTa is 75%+ sure of 'Severe', don't let ReHAN dilute it.
                # This ensures severe cases aren't labeled as 'Mild'
                hybrid_score = max(p_sev_val, raw_hybrid)
                final_label = "Severe"
                status_color = "error"
            else:
                # STANDARD HYBRID: For everything else, use the 60/40 blend
                hybrid_score = raw_hybrid
                
                # Dynamic Label Mapping based on hybrid score
                if hybrid_score < 0.25:
                    final_label = "Minimum"
                    status_color = "success"
                elif hybrid_score < 0.45:
                    final_label = "Mild"
                    status_color = "info"
                elif hybrid_score < 0.70:
                    final_label = "Moderate"
                    status_color = "warning"
                else:
                    final_label = "Severe"
                    status_color = "error"

            # 3. AESTHETIC RESULTS DISPLAY
            # Displaying the Result with color-coded status boxes
            if status_color == "success": st.success(f"### Assessment: {final_label}")
            elif status_color == "info": st.info(f"### Assessment: {final_label}")
            elif status_color == "warning": st.warning(f"### Assessment: {final_label}")
            else: st.error(f"### Assessment: {final_label}")

            cols = st.columns(3)
            cols[0].metric("RoBERTa Confidence", f"{p_sev_val:.2f}")
            cols[1].metric("ReHAN Attribution", f"{rehan_signal:.2f}")
            cols[2].metric("Final Severity Index", f"{hybrid_score:.2f}")

        else:
            # --- CAUSALITY LOGIC ---
            # (Keeping your original logic but ensuring it's wrapped in the button block)
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

            st.subheader(f"Thematic Determinant: {result_cause}")
            st.progress(hybrid_cau)
            st.write(f"Confidence Level: {hybrid_cau*100:.1f}%")

        # 4. EXPLAINABILITY SECTION (Discourse Determinants)
        st.markdown("---")
        st.write("### Linguistic Evidence Analysis")
        st.caption("The Re-HAN component identifies specific sentences that drove this assessment:")
        for score, sent in zip(importance, raw_sentences):
            if score > 0.15:
                # Using a visually distinct way to show the sentences
                st.markdown(f"> **[Signal Intensity: {score:.2f}]** {sent}")

    else:
        st.warning("Input required. Please enter patient discourse to begin analysis.")
st.markdown("---")
st.subheader("⚠️ Important Notice")
st.error("""
* **Privacy:** Do not enter any identifying patient information.
* **Validation:** This tool is not clinically validated and may produce inaccurate results.
* **Usage:** It is intended for testing and feedback only, not for diagnostic or treatment decisions.
""")