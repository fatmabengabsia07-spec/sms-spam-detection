import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from PIL import Image
import base64
import string
import time

# Télécharger les données NLTK nécessaires
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

download_nltk_data()


st.set_page_config(
    page_title="SPAM MF",
    layout="centered"
)


@st.cache_resource
def load_model():
    try:
        tfidf = pickle.load(open("vectorizer.pkl", "rb"))
        model = pickle.load(open("model.pkl", "rb"))
        return tfidf, model
    except:
        st.error("Impossible de charger model.pkl ou vectorizer.pkl.")
        st.stop()

tfidf, model = load_model()
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalnum()]
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    tokens = [ps.stem(t) for t in tokens]
    return " ".join(tokens)


st.markdown("""
<style>

    html, body, .stApp {
        background: linear-gradient(135deg,#f0f4ff,#e8f0ff,#f5e6ff) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }

    /* HEADER */
    .custom-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg,#ffffff,#f9f7ff);
        padding: 20px 35px;
        margin-bottom: 30px;
        border-radius: 18px;
        box-shadow: 0 10px 40px rgba(99,102,241,0.15);
        animation: fadeInDown 1s ease-out;
        border-left: 5px solid #6366f1;
    }

    .custom-header-title {
        font-size: 2.8rem;
        font-weight: 900;
        background: linear-gradient(135deg,#6366f1,#8b5cf6,#d946ef);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        margin: 0;
    }

    .custom-logo {
        width: 130px;
        border-radius: 12px !important;
        box-shadow: 0 5px 20px rgba(99,102,241,0.2) !important;
        background: transparent !important;
        padding: 0 !important;
        transition: transform 0.3s ease;
    }

    .custom-logo:hover {
        transform: scale(1.05);
    }

    /* SUBTITLE */
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #475569;
        margin-bottom: 30px;
        font-weight: 500;
        letter-spacing: 0.3px;
    }

    /* TEXTAREA */
    .stTextArea > div > div > textarea {
        border-radius: 16px !important;
        background: rgba(255,255,255,0.9) !important;
        border: 2px solid #a78bfa !important;
        font-size: 1.05rem !important;
        padding: 1.5rem !important;
        box-shadow: 0 8px 25px rgba(99,102,241,0.15) !important;
        font-family: 'Segoe UI' !important;
        transition: all 0.3s ease !important;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 12px 35px rgba(99,102,241,0.25) !important;
    }

    /* INPUT LABEL */
    .stTextArea > label {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #1e293b !important;
        margin-bottom: 12px !important;
    }

    /* BOUTON */
    .stButton > button {
        background: linear-gradient(135deg,#6366f1,#7c3aed,#a855f7) !important;
        border-radius: 12px !important;
        padding: 1.3rem 3rem !important;
        font-size: 1.1rem !important;
        font-weight: 700;
        box-shadow: 0 12px 30px rgba(99,102,241,0.35);
        transition: all 0.3s ease !important;
        color: white !important;
        border: none !important;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 18px 45px rgba(99,102,241,0.45) !important;
    }

    .stButton > button:active {
        transform: translateY(-1px) !important;
    }

    /* RESULTATS */
    .result-spam, .result-ham {
        border-radius: 18px;
        padding: 3rem;
        font-size: 1.8rem;
        font-weight: 800;
        box-shadow: 0 18px 50px rgba(0,0,0,0.18);
        animation: bounceIn 0.8s ease-out;
        text-align: center;
        color: white;
        margin-top: 40px;
        margin-bottom: 20px;
        letter-spacing: 1px;
        border: 3px solid rgba(255,255,255,0.3);
    }

    .result-spam {
        background: linear-gradient(135deg,#ef4444,#dc2626,#b91c1c);
    }

    .result-ham {
        background: linear-gradient(135deg,#10b981,#059669,#047857);
    }

    /* FOOTER */
    .footer {
        margin-top: 4rem;
        padding: 2rem;
        text-align: center;
        opacity: 0.85;
        font-size: 0.95rem;
        color: #475569;
        border-top: 1px solid rgba(99,102,241,0.1);
        background: rgba(255,255,255,0.4);
        border-radius: 12px;
        font-weight: 500;
    }

    /* ANIMATIONS */
    @keyframes fadeInDown { from {opacity:0; transform:translateY(-30px);} to {opacity:1; transform:translateY(0);} }
    @keyframes bounceIn { 0% {transform:scale(0.8); opacity:0;} 50% {opacity:1;} 100% {transform:scale(1);} }

    /* CONTAINER */
    .main-container {
        background: rgba(255,255,255,0.5);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(99,102,241,0.1);
        backdrop-filter: blur(10px);
    }

</style>
""", unsafe_allow_html=True)


try:
    logo = Image.open("spam1.png")

    # Conversion Base64
    import io
    buffer = io.BytesIO()
    logo.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()

    st.markdown(
        f"""
        <div class="custom-header">
            <div class="custom-header-title">📛 SPAM MF</div>
            <img src="data:image/png;base64,{img_b64}" class="custom-logo">
        </div>
        """,
        unsafe_allow_html=True
    )
except:
    st.markdown(
        """
        <div class="custom-header">
        </div>
        """,
        unsafe_allow_html=True
    )



st.markdown(
    "<p class='subtitle'>🔍 Détectez instantanément les messages SPAM avec notre IA avancée</p>",
    unsafe_allow_html=True
)

# Afficher le logo spam en haut à gauche aggrandi
try:
    logo = Image.open("logo_spam.jpg")
    col_left, col_space = st.columns([1.5, 5])
    with col_left:
        st.image(logo, use_container_width=True)
except:
    pass

# Zone de saisie avec meilleure présentation
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([0.25, 3.5, 0.25])
with col2:
    st.markdown("<h3 style='text-align:center; color:#1e293b;'>Analysez votre message</h3>", unsafe_allow_html=True)
    input_sms = st.text_area(
        "Entrez le texte à analyser",
        placeholder="Collez votre SMS ou email ici...",
        height=180,
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([0.8, 1.5, 0.8])
    with col_btn2:
        analyze_btn = st.button("🚀 Analyser", type="primary", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

if analyze_btn and input_sms.strip():
    with st.spinner("⏳ Analyse en cours..."):
        time.sleep(0.5)  # Effet de chargement
        sms = transform_text(input_sms)
        vector = tfidf.transform([sms])
        result = model.predict(vector)[0]

        if result == 1:
            st.markdown(
                '<div class="result-spam">🚨 SPAM DÉTECTÉ 🚨</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                "<p style='text-align:center; color:#b91c1c; font-weight:600;'>⚠️ Ce message a été identifié comme SPAM</p>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-ham">✅ MESSAGE LÉGITIME ✅</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                "<p style='text-align:center; color:#047857; font-weight:600;'>✓ Ce message semble sûr et fiable</p>",
                unsafe_allow_html=True
            )


st.markdown(
    '<div class="footer">🔒 Vos données restent privées | ⚡ Analyse instantanée | 🤖 Technologie IA avancée</div>',
    unsafe_allow_html=True
)
