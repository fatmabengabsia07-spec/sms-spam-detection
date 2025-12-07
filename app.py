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
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

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
        background: linear-gradient(135deg,#eef2ff,#e0e7ff,#fdf2f8) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* HEADER */
    .custom-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff;
        padding: 25px 40px;
        margin-bottom: 25px;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        animation: fadeInDown 1s ease-out;
    }

    .custom-header-title {
        font-size: 3.4rem;
        font-weight: 900;
        background: linear-gradient(135deg,#6366f1,#8b5cf6,#d946ef);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -2px;
    }

    .custom-logo {
    width: 180px;
    border-radius: 0px !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 0 !important;
}

    /* TEXTAREA */
    .stTextArea > div > div > textarea {
        border-radius: 16px !important;
        background: rgba(255,255,255,0.75) !important;
        border: 2px solid #a78bfa !important;
        font-size: 1.15rem !important;
        padding: 1.4rem !important;
        box-shadow: 0 8px 22px rgba(99,102,241,0.20);
    }

    /* BOUTON */
    .stButton > button {
        background: linear-gradient(135deg,#6366f1,#7c3aed) !important;
        border-radius: 14px !important;
        padding: 1.1rem !important;
        font-size: 1.25rem !important;
        font-weight: 700;
        box-shadow: 0 10px 25px rgba(99,102,241,0.25);
        transition: 0.3s;
        width: 100%;
        color: white !important;
    }

    .stButton > button:hover {
        transform: translateY(-4px);
        box-shadow: 0 15px 35px rgba(99,102,241,0.35);
    }

    /* RESULTATS */
    .result-spam, .result-ham {
        border-radius: 20px;
        padding: 2.8rem;
        font-size: 2rem;
        font-weight: 800;
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
        animation: bounceIn 0.8s ease-out;
        text-align: center;
        color: white;
        margin-top: 30px;
    }

    .result-spam {
        background: linear-gradient(135deg,#ef4444,#dc2626,#b91c1c);
    }

    .result-ham {
        background: linear-gradient(135deg,#10b981,#059669,#047857);
    }

    /* FOOTER */
    .footer {
        margin-top: 3rem;
        padding: 1.5rem;
        text-align: center;
        opacity: 0.8;
        font-size: 0.9rem;
        color: #475569;
    }

    @keyframes fadeInDown { from {opacity:0; transform:translateY(-30px);} to {opacity:1; transform:translateY(0);} }
    @keyframes bounceIn { 0% {transform:scale(0.8); opacity:0;} 50% {opacity:1;} 100% {transform:scale(1);} }

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
    "<p style='text-align:center; font-size:1.3rem; color:#475569;'>Analyse intelligente des messages SPAM</p>",
    unsafe_allow_html=True
)

# Afficher le logo spam en haut à gauche aggrandi
try:
    logo = Image.open("logo_spam.jpg")
    col_left, col_space = st.columns([1.8, 4.7])
    with col_left:
        st.image(logo, use_container_width=True)
except:
    pass

col1, col2, col3 = st.columns([0.25, 3.5, 0.25])
with col2:
    input_sms = st.text_area(
        "",
        placeholder="Entrez votre message à analyser...",
        height=220
    )

    analyze_btn = st.button(" Analyser le message", type="primary")

if analyze_btn and input_sms.strip():
    with st.spinner("Analyse en cours..."):
        sms = transform_text(input_sms)
        vector = tfidf.transform([sms])
        result = model.predict(vector)[0]

        if result == 1:
            st.markdown(
                '<div class="result-spam"> SPAM DÉTECTÉ </div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-ham"> MESSAGE SÛR </div>',
                unsafe_allow_html=True
            )


st.markdown(
    '<div class="footer"> Vos données restent privées |  Analyse instantanée </div>',
    unsafe_allow_html=True
)
