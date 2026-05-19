import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set page configuration for a premium look
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Custom CSS for modern design
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    h1 {
        color: #1e3d59;
        text-align: center;
        font-weight: 800;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #ff6e40;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        width: 100%;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff5252;
        box-shadow: 0 4px 12px rgba(255, 82, 82, 0.4);
    }
    .prediction-card-fake {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #f44336;
        color: #c62828;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    .prediction-card-real {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        color: #2e7d32;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_vectorizer():
    # Load the main model
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
    except Exception as e:
        model = None
        st.error(f"Error loading model.pkl: {e}")

    # Try loading a vectorizer if it was saved, since model needs numerical inputs
    try:
        with open('vectorizer.pkl', 'rb') as file:
            vectorizer = pickle.load(file)
    except FileNotFoundError:
        vectorizer = None
        
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

st.markdown("<h1>📰 AI Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555; margin-bottom: 40px;'>Enter the details of the news article below to verify its authenticity using Machine Learning.</p>", unsafe_allow_html=True)

# Input form
with st.container():
    author = st.text_input("✍️ Author Name", placeholder="e.g. John Doe")
    title = st.text_area("🗞️ News Title", placeholder="Enter the news headline here...", height=100)
    
    if st.button("Analyze News"):
        if not author.strip() or not title.strip():
            st.warning("Please provide both the author name and the news title.")
        else:
            with st.spinner("Analyzing text patterns and linguistic features..."):
                content = author + ' ' + title
                processed_content = stemming(content)
                
                if model is not None:
                    try:
                        # If a vectorizer exists, use it. Otherwise, assume model handles text (e.g. Pipeline)
                        if vectorizer:
                            vectorized_input = vectorizer.transform([processed_content])
                            prediction = model.predict(vectorized_input)
                        else:
                            prediction = model.predict([processed_content])
                        
                        # Interpret prediction based on notebook logic
                        if prediction[0] == 0:
                            st.markdown("<div class='prediction-card-real'>✅ Verified: This news appears to be REAL.</div>", unsafe_allow_html=True)
                            st.balloons()
                        else:
                            st.markdown("<div class='prediction-card-fake'>⚠️ Warning: This news is likely FAKE.</div>", unsafe_allow_html=True)
                    except ValueError as ve:
                        st.error(f"Prediction Error: The model expected a different input format. Make sure you have trained and saved the TfidfVectorizer as 'vectorizer.pkl'. Details: {ve}")
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")
                else:
                    st.error("Model is not loaded properly.")
