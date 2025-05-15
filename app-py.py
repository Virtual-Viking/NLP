# Rename new.py to app.py
# The content is the same as your original new.py file

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import csv
import nltk
import string

# Download NLTK resources
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set page title
st.set_page_config(page_title="Wordsmith - Text Analysis Suite", layout="wide")

# Add custom CSS after the page config to constrain the width
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1024px;
        margin-left: auto;
        margin-right: auto;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        text-align: center;
        font-weight: bold;
    }
    div.stButton > button.tf-idf-btn {
        background-color: #4CAF50; 
        color: white;
        padding: 10px 24px;
        margin: 10px 0px;
        font-weight: bold;
        border-radius: 4px;
        border: none;
    }
    div.stButton > button.tf-idf-btn:hover {
        background-color: #45a049;
    }
    div.stButton > button#tf-idf-button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        font-weight: bold;
        border-radius: 4px;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Wordsmith - Msg Analysis Suite")
st.write("This NLP project tackles the important problem of SMS spam detection using machine learning techniques. The solution consists of a complete end-to-end pipeline including data preprocessing, feature extraction, model training, evaluation, and a user-friendly Streamlit interface.")
st.write("Project developed by - .")
st.write("__________________1. Mohit Prjapati")
st.write("__________________2. Sujit Kumar Shah")
st.write("__________________3. Md. Ali Alkama")
st.write("__________________4. Modassir Alam")
st.write("__________________5. Zeeshan Ahmad")

# The rest of your Streamlit code follows here...
# I'm not including the full code for brevity, but your original file
# would be copied here with "new.py" renamed to "app.py"
