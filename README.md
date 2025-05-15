# Wordsmith - SMS Analysis Suite

This NLP project tackles the important problem of SMS spam detection using machine learning techniques. The solution consists of a complete end-to-end pipeline including data preprocessing, feature extraction, model training, evaluation, and a user-friendly Streamlit interface.

## Features

- Data loading and preprocessing
- Text analysis with TF-IDF
- Machine learning model training (Naive Bayes, Logistic Regression, SVM, Random Forest)
- Model evaluation and hyperparameter tuning
- Custom message classification

## Developed by
1. Mohit Prjapati
2. Sujit Kumar Shah
3. Md. Ali Alkama  
4. Modassir Alam
5. Zeeshan Ahmad

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

## Dataset
Upload a CSV file with at least two columns:
- 'v1': Class labels (spam/ham)
- 'v2': SMS messages

## About
This project demonstrates NLP techniques for text classification, featuring comprehensive preprocessing steps, model training and evaluation, and a user-friendly interface.
