# src/predict_binary.py

import joblib
from sentimentAnalysisOfCyberbullying.src.preprocess import clean_text

MODEL_PATH = "/Users/shivanshusingh/Downloads/study material/sentiment analysis/models/binary_model.pkl"
VEC_PATH = "/Users/shivanshusingh/Downloads/study material/sentiment analysis/models/binary_tfidf.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

def predict_text_binary(text: str) -> str:
    cleaned = clean_text(text)
    X_vec = vectorizer.transform([cleaned])
    pred = model.predict(X_vec)[0]
    return pred