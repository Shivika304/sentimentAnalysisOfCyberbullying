# src/predict.py
import joblib
from sentimentAnalysisOfCyberbullying.src.preprocess import clean_text

MODEL_PATH = "models/cyberbully_model.pkl"
VEC_PATH = "models/tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VEC_PATH)

def predict_text(text: str) -> str:
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return pred

if __name__ == "__main__":
    while True:
        txt = input("Enter a comment (or 'q' to quit): ")
        if txt.lower() == 'q':
            break
        print("Prediction:", predict_text(txt))