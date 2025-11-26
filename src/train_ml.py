# src/train_ml.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import joblib

from sentimentAnalysisOfCyberbullying.src.preprocess import clean_text


DATA_PATH = "/Users/shivanshusingh/Downloads/study material/sentiment analysis/cyberbullying_tweets.csv"   # adjust path if needed


def load_data_binary():
    """
    Create a binary label:
    - 'bullying' for any type except 'not_cyberbullying'
    - 'not_cyberbullying' as is
    """
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["tweet_text", "cyberbullying_type"])

    def map_binary(lbl):
        if lbl == "not_cyberbullying":
            return "not_cyberbullying"
        else:
            return "bullying"

    df["label"] = df["cyberbullying_type"].apply(map_binary)
    df["clean_text"] = df["tweet_text"].apply(clean_text)

    X = df["clean_text"]
    y = df["label"]
    return X, y


def load_data_multiclass():
    """
    Use the original cyberbullying_type as multi-class label.
    e.g. age, religion, gender, ethnicity, other_cyberbullying, not_cyberbullying
    """
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["tweet_text", "cyberbullying_type"])

    df["label"] = df["cyberbullying_type"]
    df["clean_text"] = df["tweet_text"].apply(clean_text)

    X = df["clean_text"]
    y = df["label"]
    return X, y


def train_and_eval(X, y, model_name_prefix):
    """
    Generic training function:
    - splits into train/test
    - TF-IDF vectorizer
    - Logistic Regression
    - prints metrics
    - saves model & vectorizer
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2)
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Logistic Regression (you can switch to LinearSVC if you want)
    model = LogisticRegression(
        max_iter=500,
        n_jobs=-1,
        class_weight="balanced"  
    )
    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n===== {model_name_prefix.upper()} MODEL =====")
    print("Accuracy:", acc)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save
    joblib.dump(model, f"../models/{model_name_prefix}_model.pkl")
    joblib.dump(vectorizer, f"../models/{model_name_prefix}_tfidf.pkl")

    print(f"\nSaved model: ../models/{model_name_prefix}_model.pkl")
    print(f"Saved vectorizer: ../models/{model_name_prefix}_tfidf.pkl\n")

    return acc


def main():
    # 1) Binary classification: bullying vs not_cyberbullying
    X_bin, y_bin = load_data_binary()
    acc_bin = train_and_eval(X_bin, y_bin, model_name_prefix="binary")

    # 2) Multi-class classification: type of cyberbullying
    X_multi, y_multi = load_data_multiclass()
    acc_multi = train_and_eval(X_multi, y_multi, model_name_prefix="multiclass")

    print("Summary:")
    print(f"Binary model accuracy      : {acc_bin:.4f}")
    print(f"Multi-class model accuracy : {acc_multi:.4f}")


if __name__ == "__main__":
    main()