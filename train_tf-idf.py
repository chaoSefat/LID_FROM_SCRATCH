import os
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
import numpy as np

# Constants
DATA_DIR = "data_curation/glotlid_top20/outputs/"
SAVE_DIR = "models/"
NGRAM_RANGE = (1, 5)
MAX_FEATURES = 100000 
language_codes = {
    "Mandarin Chinese": "cmn_Hani", "Spanish": "spa_Latn", "English": "eng_Latn",
    "Hindi": "hin_Deva", "Arabic": "arb_Arab", "Bengali": "ben_Beng",
    "Portuguese": "por_Latn", "Russian": "rus_Cyrl", "Japanese": "jpn_Jpan",
    "Western Punjabi": "pnb_Arab", "Marathi": "mar_Deva", "Telugu": "tel_Telu",
    "Wu Chinese": "wuu_Hani", "Turkish": "tur_Latn", "Korean": "kor_Hang",
    "French": "fra_Latn", "German": "deu_Latn", "Vietnamese": "vie_Latn",
    "Tamil": "tam_Taml", "Urdu": "urd_Arab"
}
label_list = list(language_codes.values())

def load_data(split='train'):
    texts, labels = [], []
    print(f"Loading {split} data...")
    for label in tqdm(label_list):
        file_path = os.path.join(DATA_DIR, f"{label}_{split}.csv")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            continue
        df = pd.read_csv(file_path)
        texts.extend(df['text'].astype(str).tolist())
        labels.extend([label] * len(df))
    return texts, labels

def train_with_progress(X_train_vec, y_train_enc, epochs=5, batch_size=10000):
    clf = SGDClassifier(
        loss='log_loss', max_iter=1, warm_start=True, verbose=0,
        n_jobs=-1, random_state=42
    )
    n_samples = X_train_vec.shape[0]
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        X_train_vec, y_train_enc = shuffle(X_train_vec, y_train_enc, random_state=epoch)
        for i in tqdm(range(0, n_samples, batch_size), desc="Training progress"):
            end = i + batch_size
            clf.partial_fit(X_train_vec[i:end], y_train_enc[i:end], classes=np.unique(y_train_enc) if epoch == 0 and i == 0 else None)
    return clf

def main():
    print("Loading and preparing data...")
    X_train, y_train = load_data('train')
    print(f"Total training samples: {len(X_train)}")

    print("Encoding labels...")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    print("Shuffling data...")
    X_train, y_train_enc = shuffle(X_train, y_train_enc, random_state=42)

    print("Extracting TF-IDF char n-gram features...")
    vectorizer = TfidfVectorizer(
        analyzer='char', ngram_range=NGRAM_RANGE,
        lowercase=False, max_features=MAX_FEATURES
    )
    X_train_vec = vectorizer.fit_transform(tqdm(X_train, desc="TF-IDF fitting"))

    print("Training Logistic Regression model...")
    clf = train_with_progress(X_train_vec, y_train_enc)

    print("Saving model, vectorizer, and label encoder...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(SAVE_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(clf, os.path.join(SAVE_DIR, "logreg_model.pkl"))
    joblib.dump(le, os.path.join(SAVE_DIR, "label_encoder.pkl"))

    print("Training completed and artifacts saved.")

if __name__ == "__main__":
    main()
