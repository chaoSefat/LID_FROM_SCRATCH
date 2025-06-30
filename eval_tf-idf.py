import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    top_k_accuracy_score
)

from tqdm import tqdm

# Constants
DATA_DIR = "data_curation/glotlid_top20/outputs/"
MODEL_DIR = "models/"
SAVE_FIGS_DIR = "figures/"

language_codes = {
    "Mandarin Chinese": "cmn_Hani", "Spanish": "spa_Latn", "English": "eng_Latn",
    "Hindi": "hin_Deva", "Arabic": "arb_Arab", "Bengali": "ben_Beng",
    "Portuguese": "por_Latn", "Russian": "rus_Cyrl", "Japanese": "jpn_Jpan",
    "Western Punjabi": "pnb_Arab", "Marathi": "mar_Deva", "Telugu": "tel_Telu",
    "Wu Chinese": "wuu_Hani", "Turkish": "tur_Latn", "Korean": "kor_Hang",
    "French": "fra_Latn", "German": "deu_Latn", "Vietnamese": "vie_Latn",
    "Tamil": "tam_Taml", "Urdu": "urd_Arab"
}

# Reverse mapping for label to language name
label_to_lang = {v: k for k, v in language_codes.items()}
label_list = list(language_codes.values())
lang_names = [label_to_lang[label] for label in label_list]


def load_val_data():
    texts, labels = [], []
    print("Loading validation data...")
    for label in tqdm(label_list):
        file_path = os.path.join(DATA_DIR, f"{label}_val.csv")
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            continue
        df = pd.read_csv(file_path)
        texts.extend(df['text'].astype(str).tolist())
        labels.extend([label] * len(df))
    return texts, labels


def plot_confusion_matrix(y_true, y_pred, labels, label_names, filename):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                cbar=True)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.xlabel("Predicted Language")
    plt.ylabel("True Language")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_f1_scores(y_true, y_pred, labels, label_names, filename):
    _, _, f1_scores, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    # Sort by F1 descending
    sorted_indices = np.argsort(f1_scores)[::-1]
    sorted_labels = [label_names[i] for i in sorted_indices]
    sorted_f1 = f1_scores[sorted_indices]

    plt.figure(figsize=(14, 6))
    sns.barplot(x=sorted_labels, y=sorted_f1, palette="viridis")
    plt.xticks(rotation=90)
    plt.ylim(0, 1)
    plt.ylabel("F1 Score")
    plt.title("Per-Class F1 Scores (sorted)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_top_k_accuracy(y_true, y_proba, labels, label_names, k=3, filename=None):
    """
    Compute top-k accuracy for each class and plot bar chart.
    """
    n_classes = len(labels)
    # For each class, compute top-k accuracy on samples of that class
    topk_accs = []
    for class_idx in range(n_classes):
        # mask for samples belonging to this class
        mask = (y_true == labels[class_idx])
        if np.sum(mask) == 0:
            topk_accs.append(0.0)
            continue
        topk_acc = top_k_accuracy_score(
            y_true[mask], y_proba[mask], k=k, labels=labels
        )
        topk_accs.append(topk_acc)

    # Sort by top-k accuracy descending
    sorted_indices = np.argsort(topk_accs)[::-1]
    sorted_labels = [label_names[i] for i in sorted_indices]
    sorted_accs = np.array(topk_accs)[sorted_indices]

    plt.figure(figsize=(14, 6))
    sns.barplot(x=sorted_labels, y=sorted_accs, palette="magma")
    plt.xticks(rotation=90)
    plt.ylim(0, 1)
    plt.ylabel(f"Top-{k} Accuracy")
    plt.title(f"Per-Class Top-{k} Accuracy (sorted)")
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def evaluate():
    print("Loading models and data...")
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    clf = joblib.load(os.path.join(MODEL_DIR, "logreg_model.pkl"))
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

    X_val_texts, y_val_labels = load_val_data()
    y_val = le.transform(y_val_labels)

    print("Vectorizing validation texts...")
    X_val_vec = vectorizer.transform(X_val_texts)

    print("Predicting...")
    y_pred = clf.predict(X_val_vec)
    y_proba = clf.predict_proba(X_val_vec)

    print("Evaluation Metrics:")
    acc = accuracy_score(y_val, y_pred)
    print(f"Accuracy: {acc:.4f}\n")

    print("Classification Report (macro & weighted):")
    print(classification_report(
        y_val, y_pred, target_names=le.classes_,
        zero_division=0, digits=4
    ))

    os.makedirs(SAVE_FIGS_DIR, exist_ok=True)

    print("Plotting confusion matrix...")
    plot_confusion_matrix(
        y_val, y_pred,
        labels=list(range(len(le.classes_))),
        label_names=lang_names,
        filename=os.path.join(SAVE_FIGS_DIR, "confusion_matrix_annotated.png")
    )

    print("Plotting F1 scores...")
    plot_f1_scores(
        y_val, y_pred,
        labels=list(range(len(le.classes_))),
        label_names=lang_names,
        filename=os.path.join(SAVE_FIGS_DIR, "f1_scores_sorted.png")
    )

    print("Plotting Top-3 accuracy...")
    plot_top_k_accuracy(
        y_val, y_proba,
        labels=list(range(len(le.classes_))),
        label_names=lang_names,
        k=3,
        filename=os.path.join(SAVE_FIGS_DIR, "top3_accuracy_sorted.png")
    )

    print(f"All plots saved in `{SAVE_FIGS_DIR}`.")
    print("Evaluation complete.")


if __name__ == "__main__":
    evaluate()
