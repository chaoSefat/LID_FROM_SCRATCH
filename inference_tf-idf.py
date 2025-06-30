

import os
import joblib
import numpy as np
import argparse
from typing import List, Tuple

# Configuration
MODEL_DIR = "models/"
LANGUAGE_MAPPING = {
    "cmn_Hani": "Mandarin Chinese",
    "spa_Latn": "Spanish",
    "eng_Latn": "English",
    "hin_Deva": "Hindi",
    "arb_Arab": "Arabic",
    "ben_Beng": "Bengali",
    "por_Latn": "Portuguese",
    "rus_Cyrl": "Russian",
    "jpn_Jpan": "Japanese",
    "pnb_Arab": "Western Punjabi",
    "mar_Deva": "Marathi",
    "tel_Telu": "Telugu",
    "wuu_Hani": "Wu Chinese",
    "tur_Latn": "Turkish",
    "kor_Hang": "Korean",
    "fra_Latn": "French",
    "deu_Latn": "German",
    "vie_Latn": "Vietnamese",
    "tam_Taml": "Tamil",
    "urd_Arab": "Urdu"
}

def load_models() -> Tuple:
    try:
        vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
        model = joblib.load(os.path.join(MODEL_DIR, "logreg_model.pkl"))
        label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        return vectorizer, model, label_encoder
    except Exception as e:
        raise RuntimeError(f"Error loading models: {str(e)}")

def predict_language(text: str, vectorizer, model, label_encoder, top_k: int = 3) -> Tuple:
    X = vectorizer.transform([text])
    probabilities = model.predict_proba(X)[0]
    

    top_k_indices = np.argsort(probabilities)[-top_k:][::-1]
    top_k_labels = label_encoder.inverse_transform(top_k_indices)
    top_k_scores = probabilities[top_k_indices]
    
    # Create list of (language, score) tuples
    top_k_predictions = [
        (LANGUAGE_MAPPING.get(label, label), float(score))
        for label, score in zip(top_k_labels, top_k_scores)
    ]
    
    # Get primary prediction
    primary_label = label_encoder.inverse_transform([np.argmax(probabilities)])[0]
    primary_language = LANGUAGE_MAPPING.get(primary_label, primary_label)
    confidence = float(np.max(probabilities))
    
    return primary_language, confidence, top_k_predictions

def format_predictions(primary_lang: str, confidence: float, top_k: List[Tuple]) -> str:
    """Format prediction results for display"""
    result = [
        f"\nPrimary Prediction: {primary_lang} (confidence: {confidence:.2%})",
        "\nTop Predictions:"
    ]
    
    for i, (lang, score) in enumerate(top_k, 1):
        result.append(f"{i}. {lang}: {score:.2%}")
    
    return "\n".join(result)

def interactive_mode(vectorizer, model, label_encoder):
    """Run in interactive mode for multiple predictions"""
    print("\nInteractive Language Identification")
    print("Enter text to identify (or 'quit' to exit):")
    
    while True:
        try:
            text = input("\nInput text: ").strip()
            if text.lower() in ('quit', 'exit', 'q'):
                break
                
            if not text:
                print("Please enter some text.")
                continue
                
            primary, confidence, top_k = predict_language(
                text, vectorizer, model, label_encoder
            )
            print(format_predictions(primary, confidence, top_k))
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Language Identification using TF-IDF + Logistic Regression",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-t', '--text',
        help="Text to identify language (if not provided, enters interactive mode)",
        type=str,
        default=None
    )
    parser.add_argument(
        '-k', '--top-k',
        help="Number of top predictions to show",
        type=int,
        default=3
    )
    parser.add_argument(
        '-f', '--file',
        help="File containing text to identify",
        type=str,
        default=None
    )
    args = parser.parse_args()

    # Load models
    try:
        print("Loading language identification models...")
        vectorizer, model, label_encoder = load_models()
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Failed to load models: {str(e)}")
        return

    # Handle file input
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                args.text = f.read()
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return

    # Run prediction
    if args.text:
        try:
            primary, confidence, top_k = predict_language(
                args.text, vectorizer, model, label_encoder, args.top_k
            )
            print("\nText Sample:", args.text[:200] + ("..." if len(args.text) > 200 else ""))
            print(format_predictions(primary, confidence, top_k))
        except Exception as e:
            print(f"Prediction error: {str(e)}")
    else:
        interactive_mode(vectorizer, model, label_encoder)

if __name__ == "__main__":
    main()