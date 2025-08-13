# Language Identification Using TF-IDF and Logistic Regression

A comprehensive language identification system that classifies text samples across 20 major world languages using TF-IDF vectorization and logistic regression. This project achieves 98.23% accuracy on a balanced multilingual dataset from the GlotLID corpus.

## 🌍 Overview

This repository contains a complete implementation and evaluation of a language identification model designed to classify text across 20 of the world's most spoken languages. The model uses classical machine learning techniques (TF-IDF + Logistic Regression) to achieve state-of-the-art performance while maintaining interpretability and computational efficiency.

### Supported Languages

The model supports 20 languages spanning 8 language families and 10 writing systems:

| Language | Family | Script | Code |
|----------|--------|--------|------|
| Mandarin Chinese | Sino-Tibetan | Han | cmn_Hani |
| Spanish | Indo-European | Latin | spa_Latn |
| English | Indo-European | Latin | eng_Latn |
| Hindi | Indo-European | Devanagari | hin_Deva |
| Arabic | Afro-Asiatic | Arabic | arb_Arab |
| Bengali | Indo-European | Bengali | ben_Beng |
| Portuguese | Indo-European | Latin | por_Latn |
| Russian | Indo-European | Cyrillic | rus_Cyrl |
| Japanese | Japonic | Japanese | jpn_Jpan |
| Western Punjabi | Indo-European | Arabic | pnb_Arab |
| Marathi | Indo-European | Devanagari | mar_Deva |
| Telugu | Dravidian | Telugu | tel_Telu |
| Wu Chinese | Sino-Tibetan | Han | wuu_Hani |
| Turkish | Turkic | Latin | tur_Latn |
| Korean | Koreanic | Hangul | kor_Hang |
| French | Indo-European | Latin | fra_Latn |
| German | Indo-European | Latin | deu_Latn |
| Vietnamese | Austroasiatic | Latin | vie_Latn |
| Tamil | Dravidian | Tamil | tam_Taml |
| Urdu | Indo-European | Arabic | urd_Arab |

## 🚀 Key Features

- **High Accuracy**: 98.23% overall classification accuracy
- **Excellent Calibration**: Reliable confidence scores with calibration curves close to perfect
- **Robust Ranking**: 99.94% top-3 accuracy, 99.97% top-5 accuracy
- **Comprehensive Evaluation**: Detailed metrics, confusion analysis, and linguistic insights
- **Production Ready**: Well-calibrated probabilities suitable for real-world deployment
- **Interpretable**: Classical ML approach with clear feature importance and error patterns

## 📊 Performance Highlights

### **For figures and detailed report please view the PDF report.**

### Overall Metrics
- **Overall Accuracy**: 98.23%
- **Balanced Accuracy**: 98.23%
- **Macro F1-Score**: 98.23%
- **Weighted F1-Score**: 98.23%
- **Top-3 Accuracy**: 99.94%
- **Top-5 Accuracy**: 99.97%
- **Average ROC AUC**: 99.97%

### Language Family Performance
- **Sino-Tibetan**: 100% (Mandarin Chinese, Wu Chinese)
- **Koreanic**: 100% (Korean)
- **Japonic**: 100% (Japanese)
- **Turkic**: 99% (Turkish)
- **Dravidian**: 99% (Telugu, Tamil)
- **Indo-European**: 97% (varies by language due to shared features)

## 🏗️ Repository Structure

```
language-identification/
├── data_curation/
│   └── glotlid_top20/
│       └── outputs/           # Validation data files
├── models/                    # Trained model artifacts
│   ├── tfidf_vectorizer.pkl
│   ├── logreg_model.pkl
│   └── label_encoder.pkl
├── figures/                   # Generated evaluation plots
│   ├── class_distribution.png
│   ├── confusion_matrix_normalized.png
│   ├── f1_by_family.png
│   ├── calibration_curves.png
│   └── top_k_heatmap.png
├── reports/                   # Detailed evaluation reports
│   └── detailed_report.txt
├── evaluate_tfidf.py         # Main evaluation script
└── README.md
```

## 🛠️ Installation and Setup

### Prerequisites
- Python 3.7+
- Required packages (install via pip):

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib tqdm
```

### Dataset
The model uses the GlotLID corpus with:
- **Training**: 30,000 samples per language (600,000 total)
- **Validation**: 5,000 samples per language (100,000 total)

## 📈 Usage

### Running Evaluation

Execute the comprehensive evaluation script:

```bash
python evaluate_tfidf.py
```

This will:
1. Load validation data from the GlotLID corpus
2. Load pre-trained models (TF-IDF vectorizer, logistic regression classifier, label encoder)
3. Generate predictions and calculate comprehensive metrics
4. Create detailed visualizations and reports

### Model Architecture

The system uses a two-stage pipeline:

1. **TF-IDF Vectorization**:
   - Character n-grams (1-5 characters)
   - Captures both character-level and subword patterns
   - Effective for different scripts and languages

2. **Logistic Regression Classification**:
   - Stochastic Gradient Descent (SGD) optimization
   - Probabilistic outputs for confidence estimation
   - Multi-class classification with one-vs-rest approach

## 📊 Evaluation Framework

The evaluation includes multiple complementary analyses:

### 1. Classification Metrics
- Per-class and overall accuracy
- Precision, recall, and F1-scores
- Balanced accuracy for fairness assessment

### 2. Ranking Metrics
- Top-k accuracy (k=1,2,3,4,5)
- Reveals model's ranking capabilities beyond top-1 predictions

### 3. Calibration Analysis
- Reliability diagrams showing probability calibration
- Essential for confidence-based decision making

### 4. Confusion Analysis
- Normalized confusion matrix
- Identification of systematic error patterns
- Linguistic interpretation of misclassifications

### 5. Family-Level Analysis
- Performance grouped by language families
- Insights into script and linguistic similarity effects

## 🔍 Key Findings

### Perfect Performers
Languages with unique scripts achieved perfect classification:
- **Japanese** (100%)
- **Korean** (100%)
- **Mandarin Chinese** (100%)
- **Wu Chinese** (100%)
- **Tamil** (100%)
- **Spanish** (100%)
- **French** (100%)

### Main Challenge: English-Urdu Confusion
The primary source of errors is confusion between English and Urdu:
- **English → Urdu**: 557 misclassifications
- **Urdu → English**: 347 misclassifications

This confusion is linguistically motivated due to:
- Shared vocabulary from historical contact
- Romanized noise in datasets
- Similar structural patterns in certain contexts

### Script-Based Success
The model's effectiveness correlates strongly with script uniqueness:
- Unique scripts (Japanese, Korean, Chinese) → Perfect performance
- Shared scripts with distinct patterns (Latin-based European languages) → High performance
- Script overlap with linguistic similarity → Some confusion

## 🚀 Applications

This model is suitable for:
- **Content Management**: Automatically categorizing multilingual documents
- **Social Media**: Language detection for posts and comments
- **Translation Pipeline**: Pre-processing for machine translation systems
- **Web Crawling**: Filtering content by language
- **Academic Research**: Corpus linguistics and multilingual analysis

## 🔧 Model Artifacts

The repository includes pre-trained models:
- `tfidf_vectorizer.pkl`: TF-IDF feature extractor (1-5 character n-grams)
- `logreg_model.pkl`: Logistic regression classifier with SGD
- `label_encoder.pkl`: Maps between language names and numeric labels

## 📋 Requirements

### System Requirements
- **Memory**: ~2GB RAM for model loading and inference
- **Storage**: ~500MB for model artifacts and data
- **CPU**: Multi-core recommended for faster vectorization

### Dependencies
```
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0
tqdm>=4.62.0
```

## 🔗 References

- **GlotLID Corpus**: 

```
@inproceedings{
    kargaran2023glotlid,
    title={{G}lot{LID}: Language Identification for Low-Resource Languages},
    author={Kargaran, Amir Hossein and Imani, Ayyoob and Yvon, Fran{\c{c}}ois and Sch{\"u}tze, Hinrich},
    booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing},
    year={2023},
    url={https://openreview.net/forum?id=dl4e3EBz5j}
}
```

- **Scikit-learn**: Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please open an issue or contact in.sefat@tum.de or abd.al.sefat@gmail.com

---