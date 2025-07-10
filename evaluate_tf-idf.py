import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, top_k_accuracy_score,
    roc_auc_score, balanced_accuracy_score, f1_score
)
from sklearn.calibration import calibration_curve
from tqdm import tqdm
from collections import Counter
import textwrap

# ====================== CONFIGURATION ======================
# Constants with enhanced documentation
DATA_DIR = "data_curation/glotlid_top20/outputs/"  # Path to validation data
MODEL_DIR = "models/"  # Directory containing saved models
SAVE_FIGS_DIR = "figures/"  # Directory to save evaluation visualizations
REPORT_DIR = "reports/"  # Directory for text reports

LANGUAGE_METADATA = {
    "Mandarin Chinese": {"code": "cmn_Hani", "family": "Sino-Tibetan", "script": "Han"},
    "Spanish": {"code": "spa_Latn", "family": "Indo-European", "script": "Latin"},
    "English": {"code": "eng_Latn", "family": "Indo-European", "script": "Latin"},
    "Hindi": {"code": "hin_Deva", "family": "Indo-European", "script": "Devanagari"},
    "Arabic": {"code": "arb_Arab", "family": "Afro-Asiatic", "script": "Arabic"},
    "Bengali": {"code": "ben_Beng", "family": "Indo-European", "script": "Bengali"},
    "Portuguese": {"code": "por_Latn", "family": "Indo-European", "script": "Latin"},
    "Russian": {"code": "rus_Cyrl", "family": "Indo-European", "script": "Cyrillic"},
    "Japanese": {"code": "jpn_Jpan", "family": "Japonic", "script": "Japanese"},
    "Western Punjabi": {"code": "pnb_Arab", "family": "Indo-European", "script": "Arabic"},
    "Marathi": {"code": "mar_Deva", "family": "Indo-European", "script": "Devanagari"},
    "Telugu": {"code": "tel_Telu", "family": "Dravidian", "script": "Telugu"},
    "Wu Chinese": {"code": "wuu_Hani", "family": "Sino-Tibetan", "script": "Han"},
    "Turkish": {"code": "tur_Latn", "family": "Turkic", "script": "Latin"},
    "Korean": {"code": "kor_Hang", "family": "Koreanic", "script": "Hangul"},
    "French": {"code": "fra_Latn", "family": "Indo-European", "script": "Latin"},
    "German": {"code": "deu_Latn", "family": "Indo-European", "script": "Latin"},
    "Vietnamese": {"code": "vie_Latn", "family": "Austroasiatic", "script": "Latin"},
    "Tamil": {"code": "tam_Taml", "family": "Dravidian", "script": "Tamil"},
    "Urdu": {"code": "urd_Arab", "family": "Indo-European", "script": "Arabic"}
}

# mappings
language_codes = {lang: meta["code"] for lang, meta in LANGUAGE_METADATA.items()}
label_to_lang = {v: k for k, v in language_codes.items()}
label_list = list(language_codes.values())
lang_names = list(LANGUAGE_METADATA.keys())

#
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_palette("husl")

# ====================== CORE FUNCTIONS ======================

def load_validation_data():
    """
    Load validation data with enhanced error handling and progress tracking.
    Returns:
        texts (list): List of text samples
        labels (list): Corresponding language labels
        count_df (DataFrame): Statistics about loaded data
    """
    texts, labels = [], []
    file_stats = []
    
    print("Loading validation data...")
    for label in tqdm(label_list, desc="Processing language files"):
        file_path = os.path.join(DATA_DIR, f"{label}_val.csv")
        if not os.path.exists(file_path):
            print(f"\nWarning: {file_path} not found. Skipping...")
            continue
            
        try:
            df = pd.read_csv(file_path)
            sample_count = len(df)
            texts.extend(df['text'].astype(str).tolist())
            labels.extend([label] * sample_count)
            
            # Collect stats
            lang_name = label_to_lang[label]
            file_stats.append({
                "Language": lang_name,
                "Language Family": LANGUAGE_METADATA[lang_name]["family"],
                "Script": LANGUAGE_METADATA[lang_name]["script"],
                "Samples": sample_count,
                "Loaded": True
            })
        except Exception as e:
            print(f"\nError loading {file_path}: {str(e)}")
            file_stats.append({
                "Language": label_to_lang.get(label, "Unknown"),
                "Samples": 0,
                "Loaded": False,
                "Error": str(e)
            })
    
    # Create stats dataframe
    count_df = pd.DataFrame(file_stats)
    total_samples = len(texts)
    
    print(f"\nSuccessfully loaded {total_samples:,} samples across {len(label_list)} languages")
    return texts, labels, count_df


def generate_classification_report(y_true, y_pred, y_proba, labels, label_names, report_dir):
    """
    Generate comprehensive classification report with extended metrics.
    Saves both console output and detailed text report.
    """
    # Standard classification report
    clf_report = classification_report(
        y_true, y_pred, 
        target_names=label_names,
        zero_division=0, 
        digits=4,
        output_dict=True
    )
    
    # Convert to dataframe for better handling
    report_df = pd.DataFrame(clf_report).transpose()
    
    # Add additional metrics
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Calculate top-k metrics
    top3_acc = top_k_accuracy_score(y_true, y_proba, k=3, labels=labels)
    top5_acc = top_k_accuracy_score(y_true, y_proba, k=5, labels=labels)
    
    # Create summary dictionary
    summary_metrics = {
        "Overall Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_acc,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Top-3 Accuracy": top3_acc,
        "Top-5 Accuracy": top5_acc,
        "Average ROC AUC": roc_auc_score(
            y_true, y_proba, multi_class='ovr', average='weighted'
        )
    }
    
    # Save detailed report to file
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "detailed_report.txt"), "w") as f:
        f.write("=== LANGUAGE IDENTIFICATION MODEL EVALUATION ===\n\n")
        f.write("=== SUMMARY METRICS ===\n")
        for metric, value in summary_metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
        
        f.write("\n=== PER-CLASS METRICS ===\n")
        report_df.to_string(f)
        
        # Add confusion matrix analysis
        f.write("\n\n=== CONFUSION ANALYSIS ===\n")
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        most_confused = analyze_confusion_matrix(cm, label_names)
        f.write("\nMost Frequent Confusions:\n")
        for pair, count in most_confused.items():
            f.write(f"{pair}: {count} misclassifications\n")
    
    # Print summary to console
    print("\n=== MODEL EVALUATION SUMMARY ===")
    for metric, value in summary_metrics.items():
        print(f"{metric:<20}: {value:.4f}")
    
    return report_df, summary_metrics


def analyze_confusion_matrix(cm, label_names):
    """Identify most common confusions from the confusion matrix"""
    confusions = {}
    n = len(label_names)
    
    for i in range(n):
        for j in range(n):
            if i != j and cm[i,j] > 0:
                pair = f"{label_names[i]} → {label_names[j]}"
                confusions[pair] = cm[i,j]
    
    # Return top 10 most common confusions
    return dict(sorted(confusions.items(), key=lambda item: item[1], reverse=True)[:10])


def plot_class_distribution(count_df, filename):
    """Visualize class distribution with family/script information"""
    plt.figure(figsize=(14, 8))
    
    # Sort by sample count
    count_df = count_df.sort_values("Samples", ascending=False)
    
    # Create color mapping for language families
    families = count_df["Language Family"].unique()
    family_palette = sns.color_palette("husl", len(families))
    family_colors = dict(zip(families, family_palette))
    
    # Create plot
    ax = sns.barplot(
        data=count_df,
        x="Samples",
        y="Language",
        hue="Language Family",
        palette=family_colors,
        dodge=False
    )
    
    # Add script information
    for i, (_, row) in enumerate(count_df.iterrows()):
        ax.text(
            row["Samples"] + max(count_df["Samples"]) * 0.01,
            i,
            row["Script"],
            ha='left',
            va='center',
            fontsize=9,
            color='gray'
        )
    
    plt.title("Validation Set Class Distribution\n(Language Family and Script)", pad=20)
    plt.xlabel("Number of Samples")
    plt.ylabel("Language")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels, label_names, filename):
    """Enhanced confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Wrap long language names
    wrapped_labels = ['\n'.join(textwrap.wrap(name, width=15)) for name in label_names]
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=wrapped_labels,
        yticklabels=wrapped_labels,
        cbar_kws={'label': 'Normalized Proportion'}
    )
    
    plt.title('Normalized Confusion Matrix\n(Rows normalized to 1)', pad=20)
    plt.xlabel('Predicted Language', labelpad=10)
    plt.ylabel('True Language', labelpad=10)
    plt.xticks(rotation=90, ha='center')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_metric_by_family(metrics_df, metric_name, filename):
    """Visualize performance metrics grouped by language family"""
    plt.figure(figsize=(12, 12))
    
    # Prepare data
    plot_df = metrics_df.copy()
    plot_df['Language Family'] = plot_df.index.map(
        lambda x: LANGUAGE_METADATA[x]["family"]
    )
    
    # Calculate family averages
    family_avg = plot_df.groupby('Language Family')[metric_name].mean()
    
    # Sort languages by family then metric value
    plot_df = plot_df.sort_values(
        by=['Language Family', metric_name],
        ascending=[True, False]
    )
    
    # Create color mapping
    families = plot_df['Language Family'].unique()
    palette = sns.color_palette("husl", len(families))
    family_colors = dict(zip(families, palette))
    
    # Create plot
    ax = sns.barplot(
        data=plot_df,
        x=metric_name,
        y=plot_df.index,
        hue='Language Family',
        palette=family_colors,
        dodge=False
    )
    
    # Add family average lines
    for i, family in enumerate(families):
        avg = family_avg[family]
        ax.axvline(
            avg,
            color=family_colors[family],
            linestyle='--',
            alpha=0.7,
            linewidth=1
        )
        ax.text(
            avg * 1.01,
            len(plot_df) - (i * 5) - 3,
            f'{family} avg: {avg:.2f}',
            color=family_colors[family],
            va='center'
        )
    
    plt.title(f'{metric_name} by Language (Grouped by Family)', pad=20)
    plt.xlabel(metric_name)
    plt.ylabel('Language')
    plt.xlim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_calibration_curve(y_true, y_proba, labels, label_names, filename):
    """Visualize model calibration by language"""
    plt.figure(figsize=(12, 10))
    
    # Select a subset of languages for clarity
    n_languages = len(labels)
    if n_languages > 20:
        # Select languages with most samples or best/worst calibration
        sample_counts = np.bincount(y_true)
        selected_indices = np.argsort(sample_counts)[-12:]  # Top 12 by sample size
    else:
        selected_indices = range(n_languages)
    
    # Plot calibration for each selected language
    for i in selected_indices:
        true_binary = (y_true == labels[i]).astype(int)
        prob_true, prob_pred = calibration_curve(
            true_binary,
            y_proba[:, i],
            n_bins=10,
            strategy='quantile'
        )
        
        plt.plot(
            prob_pred,
            prob_true,
            marker='o',
            markersize=4,
            label=f"{label_names[i]}"
        )
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect calibration")
    
    plt.title('Calibration Curves by Language', pad=20)
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_top_k_heatmap(y_true, y_proba, labels, label_names, max_k=5, filename=None):
    """Visualize top-k accuracy across different k values"""
    k_values = range(1, max_k + 1)
    k_accuracies = []
    
    for k in k_values:
        k_acc = []
        for class_idx in range(len(labels)):
            mask = (y_true == labels[class_idx])
            if np.sum(mask) == 0:
                k_acc.append(0.0)
                continue
            acc = top_k_accuracy_score(
                y_true[mask], y_proba[mask], k=k, labels=labels
            )
            k_acc.append(acc)
        k_accuracies.append(k_acc)
    
    # Create dataframe for heatmap
    heatmap_df = pd.DataFrame(
        np.array(k_accuracies).T,
        index=label_names,
        columns=[f"Top-{k}" for k in k_values]
    )
    
    # Sort by Top-1 accuracy
    heatmap_df = heatmap_df.sort_values("Top-1", ascending=False)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        linewidths=0.5
    )
    
    plt.title(f'Top-k Accuracy by Language (k=1 to {max_k})', pad=20)
    plt.xlabel('k Value')
    plt.ylabel('Language')
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def evaluate_model():
    """Main evaluation function with enhanced reporting"""
    print("\n=== LANGUAGE IDENTIFICATION MODEL EVALUATION ===")
    
    # Create output directories
    os.makedirs(SAVE_FIGS_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # Load data with statistics
    X_val_texts, y_val_labels, count_df = load_validation_data()
    
    # Load models
    print("\nLoading trained models...")
    try:
        vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
        clf = joblib.load(os.path.join(MODEL_DIR, "logreg_model.pkl"))
        le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return
    
    # Transform and predict
    print("\nTransforming validation data...")
    X_val_vec = vectorizer.transform(X_val_texts)
    
    print("Making predictions...")
    y_val = le.transform(y_val_labels)
    y_pred = clf.predict(X_val_vec)
    y_proba = clf.predict_proba(X_val_vec)
    
    # Generate comprehensive report
    report_df, summary_metrics = generate_classification_report(
        y_val, y_pred, y_proba,
        labels=np.arange(len(le.classes_)),
        label_names=lang_names,
        report_dir=REPORT_DIR
    )
    
    # Generate visualizations
    print("\nGenerating evaluation visualizations...")
    
    # 1. Class distribution
    plot_class_distribution(
        count_df,
        filename=os.path.join(SAVE_FIGS_DIR, "class_distribution.png")
    )
    
    # 2. Confusion matrix
    plot_confusion_matrix(
        y_val, y_pred,
        labels=np.arange(len(le.classes_)),
        label_names=lang_names,
        filename=os.path.join(SAVE_FIGS_DIR, "confusion_matrix_normalized.png")
    )
    
    # 3. Performance by language family
    plot_metric_by_family(
        report_df.loc[lang_names],
        'f1-score',
        filename=os.path.join(SAVE_FIGS_DIR, "f1_by_family.png")
    )
    
    # 4. Calibration curves
    plot_calibration_curve(
        y_val, y_proba,
        labels=np.arange(len(le.classes_)),
        label_names=lang_names,
        filename=os.path.join(SAVE_FIGS_DIR, "calibration_curves.png")
    )
    
    # 5. Top-k accuracy heatmap
    plot_top_k_heatmap(
        y_val, y_proba,
        labels=np.arange(len(le.classes_)),
        label_names=lang_names,
        max_k=5,
        filename=os.path.join(SAVE_FIGS_DIR, "top_k_heatmap.png")
    )
    
    print(f"\nEvaluation complete. Results saved in:")
    print(f"- Visualizations: {SAVE_FIGS_DIR}")
    print(f"- Reports: {REPORT_DIR}")


if __name__ == "__main__":
    evaluate_model()