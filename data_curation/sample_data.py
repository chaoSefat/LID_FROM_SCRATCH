import pandas as pd
import os

language_codes = {
    "Mandarin Chinese": "cmn_Hani",
    "Spanish": "spa_Latn",
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "Arabic": "arb_Arab",
    "Bengali": "ben_Beng",
    "Portuguese": "por_Latn",
    "Russian": "rus_Cyrl",
    "Japanese": "jpn_Jpan",
    "Western Punjabi": "pnb_Arab",
    "Marathi": "mar_Deva",
    "Telugu": "tel_Telu",
    "Wu Chinese": "wuu_Hani",
    "Turkish": "tur_Latn",
    "Korean": "kor_Hang",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Vietnamese": "vie_Latn",
    "Tamil": "tam_Taml",
    "Urdu": "urd_Arab"
}

def select_random_rows(input_csv, N, n_val):
    # Define output directory path
    output_dir = os.path.join(os.getcwd(), 'outputs')

    # Fallback: create 'outputs' directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"🛠️ Created output directory at: {output_dir}")
    else:
        print(f"📁 Output directory already exists at: {output_dir}")

    # Read the input CSV
    try:
        df = pd.read_csv(f'{input_csv}.csv')
    except Exception as e:
        print(f"❌ Error reading the input file {input_csv}.csv: {e}")
        return

    total_required = N + n_val
    if total_required > len(df):
        print(f"⚠️ Not enough rows in the dataset. Requested {total_required}, but only {len(df)} available.")
        return

    # Shuffle the data
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into train and validation sets
    df_train = df_shuffled.iloc[:N]
    df_val = df_shuffled.iloc[N:N + n_val]

    # Output file paths
    train_output_path = os.path.join(output_dir, f'{input_csv}_train.csv')
    val_output_path = os.path.join(output_dir, f'{input_csv}_val.csv')

    # Save the results
    try:
        df_train.to_csv(train_output_path, index=False)
        print(f"✅ Saved {N} rows to training set: {train_output_path}")
        df_val.to_csv(val_output_path, index=False)
        print(f"✅ Saved {n_val} rows to validation set: {val_output_path}")
    except Exception as e:
        print(f"❌ Error saving the CSV files: {e}")
if __name__ == "__main__":
    # Example usage
    for name, code in language_codes.items():
        input_csv = os.path.join("glotlid_top20", code)
        select_random_rows(input_csv, N=30000, n_val=5000)
        print(f"Processed {name} ({code}) dataset.")