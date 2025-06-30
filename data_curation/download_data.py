from datasets import load_dataset
import pandas as pd
import os
from tqdm import tqdm
from huggingface_hub import snapshot_download
from huggingface_hub import login

# Top 20 most spoken languages with ISO-639-3 and script codes used in GlotLID
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

hf_token = "" # Insert HF Token here or set it as an environment variable
if hf_token:
    login(token=hf_token)
    print("Successfully logged in to Hugging Face!")
else:
    print("Token is not set. Please save the token first.")

output_dir = "glotlid_top20"
os.makedirs(output_dir, exist_ok=True)

version = "v3.1"

for name, code in tqdm(language_codes.items(), desc="Downloading languages"):
    try:
        folder = snapshot_download(
            repo_id="cis-lmu/glotlid-corpus",
            repo_type="dataset",
            local_dir=os.path.join(output_dir, code),
            allow_patterns=f"{version}/{code}/*",
        )
        # Load all .txt files in the downloaded folder for this code
        texts = []
        for root, _, files in os.walk(folder):
            for fname in files:
                if fname.endswith(".txt"):
                    path = os.path.join(root, fname)
                    with open(path, encoding="utf-8") as f:
                        texts.extend([line.strip() for line in f if line.strip()])
        # Save to one CSV
        df = pd.DataFrame({"text": texts})
        df.to_csv(os.path.join(output_dir, f"{code}.csv"), index=False)
    except Exception as e:
        print(f"❌ Failed for {name} ({code}): {e}")
