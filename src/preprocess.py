import re, json, pickle
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_imdb(csv_path, output_dir, max_vocab=10000):
    df = pd.read_csv(csv_path)
    df['clean_review'] = df['review'].apply(clean_text)
    tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['clean_review'].tolist())
    with open(f"{output_dir}/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    vocab_size = min(max_vocab, len(tokenizer.word_index) + 1)
    stats = {"rows": len(df), "vocab_size": vocab_size}
    with open(f"{output_dir}/preprocess_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return df, tokenizer
