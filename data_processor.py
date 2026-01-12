import pandas as pd
import ast
import os
import glob
import re
from config import (DATA_FILE, AUTO_CLEAN_ON_NEW_DATASET, INDEX_FILE, EMBEDDINGS_FILE)

def load_data(file_path=None):
    """Load data from CSV file"""
    if file_path is None:
        file_path = DATA_FILE
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    df = pd.read_csv(file_path)
    return df

def clean_genre_text(genres_value):
    """Convert complex lists to clean text that the model can understand"""
    if pd.isna(genres_value) or str(genres_value).strip() in ["", "[]", "nan"]:
        return ""
    try:
        parsed = ast.literal_eval(str(genres_value))
        if isinstance(parsed, list):
            return " ".join([str(g).strip() for g in parsed])
        return str(parsed).strip()
    except:
        clean = re.sub(r"[\[\]\'\"]", "", str(genres_value))
        return clean.replace(",", " ").strip()

def prepare_text_column(df):
    """Prepare and clean text columns to improve search accuracy"""
    df.columns = [col.lower() for col in df.columns]

    required_cols = ['title', 'author', 'description']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Required columns {required_cols} not found.")

    if 'genres' in df.columns:
        df['clean_genres'] = df['genres'].apply(clean_genre_text)
    else:
        df['clean_genres'] = ""

    for col in required_cols:
        df[col] = df[col].astype(str).replace('nan', '').fillna("")

    df["text"] = (
        "Book Title: " + df["title"] + " | " +
        "Category/Genres: " + df["clean_genres"] + " | " +
        "Written by: " + df["author"] + " | " +
        "Summary: " + df["description"]
    )

    df = df.rename(columns={
        'title': 'Title',
        'author': 'Author',
        'description': 'Description',
        'clean_genres': 'GenresStr'
    })

    print(f"Successfully prepared {len(df)} books with high accuracy!")
    return df

def clean_old_index_files(force=False):
    """Delete old index files"""
    if not force and not AUTO_CLEAN_ON_NEW_DATASET:
        return []
    deleted_files = []
    patterns = [INDEX_FILE, EMBEDDINGS_FILE, '*.faiss', '*.npy']
    for pattern in patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
            except:
                pass
    return deleted_files

def process_data(force_rebuild=False):
    """Main engine"""
    if force_rebuild:
        clean_old_index_files(force=True)
    df = load_data()
    df = prepare_text_column(df)
    return df