# ========== DATA PROCESSING (OPTIMIZED FOR MULTIPLE GENRES) ==========

import pandas as pd
import ast
import os
import glob
import re
from config import (DATA_FILE, AUTO_CLEAN_ON_NEW_DATASET, INDEX_FILE, EMBEDDINGS_FILE)

def load_data(file_path=None):
    """تحميل البيانات من ملف CSV"""
    if file_path is None:
        file_path = DATA_FILE
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"لم يتم العثور على ملف البيانات في: {file_path}")
    df = pd.read_csv(file_path)
    return df

def clean_genre_text(genres_value):
    """تحويل القوائم المعقدة إلى نص نظيف يسهل على الموديل فهمه"""
    if pd.isna(genres_value) or str(genres_value).strip() in ["", "[]", "nan"]:
        return ""
    try:
        # محاولة تحويل النص إلى قائمة حقيقية
        parsed = ast.literal_eval(str(genres_value))
        if isinstance(parsed, list):
            return " ".join([str(g).strip() for g in parsed])
        return str(parsed).strip()
    except:
        # إذا لم يكن بصيغة قائمة، ننظفه من الرموز يدوياً
        clean = re.sub(r"[\[\]\'\"]", "", str(genres_value))
        return clean.replace(",", " ").strip()

def prepare_text_column(df):
    """تجهيز وتنظيف أعمدة النصوص لزيادة دقة البحث"""
    # توحيد الأسماء
    df.columns = [col.lower() for col in df.columns]

    required_cols = ['title', 'author', 'description']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"الأعمدة المطلوبة {required_cols} غير موجودة.")

    # 1. تنظيف التصنيفات (Genres) لتصبح نصاً مفهوماً
    if 'genres' in df.columns:
        df['clean_genres'] = df['genres'].apply(clean_genre_text)
    else:
        df['clean_genres'] = ""

    # 2. تنظيف النصوص الأساسية
    for col in required_cols:
        df[col] = df[col].astype(str).replace('nan', '').fillna("")

    # 3. بناء نص البحث (هذا ما يراه محرك البحث FAISS)
    # وضعنا العنوان والتصنيفات في البداية لزيادة وزنهم (Weight) في البحث
    df["text"] = (
        "Book Title: " + df["title"] + " | " +
        "Category/Genres: " + df["clean_genres"] + " | " +
        "Written by: " + df["author"] + " | " +
        "Summary: " + df["description"]
    )

    # 4. إعادة التسمية للواجهة
    df = df.rename(columns={
        'title': 'Title',
        'author': 'Author',
        'description': 'Description',
        'clean_genres': 'GenresStr'
    })

    print(f"✅ تم تجهيز {len(df)} كتاب بدقة عالية!")
    return df

def clean_old_index_files(force=False):
    """حذف ملفات الفهرس القديمة"""
    if not force and not AUTO_CLEAN_ON_NEW_DATASET:
        return []
    deleted_files = []
    patterns = [INDEX_FILE, EMBEDDINGS_FILE, '*.faiss', '*.npy']
    for pattern in patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                deleted_files.append(file_path)
            except: pass
    return deleted_files

def process_data(force_rebuild=False):
    """المحرك الرئيسي"""
    if force_rebuild:
        clean_old_index_files(force=True)
    df = load_data()
    df = prepare_text_column(df)
    return df