# ========== BookMind Configuration File ==========
# جميع الإعدادات المركزية للمشروع

import os

# ==================== المسارات ====================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(PROJECT_DIR, "Global_Library_Dataset.csv")
INDEX_FILE = os.path.join(PROJECT_DIR, "faiss_index.bin")
EMBEDDINGS_FILE = os.path.join(PROJECT_DIR, "embeddings.npy")

# ==================== نماذج الذكاء الاصطناعي ====================
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # نموذج Embedding سريع وفعال

GEMINI_MODEL = "gemini-1.5-flash"  # نموذج Gemini الأحدث والأسرع

# ==================== إعدادات البحث والمعالجة ====================
SEARCH_K = 3  # عدد أقرب الكتب المسترجعة من الفهرس
BATCH_SIZE = 32  # حجم الدفعة عند معالجة النصوص

# ==================== إعدادات تنقية البيانات ====================
MIN_PRICE = 0
MAX_PRICE = 500
MIN_PAGES = 10
MAX_PAGES = 10000

# ==================== إعدادات التصميم ====================
PLT_FONT_FAMILY = "Arial"

# ==================== إعدادات الفهرسة ====================
AUTO_CLEAN_ON_NEW_DATASET = False  # ❌ لا تحذف الفهرس القديم - احفظه!

# ==================== التحقق من وجود ملفات البيانات ====================
if not os.path.exists(DATA_FILE):
    print(f"⚠️  تحذير: ملف البيانات غير موجود في: {DATA_FILE}")
    print(f"   الرجاء التأكد من وجود ملف CSV باسم 'books_data.csv'")