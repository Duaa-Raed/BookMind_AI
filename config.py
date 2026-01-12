import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(PROJECT_DIR, "Global_Library_Dataset.csv")
INDEX_FILE = os.path.join(PROJECT_DIR, "faiss_index.bin")
EMBEDDINGS_FILE = os.path.join(PROJECT_DIR, "embeddings.npy")

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

GEMINI_MODEL = "gemini-1.5-flash"

SEARCH_K = 3
BATCH_SIZE = 32

MIN_PRICE = 0
MAX_PRICE = 500
MIN_PAGES = 10
MAX_PAGES = 10000

PLT_FONT_FAMILY = "Arial"

AUTO_CLEAN_ON_NEW_DATASET = False

if not os.path.exists(DATA_FILE):
    print(f"Warning: Data file not found at: {DATA_FILE}")
    print(f"   Please ensure the CSV file 'books_data.csv' exists")