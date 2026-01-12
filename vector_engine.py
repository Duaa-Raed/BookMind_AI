import numpy as np
import torch
import faiss
import os
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from config import EMBEDDING_MODEL, GEMINI_MODEL, BATCH_SIZE, SEARCH_K, INDEX_FILE, EMBEDDINGS_FILE


class VectorEngine:
    """Handles embeddings, FAISS index, and search operations with Saving/Loading features"""

    def __init__(self, gemini_api_key=None):
        self.model = None
        self.index = None
        self.embeddings_array = None
        self.gemini_model = None
        # أسماء ملفات الحفظ والتخزين من config
        self.index_file = INDEX_FILE
        self.embeddings_file = EMBEDDINGS_FILE

        if gemini_api_key:
            self._setup_gemini(gemini_api_key)

    def _setup_gemini(self, api_key):
        """إعداد اتصال Gemini"""
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        print("✓ Gemini activated!")

    def is_index_ready(self):
        """التحقق من وجود الملفات المحفوظة مسبقاً"""
        exists = os.path.exists(self.index_file) and os.path.exists(self.embeddings_file)
        if exists:
            print(f"✓ وجدت فهرس محفوظ سابقاً ({self.index_file})")
        return exists

    def load_saved_index(self):
        """تحميل الفهرس والبيانات المحفوظة فوراً من القرص"""
        print("⚡ تحميل الفهرس المحفوظ... يرجى الانتظار قليلاً.")
        self.index = faiss.read_index(self.index_file)
        self.embeddings_array = np.load(self.embeddings_file)
        # تحميل موديل التشفير لاستخدامه في تشفير الأسئلة الجديدة
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✓ تم تحميل الفهرس بنجاح! ({len(self.embeddings_array)} كتاب)")

    def clean_gpu_memory(self):
        """تنظيف الذاكرة لضمان عدم حدوث تعليق"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✓ تم تنظيف ذاكرة GPU")

    def create_embeddings(self, texts, show_progress=True):
        """إنشاء بصمات النصوص (Embeddings)"""
        print("🔍 جاري إنشاء قاعدة البحث... (هذا سيتم مرة واحدة فقط)")
        self.clean_gpu_memory()
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            batch_size=BATCH_SIZE
        )
        self.embeddings_array = np.array(embeddings).astype('float32')
        return self.embeddings_array

    def build_index(self, embeddings_array=None):
        """بناء فهرس البحث وحفظه على القرص"""
        if embeddings_array is None:
            embeddings_array = self.embeddings_array

        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)

        # حفظ الملفات فوراً بعد البناء
        faiss.write_index(self.index, self.index_file)
        np.save(self.embeddings_file, self.embeddings_array)

        print(f"✓ تم بناء الفهرس وحفظه بنجاح! ({len(embeddings_array)} كتاب)")
        print(f"  📁 الملفات المحفوظة:")
        print(f"     - {self.index_file}")
        print(f"     - {self.embeddings_file}")
        return self.index

    def search(self, query, k=None):
        """البحث عن الكتب الأكثر تشابهاً مع السؤال"""
        if self.index is None or self.model is None:
            raise ValueError("Index or Model not initialized. Please load or create them first.")

        k = k if k else SEARCH_K
        query_emb = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_emb, k)
        return indices[0], distances[0]

    def ask_gemini(self, query, df, k=None):
        """إرسال السؤال مع الكتب المسترجعة إلى Gemini للحصول على إجابة ذكية"""
        if self.gemini_model is None:
            raise ValueError("Gemini model not initialized.")

        k = k if k else SEARCH_K
        indices, distances = self.search(query, k)

        # تجميع معلومات الكتب المسترجعة كـ سياق (Context)
        books_info = []
        for idx in indices:
            book = df.iloc[idx]
            # معالجة الوصف في حال كان فارغاً
            desc = str(book['Description'])[:300] if book['Description'] is not None else "No description available."
            books_info.append(f"Title: {book['Title']}\nAuthor: {book['Author']}\nDescription: {desc}")

        context = "\n\n---\n\n".join(books_info)

        prompt = f"""
        You are 'BookMind', a helpful and expert library assistant.
        Use the following book data to answer the user's question. 
        If the books provided don't exactly match, explain why and suggest the most relevant ones.

        Context:
        {context}

        User Question: {query}

        Answer in a helpful way using the same language as the user.
        """

        try:
            response = self.gemini_model.generate_content(prompt)
            print("\n" + "=" * 50)
            print("**BookMind Answer:**")
            print(response.text)
            print("\n--- Suggested Books (Ranked by Relevance) ---")
            for i, idx in enumerate(indices):
                print(f"{i + 1}. {df.iloc[idx]['Title']} by {df.iloc[idx]['Author']}")
            print("=" * 50 + "\n")
        except Exception as e:
            print(f"❌ Error with Gemini API: {e}")