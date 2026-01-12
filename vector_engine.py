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
        self.index_file = INDEX_FILE
        self.embeddings_file = EMBEDDINGS_FILE

        if gemini_api_key:
            self._setup_gemini(gemini_api_key)

    def _setup_gemini(self, api_key):
        """Setup Gemini connection"""
        genai.configure(api_key=api_key)
        self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        print("Gemini activated!")

    def is_index_ready(self):
        """Check if previously saved files exist"""
        exists = os.path.exists(self.index_file) and os.path.exists(self.embeddings_file)
        if exists:
            print(f"Found previously saved index ({self.index_file})")
        return exists

    def load_saved_index(self):
        """Load the index and saved data immediately from disk"""
        print("Loading saved index... please wait a moment.")
        self.index = faiss.read_index(self.index_file)
        self.embeddings_array = np.load(self.embeddings_file)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"Index loaded successfully! ({len(self.embeddings_array)} books)")

    def clean_gpu_memory(self):
        """Clean memory to ensure no freezing"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("GPU memory cleaned")

    def create_embeddings(self, texts, show_progress=True):
        """Create text embeddings (fingerprints)"""
        print("Creating search database... (this will only happen once)")
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
        """Build search index and save it to disk"""
        if embeddings_array is None:
            embeddings_array = self.embeddings_array

        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)

        faiss.write_index(self.index, self.index_file)
        np.save(self.embeddings_file, self.embeddings_array)

        print(f"Index built and saved successfully! ({len(embeddings_array)} books)")
        print(f"  Saved files:")
        print(f"     - {self.index_file}")
        print(f"     - {self.embeddings_file}")
        return self.index

    def search(self, query, k=None):
        """Search for books most similar to the question"""
        if self.index is None or self.model is None:
            raise ValueError("Index or Model not initialized. Please load or create them first.")

        k = k if k else SEARCH_K
        query_emb = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_emb, k)
        return indices[0], distances[0]

    def ask_gemini(self, query, df, k=None):
        """Send the question with retrieved books to Gemini for a smart answer"""
        if self.gemini_model is None:
            raise ValueError("Gemini model not initialized.")

        k = k if k else SEARCH_K
        indices, distances = self.search(query, k)

        books_info = []
        for idx in indices:
            book = df.iloc[idx]
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
            print("BookMind Answer:")
            print(response.text)
            print("\nSuggested Books (Ranked by Relevance)")
            for i, idx in enumerate(indices):
                print(f"{i + 1}. {df.iloc[idx]['Title']} by {df.iloc[idx]['Author']}")
            print("=" * 50 + "\n")
        except Exception as e:
            print(f"Error with Gemini API: {e}")