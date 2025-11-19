----
BookMind AI
---
----

*Advanced Book Search and Recommendation System*


**Overview**

A complete suite of intelligent tools for book search and recommendation, built across multiple versions that gradually introduce more advanced features:

| Version | Name | Status | Description |
|---------|------|--------|--------|
| **v1** | `book-recommender-nlp-v1` |  Stable | Basic recommendation system with semantic search |
| **v2** | `book-agent-v2` |  In Development | Advanced intelligent agent with multiple capabilities |


# Version 1: book-recommender-nlp-v1

### Objective

*A smart book search and recommendation system built using NLP and semantic retrieval techniques.*

### Features

- Semantic Search

Uses Sentence Transformers to understand the meaning behind user queries.

- Vector Database

Fast vector similarity search using FAISS.

- Enhanced Answers

Answers refined using Gemini AI for more natural and useful responses.

- Comprehensive Data Analysis (EDA)

Visual exploration of books dataset with multiple insights.

- Intelligent Cleaning

Automated data preprocessing and anomaly removal.

- Interactive Query Mode

Simple conversational interface for user queries.

### Core Capabilities

*Exploratory Data Analysis (EDA)*

Comprehensive visualizations including:

- Most prolific authors

- Most common categories

- Price and page distributions

- Correlation between price & page count

*Data Cleaning*

- Automated data processing:

- Remove missing/zero values

- Detect & remove outliers


### How Version 1 Works (Technical Pipeline)

Below is a concise overview of how *book-recommender-nlp-v1* works internally:

**Data Processing**
```python
import pandas as pd

df = pd.read_csv("data/books.csv")
df.dropna(subset=["title", "description"], inplace=True)
df = df[df["price"] > 0]  # remove invalid prices
```

**Embedding Generation**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["description"].tolist(), show_progress_bar=True)
```
**Vector Index with FAISS**
```python
import faiss
import numpy as np

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))
```
**Query Understanding + Semantic Search**
```python
def search_books(query, k=3):
    query_vec = model.encode([query])
    distances, ids = index.search(query_vec, k)
    return df.iloc[ids[0]]
```
**Gemini Refinement** 
```python
import google.generativeai as genai

def ask_gemini(query, context):
    prompt = f"User asked: {query}\nRelevant books:\n{context}\nProvide a helpful answer."
    return genai.GenerativeModel("gemini-pro").generate_content(prompt).text
```
**Final Response**

- Retrieve top 3 semantic matches

- Feed them as context to Gemini

- Generate final natural answer

### Technologies Used (Clean Version)
🔹 Data & Processing

Pandas, NumPy

🔹 NLP & Embeddings

Sentence Transformers

spaCy (light preprocessing)

🔹 Vector Search

FAISS (v1)

Redis / FAISS Hybrid (v2)

🔹 AI / LLM Integration

Google Gemini API

LangChain (v2)

🔹 Backend

Python CLI (v1)

FastAPI (v2)

🔹 Storage

CSV (v1)

PostgreSQL (v2)

🔹 Visualizations

Matplotlib, Seaborn
