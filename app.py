# ========== BookMind AI - FINAL GLOWING & FIXED EDITION ==========

import streamlit as st
import pandas as pd
import google.generativeai as genai
from data_processor import process_data
from vector_engine import VectorEngine
import os
import warnings

# إعدادات الموديل المستقرة
GEMINI_MODEL_NAME = "gemini-1.5-flash"
SEARCH_K = 3

warnings.filterwarnings('ignore')

# 1. إعداد الصفحة
st.set_page_config(page_title="BookMind Library", layout="wide")

# 2. الهوية البصرية (Custom Glowing UI)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Cairo:wght@400;700&family=Playfair+Display:wght@900&display=swap');

    /* إخفاء العناصر المزعجة */
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stHeader"] {background: transparent !important;}
    [data-testid="stBottom"] {background: transparent !important; border: none !important;}

    /* الخلفية الفخمة */
    .stApp {
        background: #0f172a !important; 
        direction: rtl !important;
    }

    /* الاقتباسات في الخلفية */
    .stApp::before {
        content: 'To be or not to be... لسان الناس كتاب مرقوم... Je pense, donc je suis... القراءة حياة أخرى... The only thing you absolute have to know is the location of the library... اقرأ باسم ربك... Knowledge is power... الكتب بساتين العقلاء... Once you learn to read, you will be forever free...';
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        font-family: 'Amiri', serif;
        font-size: 2rem;
        color: rgba(255, 255, 255, 0.03); 
        padding: 50px;
        line-height: 2.5;
        z-index: -1;
        pointer-events: none;
        display: block;
    }

    /* تصميم اللوجو المضيء (Glowing Logo) */
    .logo-container {
        text-align: center;
        padding: 80px 0 40px 0;
    }

    .logo-text {
        font-family: 'Playfair Display', serif;
        font-size: 6rem;
        font-weight: 900;
        color: #ffffff;
        letter-spacing: -2px;
        text-shadow: 
            0 0 15px rgba(255,255,255,0.4),
            0 0 30px rgba(148, 163, 184, 0.3),
            0 0 50px rgba(148, 163, 184, 0.2);
        animation: glow 3s ease-in-out infinite alternate;
        position: relative;
        display: inline-block;
    }

    @keyframes glow {
        from {
            text-shadow: 0 0 10px rgba(255,255,255,0.4), 0 0 20px rgba(148, 163, 184, 0.2);
            transform: scale(1);
        }
        to {
            text-shadow: 0 0 25px rgba(255,255,255,0.7), 0 0 50px rgba(148, 163, 184, 0.5);
            transform: scale(1.03);
        }
    }

    .logo-tagline {
        font-family: 'Cairo', sans-serif;
        color: #94a3b8;
        font-size: 1.1rem;
        letter-spacing: 8px;
        text-transform: uppercase;
        margin-top: 10px;
        opacity: 0.8;
    }

    /* فقاعات المحادثة */
    .user-msg {
        background: linear-gradient(135deg, #334155 0%, #1e293b 100%);
        color: #f1f5f9;
        padding: 18px 25px;
        border-radius: 25px 25px 4px 25px;
        margin: 15px 0 15px auto;
        width: fit-content;
        max-width: 75%;
        border-right: 4px solid #94a3b8;
        box-shadow: 10px 10px 30px rgba(0,0,0,0.2);
        font-family: 'Cairo', sans-serif;
    }

    .bot-msg {
        background: rgba(30, 41, 59, 0.5);
        color: #cbd5e1;
        padding: 20px 25px;
        border-radius: 25px 25px 25px 4px;
        margin: 15px auto 15px 0;
        width: fit-content;
        max-width: 85%;
        border: 1px solid rgba(148, 163, 184, 0.2);
        backdrop-filter: blur(10px);
        font-family: 'Cairo', sans-serif;
    }

    /* شريط البحث */
    div[data-testid="stChatInput"] {
        border: 1px solid rgba(148, 163, 184, 0.5) !important;
        background: #1e293b !important;
        border-radius: 15px !important;
        box-shadow: 0 10px 40px rgba(0,0,0,0.4) !important;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    if 'messages' not in st.session_state: st.session_state.messages = []
    if 'initialized' not in st.session_state: st.session_state.initialized = False


def process_query_with_gemini(engine, query, df):
    try:
        indices, _ = engine.search(query, k=SEARCH_K)
        suggested_books = []
        books_context = []
        for idx in indices:
            book = df.iloc[idx]
            suggested_books.append({'title': book['Title'], 'author': book['Author']})
            books_context.append(f"كتاب {book['Title']} لـ {book['Author']}")

        # إعداد الـ API بالموديل الجديد 1.5-flash
        genai.configure(api_key=st.session_state.api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)

        prompt = f"أنت مساعد مكتبة محترف. اقترح الكتب التالية لطلب '{query}' بأسلوب راقٍ جداً وبدون استخدام أي رموز تعبيرية (إيموجي) نهائياً:\n" + "\n".join(
            books_context)

        response = model.generate_content(prompt)
        return response.text, suggested_books
    except Exception as e:
        # في حال حدوث خطأ في الموديل، نرجع الكتب بدون شرح طويل لضمان استمرار الخدمة
        return "بناءً على طلبك، إليك أفضل الترشيحات المتوفرة في قاعدة بياناتنا:", suggested_books


def main():
    initialize_session_state()

    # شعار BookMind المضيء
    st.markdown("""
    <div class="logo-container">
        <div class="logo-text">BookMind</div>
        <div class="logo-tagline">The Intelligent Library</div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.initialized:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                "<p style='text-align:center; color:#94a3b8; font-family: Cairo;'>أدخل مفتاح العبور لبدء التجربة</p>",
                unsafe_allow_html=True)
            api_key = st.text_input("Key", type="password", label_visibility="collapsed")
            if st.button("تفعيل النظام", use_container_width=True):
                if api_key:
                    st.session_state.api_key = api_key
                    with st.spinner("جاري تهيئة المكتبة الذكية..."):
                        try:
                            df = process_data()
                            st.session_state.df = df
                            engine = VectorEngine(gemini_api_key=api_key)
                            if not engine.is_index_ready():
                                engine.create_embeddings(df['text'].tolist())
                                engine.build_index()
                            else:
                                engine.load_saved_index()
                            st.session_state.engine = engine
                            st.session_state.initialized = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"خطأ في التهيئة: {e}")

    if st.session_state.initialized:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                books_html = "".join([
                    f'<div style="border-right:3px solid #94a3b8; padding:10px; margin-top:10px; background:rgba(255,255,255,0.03); border-radius: 4px;"><div style="color:#f1f5f9; font-weight:700;">{b["title"]}</div><div style="color:#94a3b8; font-size:0.85rem;">{b["author"]}</div></div>'
                    for b in msg.get("books", [])])
                st.markdown(f'<div class="bot-msg">{msg["content"]}{books_html}</div>', unsafe_allow_html=True)

        if prompt := st.chat_input("ماذا تود أن تقرأ اليوم؟"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("جاري البحث..."):
                ans, bks = process_query_with_gemini(st.session_state.engine, prompt, st.session_state.df)
                st.session_state.messages.append({"role": "assistant", "content": ans, "books": bks})
            st.rerun()


if __name__ == "__main__":
    main()