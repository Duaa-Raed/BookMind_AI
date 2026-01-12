# ========== BookMind - Main Script ==========

import os
from data_processor import process_data
from vector_engine import VectorEngine
from config import GEMINI_MODEL


def main():
    """البرنامج الرئيسي للنظام"""

    print("\n" + "=" * 60)
    print("🚀 مرحباً بك في BookMind - نظام توصية الكتب الذكي")
    print("=" * 60 + "\n")

    # 1. معالجة البيانات
    print("📚 جاري تحميل بيانات الكتب...")
    try:
        df = process_data()
        print(f"✅ تم تحميل {len(df)} كتاب بنجاح!\n")
    except Exception as e:
        print(f"❌ خطأ في تحميل البيانات: {e}")
        return

    # 2. طلب مفتاح API
    print("=" * 60)
    api_key = input("🔑 أدخل مفتاح Gemini API: ").strip()
    print("=" * 60 + "\n")

    if not api_key:
        print("❌ يجب إدخال مفتاح API!")
        return

    # 3. إعداد محرك البحث
    print("⚙️  جاري تهيئة محرك البحث...")
    try:
        engine = VectorEngine(gemini_api_key=api_key)

        # التحقق من وجود فهرس محفوظ
        if engine.is_index_ready():
            print("⚡ جاري تحميل الفهرس المحفوظ...")
            engine.load_saved_index()
        else:
            print("🔍 جاري إنشاء فهرس البحث (هذا قد يستغرق دقيقة)...")
            engine.create_embeddings(df['text'].tolist())
            engine.build_index()

        print("✅ المحرك جاهز للعمل!\n")
    except Exception as e:
        print(f"❌ خطأ في إعداد المحرك: {e}")
        return

    # 4. وضع البحث التفاعلي
    print("=" * 60)
    print("💬 ابدأ بالبحث عن الكتب (اكتب 'exit' للخروج)")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("🔍 ما الكتاب الذي تبحث عنه؟ ").strip()

            if query.lower() in ['exit', 'quit', 'خروج']:
                print("\n👋 شكراً لاستخدام BookMind! وداعاً!")
                break

            if not query:
                print("⚠️  يرجى إدخال سؤال صحيح.\n")
                continue

            print("\n" + "=" * 60)
            print("🔎 جاري البحث والتحليل...")
            print("=" * 60 + "\n")

            # استدعاء دالة البحث والرد
            engine.ask_gemini(query=query, df=df)

        except KeyboardInterrupt:
            print("\n\n👋 تم إيقاف البرنامج.")
            break
        except Exception as e:
            print(f"❌ خطأ: {e}\n")


if __name__ == "__main__":
    main()