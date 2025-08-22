#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تهيئة فهرس RAG على Redis باستخدام LangChain

تشغيل:
  python rag/ingest.py --source data/catalog.csv --index rag:retail_agri
"""

import argparse
import os
import csv
from typing import List

from dotenv import load_dotenv

try:
    from langchain_openai import OpenAIEmbeddings  # type: ignore
    from langchain_community.vectorstores import Redis as RedisVectorStore  # type: ignore
except Exception as e:
    raise SystemExit(f"LangChain deps missing: {e}. Install requirements.txt first.")

load_dotenv()


def load_texts_from_csv(path: str) -> List[str]:
    """تحميل النصوص من ملف CSV مع تنسيق محسن للـ RAG"""
    texts: List[str] = []
    
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # إنشاء نص منظم للمنتج
            product_text = f"""
            المنتج: {row.get("name", "")}
            الوصف: {row.get("description", "")}
            السعر: {row.get("price", "")} {row.get("unit", "")}
            التوفر: {row.get("availability", "")}
            الموسم: {row.get("season", "")}
            
            معلومات إضافية:
            - هذا المنتج متوفر في متجرنا
            - يمكن طلبه عبر الصفحة أو الاتصال بنا
            - نقدم نصائح زراعية مجانية مع كل طلب
            - خدمة عملاء متاحة على مدار الساعة
            """
            
            # تنظيف النص وإزالة الأسطر الفارغة الزائدة
            cleaned_text = "\n".join(line.strip() for line in product_text.split('\n') if line.strip())
            texts.append(cleaned_text)
    
    return texts


def main():
    parser = argparse.ArgumentParser(description="إدخال بيانات المنتجات إلى فهرس RAG")
    parser.add_argument("--source", required=True, help="مسار ملف CSV")
    parser.add_argument("--index", default=os.getenv("RAG_INDEX_NAME", "rag:retail_agri"))
    parser.add_argument("--redis_url", default=os.getenv("REDIS_URL", "redis://localhost:6379"))
    args = parser.parse_args()

    # التحقق من وجود مفتاح OpenAI
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    if not openai_api_key:
        raise SystemExit("OPENAI_API_KEY مطلوب للـ embeddings. أضفه في ملف .env")

    # تحميل النصوص
    print(f"📖 جاري تحميل البيانات من {args.source}...")
    texts = load_texts_from_csv(args.source)
    
    if not texts:
        raise SystemExit("❌ لم يتم تحميل أي بيانات من الملف المصدر")
    
    print(f"✅ تم تحميل {len(texts)} منتج")

    # إنشاء embeddings
    print("🧠 جاري إنشاء embeddings...")
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    
    try:
        # إنشاء أو تحديث الفهرس
        print(f"💾 جاري حفظ البيانات في Redis index '{args.index}'...")
        RedisVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            index_name=args.index,
            redis_url=args.redis_url,
        )
        
        print(f"🎉 تم إدخال {len(texts)} منتج بنجاح في فهرس Redis '{args.index}'")
        print("\n📋 المنتجات المدخلة:")
        
        # عرض قائمة المنتجات
        for i, text in enumerate(texts[:5], 1):  # عرض أول 5 منتجات فقط
            lines = text.split('\n')
            product_name = lines[0].replace('المنتج: ', '')
            print(f"  {i}. {product_name}")
        
        if len(texts) > 5:
            print(f"  ... و {len(texts) - 5} منتج آخر")
            
        print(f"\n🔍 يمكنك الآن استخدام RAG في التطبيق!")
        print(f"📊 تحقق من حالة RAG: GET /rag/status")
        
    except Exception as e:
        print(f"❌ خطأ في حفظ البيانات: {e}")
        print("\n💡 تأكد من:")
        print("  - تشغيل Redis")
        print("  - صحة رابط Redis")
        print("  - صحة مفتاح OpenAI API")
        raise SystemExit(1)


if __name__ == "__main__":
    main()


