# 🚀 دليل البدء السريع - الشات بوت العربي

## ⚡ تشغيل سريع (5 دقائق)

### 1. إعداد البيئة
```bash
# نسخ ملف البيئة
cp env.example .env

# تعديل المتغيرات الأساسية في .env
# OPENROUTER_API_KEY=sk-or-... (مطلوب)
# OPENAI_API_KEY=sk-... (اختياري للـ RAG)
```

### 2. تثبيت التبعيات
```bash
pip install -r requirements.txt
```

### 3. تشغيل التطبيق
```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

### 4. الوصول للواجهة
- 🌐 **الواجهة**: http://localhost:8080/ui
- 📚 **API Docs**: http://localhost:8080/docs
- 🔍 **فحص الصحة**: http://localhost:8080/health

## 🎯 ما يمكنك فعله الآن

### ✅ يعمل فوراً
- واجهة محادثة جميلة باللغة العربية
- تكامل مع OpenRouter للذكاء الاصطناعي
- دعم Facebook Messenger وWhatsApp
- حفظ تاريخ المحادثات
- تصميم متجاوب للموبايل

### 🔧 يحتاج إعداد
- **RAG**: لإضافة معرفة بالمنتجات
- **Redis**: لتخزين أفضل
- **مفاتيح المنصات**: لـ Facebook/WhatsApp

## 🌟 تفعيل RAG (اختياري)

### 1. إعداد Redis
```bash
# Ubuntu/Debian
sudo apt install redis-server
sudo systemctl start redis

# أو Docker
docker run -d -p 6379:6379 redis:alpine
```

### 2. إضافة مفتاح OpenAI
```bash
# في ملف .env
OPENAI_API_KEY=sk-...
```

### 3. إدخال البيانات
```bash
python rag/ingest.py --source data/catalog.csv
```

### 4. التحقق من الحالة
```bash
curl http://localhost:8080/rag/status
```

## 📱 اختبار الواجهة

1. افتح http://localhost:8080/ui
2. اكتب رسالة مثل: "أريد شراء بذور طماطم"
3. انتظر رد المساعد
4. جرب أسئلة مختلفة

## 🔍 استكشاف الأخطاء

### مشكلة: "Redis غير متاح"
- **الحل**: التطبيق يعمل بالذاكرة المؤقتة
- **لتحسين**: شغل Redis

### مشكلة: "OPENAI_API_KEY missing"
- **الحل**: RAG معطل، التطبيق يعمل طبيعي
- **لتفعيل RAG**: أضف مفتاح OpenAI

### مشكلة: "فشل في الاتصال بخدمة الذكاء"
- **الحل**: تحقق من `OPENROUTER_API_KEY`
- **لاختبار**: استخدم `/health` endpoint

## 📞 المساعدة

- 📖 **التوثيق الكامل**: README.md
- 🔧 **API**: /docs
- 📊 **الحالة**: /health
- 🆘 **RAG**: /rag/status

## 🎉 تهانينا!

لديك الآن شات بوت عربي متكامل مع:
- واجهة جميلة شبيهة بـ ChatGPT
- دعم كامل للغة العربية
- تكامل مع أفضل نماذج الذكاء الاصطناعي
- إمكانية إضافة معرفة مخصصة (RAG)
- دعم منصات التواصل الاجتماعي

ابدأ المحادثة الآن! 🚀
