# Arabic Chatbot API (FastAPI + OpenRouter) with optional RAG

تشات بوت عربي مع تكامل OpenRouter، ودعم اختياري لـ RAG باستخدام LangChain وRedis Vector Store، مع واجهة محادثة جميلة شبيهة بـ ChatGPT.

## ✨ المميزات

- 🤖 **واجهة محادثة جميلة**: تصميم عصري شبيه بـ ChatGPT مع دعم اللغة العربية
- 🌐 **API قوي**: FastAPI مع دعم Facebook Messenger وWhatsApp
- 🧠 **ذكاء اصطناعي**: تكامل مع OpenRouter للـ LLM
- 🔍 **RAG اختياري**: استرجاع معلومات من قاعدة معرفة مخصصة
- 📱 **تصميم متجاوب**: يعمل على جميع الأجهزة
- 🌙 **وضع الظلام**: دعم كامل للوضع المظلم
- 💾 **تخزين محلي**: حفظ تاريخ المحادثات في المتصفح

## 🚀 Quick start

### 1. إعداد البيئة
```bash
# نسخ ملف البيئة
cp env.example .env

# تعديل المتغيرات في .env
# OPENROUTER_API_KEY=sk-or-...
# FACEBOOK_VERIFY_TOKEN=123456789
# FACEBOOK_PAGE_ACCESS_TOKEN=your_token_here
# FB_GRAPH_VERSION=v18.0
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
- **API**: http://localhost:8080
- **الواجهة**: http://localhost:8080/ui
- **التوثيق**: http://localhost:8080/docs

## 🎨 الواجهة

### المميزات
- تصميم عربي جميل وعصري
- دعم الوضع المظلم/الفاتح
- مؤشرات الكتابة
- تاريخ المحادثات
- تصميم متجاوب للموبايل

### الاستخدام
1. اكتب رسالتك في حقل الإدخال
2. اضغط Enter أو زر الإرسال
3. انتظر رد المساعد
4. استخدم زر "محادثة جديدة" لبدء محادثة جديدة
5. استخدم زر القمر/الشمس لتبديل الوضع

## 🔧 التكوين

### المتغيرات الأساسية
- `OPENROUTER_API_KEY`: مفتاح API لـ OpenRouter
- `OPENROUTER_MODEL`: نموذج LLM المستخدم
- `SYSTEM_PROMPT`: رسالة النظام الأساسية

### متغيرات RAG (اختيارية)
- `RAG_ENABLED`: تفعيل/تعطيل RAG
- `OPENAI_API_KEY`: مفتاح OpenAI للـ embeddings
- `REDIS_URL`: رابط Redis
- `RAG_INDEX_NAME`: اسم فهرس RAG

## 📚 RAG

### تفعيل RAG
1. تأكد من وجود Redis
2. أضف `OPENAI_API_KEY` في `.env`
3. شغل `RAG_ENABLED=true`

### إدخال البيانات
```bash
# إنشاء ملف CSV للمنتجات
python rag/ingest.py --source data/catalog.csv --index rag:retail_agri
```

### تنسيق CSV
```csv
name,description,price,unit
بذور طماطم,بذور طماطم عضوية عالية الجودة,25,كيلو
سماد عضوي,سماد طبيعي للخضروات,50,كيلو
```

## 🌐 API Endpoints

- `GET /`: الصفحة الرئيسية
- `GET /ui`: واجهة المحادثة
- `POST /chat`: إرسال رسالة
- `GET /health`: فحص صحة التطبيق
- `GET /rag/status`: حالة RAG
- `GET /metrics`: إحصائيات الأداء

## 📱 المنصات المدعومة

- **Facebook Messenger**: webhook في `/webhook`
- **WhatsApp**: webhook في `/webhook`
- **API مباشر**: endpoint في `/chat`

## 🛠️ التطوير

### هيكل المشروع
```
bot-master/
├── main.py              # FastAPI app
├── services.py          # منطق الأعمال
├── config.py            # الإعدادات
├── exceptions.py        # الاستثناءات
├── static/              # ملفات الواجهة
│   ├── index.html      # HTML الرئيسي
│   ├── styles.css      # التصميم
│   └── script.js       # JavaScript
├── rag/                 # ملفات RAG
│   └── ingest.py       # سكربت الإدخال
└── requirements.txt     # التبعيات
```

### إضافة ميزات جديدة
1. عدّل `services.py` للمنطق
2. أضف endpoints في `main.py`
3. حدث الواجهة في `static/`

## 🚀 النشر

### Docker
```bash
docker build -t arabic-chatbot .
docker run -p 8080:8080 --env-file .env arabic-chatbot
```

### Heroku/Render
- استخدم `Procfile` و `runtime.txt`
- أضف متغيرات البيئة في لوحة التحكم

## 📄 الترخيص

هذا المشروع مفتوح المصدر ومتاح للاستخدام التجاري.

## 🤝 المساهمة

نرحب بالمساهمات! يرجى:
1. Fork المشروع
2. إنشاء branch للميزة الجديدة
3. Commit التغييرات
4. Push للـ branch
5. إنشاء Pull Request

## 📞 الدعم

للمساعدة أو الأسئلة:
- افتح issue في GitHub
- راجع التوثيق في `/docs`
- تحقق من `/health` لفحص حالة التطبيق


