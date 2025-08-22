# 🔧 دليل حل المشاكل - الشات بوت العربي

## المشكلة: البوت يقول "مشغول حالياً"

### السبب الرئيسي:
البوت يستخدم **OpenRouter API** الذي يعطي خطأ **429 Too Many Requests** عندما تتجاوز حد الطلبات المجانية.

### الحلول:

#### 1. انتظار فترة زمنية
- **خطة مجانية**: انتظر 5-10 دقائق بين الطلبات
- **خطط مدفوعة**: انتظر 1-2 دقيقة

#### 2. رفع مستوى الخطة
- اذهب إلى [OpenRouter](https://openrouter.ai/)
- ارفع مستوى خطتك للحصول على المزيد من الطلبات

#### 3. تغيير النموذج
- النماذج المجانية: `deepseek/deepseek-r1-0528:free`, `qwen/qwen3-30b-a3b:free`
- نماذج مدفوعة: `gpt-3.5-turbo`, `claude-3-haiku`, `deepseek/deepseek-v3.1-base`

### كيفية التحقق من المشكلة:

#### فحص السجلات:
```bash
tail -f chatbot.log | grep "429"
```

#### فحص حالة API:
```bash
curl http://localhost:8080/health
```

### إعدادات مهمة:

#### ملف .env:
```bash
# المفتاح الأساسي
OPENROUTER_API_KEY=your_key_here

# النموذج
OPENROUTER_MODEL=deepseek/deepseek-r1-0528:free

# حد الطلبات
RATE_LIMIT_MAX=30
RATE_LIMIT_TTL=60
```

#### متغيرات البيئة:
```bash
export OPENROUTER_API_KEY="your_key_here"
export OPENROUTER_MODEL="deepseek/deepseek-r1-0528:free"
```

### نصائح لتحسين الأداء:

1. **تقليل عدد الطلبات**: لا ترسل رسائل متتالية بسرعة
2. **استخدام Redis**: لتخزين المحادثات وتحسين الأداء
3. **ضبط النموذج**: استخدم نماذج أسرع للاستخدام المكثف

### رسائل الخطأ الشائعة:

- `429 Too Many Requests`: تجاوز حد الطلبات
- `503 Service Unavailable`: خدمة غير متاحة
- `500 Internal Server Error`: خطأ داخلي في الخادم

### للحصول على مساعدة إضافية:
- راجع [OpenRouter Documentation](https://openrouter.ai/docs)
- تحقق من [GitHub Issues](https://github.com/your-repo/issues)
