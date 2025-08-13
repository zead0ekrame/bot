FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# إنشاء ملف اللوج مع صلاحيات كاملة
RUN touch /app/chatbot.log && chmod 666 /app/chatbot.log

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8080

# هنا هنستخدم shell form عشان Render يقدر يقرأ متغير PORT
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
