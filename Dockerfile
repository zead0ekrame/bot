FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# إنشاء ملف اللوج مع صلاحيات كاملة
RUN touch /app/chatbot.log && chmod 666 /app/chatbot.log

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8080

CMD ["bash", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
OPENROUTER_API_KEY=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