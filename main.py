# -*- coding: utf-8 -*-
"""
🚀 الشات بوت العربي المثالي (FastAPI + OpenRouter + Redis + Facebook Messenger/WhatsApp)
نسخة محسنة مع:
✅ فصل الإعدادات في ملف config.py
✅ إدارة الأخطاء المخصصة في exceptions.py  
✅ منطق الأعمال في services.py
✅ تحسين الأداء والأمان
✅ كود أكثر تنظيماً وقابلية للصيانة
"""

import os
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import Response, ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware



# استيراد الملفات الجديدة
from config import settings
from exceptions import ChatbotException, to_http_exception
from services import chat_service, init_http_client, close_http_client

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("arabic-chatbot")

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_http_client()
    try:
        yield
    finally:
        await close_http_client()


app = FastAPI(
    title="🤖 Arabic Chatbot API", 
    description="الشات بوت العربي المثالي - آمن، سريع، وذكي", 
    version="3.1.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # السماح لجميع المصادر للـ webhook
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# Lifespan managed above
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Exception Handler
# ---------------------------------------------------------------------------
@app.exception_handler(ChatbotException)
async def chatbot_exception_handler(request: Request, exc: ChatbotException):
    """معالج الاستثناءات المخصصة"""
    logger.error(f"ChatbotException: {exc.message} - {exc.error_code}")
    raise to_http_exception(exc)

@app.get("/")
async def root():
    return {
        "message": "🤖 مرحباً بك في الشات بوت العربي!", 
        "version": "3.1.0",
        "webhooks": {
            "messenger": "/webhook",
            "whatsapp": "/webhook"
        }
    }

@app.post("/test-chat")
async def test_chat(request: Request):
    """اختبار البوت مباشرة"""
    try:
        body = await request.json()
        message = body.get("message", "")
        if not message:
            return {"error": "الرسالة مطلوبة"}
        
        # توليد الرد
        reply = await chat_service.llm.generate_response([
            {"role": "system", "content": settings.system_prompt},
            {"role": "user", "content": message}
        ])
        
        return {"reply": reply, "message": message}
    except Exception as e:
        logger.exception(f"خطأ في test_chat: {e}")
        return {"error": str(e)}

@app.get("/health")
async def health_check(fast: bool = False):
    """Health check:
    - fast=true  → لا يتواصل مع OpenRouter (للتحقق السريع)
    - fast=false → يحاول يتأكد من اتصال OpenRouter.
    """
    if fast:
        return {"status": "ok-local", "storage": "redis" if chat_service.storage.use_redis else "memory"}

    try:
        import httpx
        from config import HEADERS, OPENROUTER_URL
        async with httpx.AsyncClient() as client:
            r = await client.get(
                OPENROUTER_URL.replace("/chat/completions", "/models"),
                headers=HEADERS,
                timeout=5.0,
            )
            api_ok = r.status_code == 200
            return {"status": "healthy" if api_ok else "unhealthy", "api_status": r.status_code}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}





# ---------------------------------------------------------------------------
# Messenger Webhook
# ---------------------------------------------------------------------------
@app.get("/webhook")
async def verify_webhook(
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
):
    """تحقق من webhook Facebook"""
    if hub_mode == "subscribe" and hub_verify_token == settings.facebook_verify_token:
        return Response(content=hub_challenge, media_type="text/plain")
    raise HTTPException(status_code=403, detail="Invalid token")

@app.post("/webhook")
async def messenger_webhook(request: Request):
    """معالج webhook لكل من Facebook Messenger وWhatsApp Cloud API"""
    try:
        body = await request.json()

        object_type = body.get("object")
        if not object_type:
            logger.warning("Webhook ignored: missing object type")
            return {"status": "ignored"}

        # فرع Messenger (object == 'page')
        if object_type == "page":
            for entry in body.get("entry", []):
                for msg_event in entry.get("messaging", []):
                    sender = msg_event.get("sender", {}).get("id")
                    message = msg_event.get("message", {})
                    if message.get("is_echo") or not sender:
                        continue

                    message_id = message.get("mid")
                    if message_id and chat_service.storage.is_message_seen(message_id):
                        continue
                    if message_id:
                        chat_service.storage.mark_message_seen(message_id)

                    text = (message.get("text") or "").strip()
                    if text:
                        asyncio.create_task(chat_service.process_message("messenger", sender, text))
            return {"status": "ok"}

        # فرع WhatsApp Cloud API (object == 'whatsapp_business_account')
        if object_type == "whatsapp_business_account":
            for entry in body.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    for msg in value.get("messages", []) or []:
                        if msg.get("type") != "text":
                            continue
                        sender = msg.get("from")  # رقم هاتف المُرسل
                        message_id = msg.get("id")
                        text = (msg.get("text", {}).get("body") or "").strip()
                        if not sender or not text:
                            continue

                        if message_id and chat_service.storage.is_message_seen(message_id):
                            continue
                        if message_id:
                            chat_service.storage.mark_message_seen(message_id)

                        asyncio.create_task(chat_service.process_message("whatsapp", sender, text))
            return {"status": "ok"}

        logger.warning(f"Webhook ignored: unsupported object '{object_type}'")
        return {"status": "ignored"}
    except Exception as e:
        logger.exception(f"خطأ في webhook: {e}")
        return {"status": "error", "message": str(e)}

# ---------------------------------------------------------------------------
# باقي الكود تم تبسيطه عبر استخدام الخدمات في services.py

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
