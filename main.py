# -*- coding: utf-8 -*-
"""
🚀 الشات بوت العربي المثالي (FastAPI + OpenRouter + Redis + Facebook Messenger/WhatsApp)
نسخة محدثة بطلباتك:
1) دالة عرض ماسنجر send_messenger_action(action: typing_on/typing_off/mark_seen)
2) تعديل send_reply لاستخدام الدالة وإظهار المؤشر
3) تحسين لوجينج make_openrouter_request (LLM OK / FAIL)
4) تحقق من body.object == "page" في /webhook
"""

import os
import json
import time
import logging
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv
import httpx

import redis
from redis.exceptions import ConnectionError as RedisConnectionError

load_dotenv()

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
# Env helpers
# ---------------------------------------------------------------------------

def get_env_var(name: str, default: Any = None, required: bool = False) -> Any:
    value = os.getenv(name, default)
    if required and not value:
        raise ValueError(f"❌ المتغير {name} مطلوب في ملف .env")
    return value

def get_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        logger.warning(f"⚠️ قيمة غير صحيحة لـ {name}, استخدام: {default}")
        return default

def get_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        logger.warning(f"⚠️ قيمة غير صحيحة لـ {name}, استخدام: {default}")
        return default

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = get_env_var("OPENROUTER_API_KEY", required=True)
OPENROUTER_MODEL = get_env_var("OPENROUTER_MODEL", "qwen/qwen3-30b-a3b:free")
LLM_TEMPERATURE = get_float_env("LLM_TEMPERATURE", 0.4)
LLM_MAX_TOKENS = get_int_env("LLM_MAX_TOKENS", 350)
SYSTEM_PROMPT = get_env_var("SYSTEM_PROMPT", "أنت مساعد ذكي وودود ومفيد. تجيب بوضوح وباللغة العربية.")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
    "HTTP-Referer": get_env_var("APP_REFERRER", "http://localhost"),
    "X-Title": get_env_var("APP_TITLE", "Arabic Chatbot via OpenRouter"),
}

FACEBOOK_VERIFY_TOKEN = get_env_var("FACEBOOK_VERIFY_TOKEN", required=True)
FACEBOOK_PAGE_ACCESS_TOKEN = get_env_var("FACEBOOK_PAGE_ACCESS_TOKEN")
WHATSAPP_TOKEN = get_env_var("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = get_env_var("WHATSAPP_PHONE_NUMBER_ID")
FB_GRAPH_VERSION = get_env_var("FB_GRAPH_VERSION", "v18.0")

ALLOWED_ORIGINS = [o.strip() for o in get_env_var("ALLOWED_ORIGINS", "http://localhost,http://localhost:3000").split(",") if o.strip()]
CONV_TTL_SECONDS = get_int_env("CONV_TTL_SECONDS", 86400)
RATE_LIMIT_MAX = get_int_env("RATE_LIMIT_MAX", 30)
RATE_LIMIT_TTL = get_int_env("RATE_LIMIT_TTL", 60)

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------
app = FastAPI(title="🤖 Arabic Chatbot API", description="الشات بوت العربي المثالي - آمن، سريع، وذكي", version="2.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ---------------------------------------------------------------------------
# Storage (Redis or in-memory)
# ---------------------------------------------------------------------------
try:
    redis_client = redis.from_url(get_env_var("REDIS_URL"), decode_responses=True)
    redis_client.ping()
    use_redis = True
    logger.info("✅ Redis متصل بنجاح")
except (RedisConnectionError, ValueError, Exception):
    logger.warning("⚠️ Redis غير متاح، استخدام الذاكرة المؤقتة")
    use_redis = False
    conversations: Dict[str, List[Dict[str, str]]] = {}


def get_conversation(conversation_id: str) -> List[Dict[str, str]]:
    if use_redis:
        if redis_client.exists(conversation_id):
            return json.loads(redis_client.get(conversation_id))
        else:
            conv = [{"role": "system", "content": SYSTEM_PROMPT}]
            redis_client.setex(conversation_id, CONV_TTL_SECONDS, json.dumps(conv))
            return conv
    else:
        if conversation_id not in conversations:
            conversations[conversation_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
        return conversations[conversation_id]


def save_conversation(conversation_id: str, messages: List[Dict[str, str]]):
    if use_redis:
        redis_client.setex(conversation_id, CONV_TTL_SECONDS, json.dumps(messages))
    else:
        conversations[conversation_id] = messages


def incr_rate(user_id: str) -> bool:
    if not use_redis:
        return True
    key = f"rate:{user_id}"
    try:
        with redis_client.pipeline() as pipe:
            pipe.incr(key)
            pipe.expire(key, RATE_LIMIT_TTL)
            count, _ = pipe.execute()
            return int(count) <= RATE_LIMIT_MAX
    except Exception:
        return True

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    user_id: Optional[str] = "anonymous"

class ChatResponse(BaseModel):
    reply: str
    conversation_id: str
    response_time: float
    message_count: int
    timestamp: str

class NewConversationRequest(BaseModel):
    user_id: str = "anonymous"

# ---------------------------------------------------------------------------
# LLM call with retries + logging
# ---------------------------------------------------------------------------
async def make_openrouter_request(messages: List[Dict[str, str]]) -> str:
    async with httpx.AsyncClient() as client:
        for attempt in range(3):
            try:
                resp = await client.post(
                    OPENROUTER_URL,
                    json={
                        "model": OPENROUTER_MODEL,
                        "messages": messages,
                        "temperature": LLM_TEMPERATURE,
                        "max_tokens": LLM_MAX_TOKENS,
                    },
                    headers=HEADERS,
                    timeout=60.0,
                )
                resp.raise_for_status()
                result = resp.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content")
                if not content:
                    raise ValueError("Empty content from model")
                # Logging المطلوب
                logger.info(f"[LLM OK] tokens≈{LLM_MAX_TOKENS}, model={OPENROUTER_MODEL}")
                logger.info(f"[LLM REPLY] chars={len(content)}")
                return content
            except Exception as e:
                wait = 1.0 * (2 ** attempt) + (0.2 * attempt)
                logger.warning(f"[LLM FAIL] attempt={attempt+1}/3 type={type(e).__name__} err={e} retry_in={wait:.1f}s")
                if attempt == 2:
                    logger.error("[LLM] giving up after 3 attempts")
                    raise HTTPException(status_code=503, detail="خدمة الذكاء غير متاحة حاليًّا")
                await asyncio.sleep(wait)

# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "🤖 مرحباً بك في الشات بوت المثالي", "status": "✅ يعمل بنجاح"}

@app.get("/health")
async def health_check(fast: bool = False):
    """Health check:
    - fast=true  → لا يتواصل مع OpenRouter (للتحقق السريع)
    - fast=false → يحاول يتأكد من اتصال OpenRouter.
    """
    if fast:
        return {"status": "ok-local", "storage": "redis" if use_redis else "memory"}

    try:
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

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if not incr_rate(req.user_id):
        raise HTTPException(status_code=429, detail="طلبات كثيرة جدًا، جرّب بعد دقيقة.")
    messages = get_conversation(req.conversation_id or "default")
    messages.append({"role": "user", "content": req.message})
    reply = await make_openrouter_request(messages)
    messages.append({"role": "assistant", "content": reply})
    save_conversation(req.conversation_id or "default", messages)
    return ChatResponse(
        reply=reply,
        conversation_id=req.conversation_id or "default",
        response_time=0.0,
        message_count=len(messages)-1,
        timestamp=datetime.now().isoformat(),
    )

@app.post("/new-conversation")
async def new_conversation(req: NewConversationRequest):
    conversation_id = f"{req.user_id}_{int(time.time())}_{os.urandom(4).hex()}"
    save_conversation(conversation_id, [{"role": "system", "content": SYSTEM_PROMPT}])
    return {"conversation_id": conversation_id, "message": "تم إنشاء محادثة جديدة"}

# ---------------------------------------------------------------------------
# Messenger Webhook
# ---------------------------------------------------------------------------
@app.get("/webhook")
async def verify_webhook(
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
):
    if hub_mode == "subscribe" and hub_verify_token == FACEBOOK_VERIFY_TOKEN:
        return Response(content=hub_challenge, media_type="text/plain")
    raise HTTPException(status_code=403, detail="Invalid token")

@app.post("/webhook")
async def messenger_webhook(request: Request):
    body = await request.json()

    # ✅ المطلوب: تجاهل أي إدخال غير متوقع
    if body.get("object") != "page":
        logger.warning("Webhook ignored: object != 'page'")
        return {"status": "ignored"}

    for entry in body.get("entry", []):
        for msg_event in entry.get("messaging", []):
            sender = msg_event.get("sender", {}).get("id")
            message = msg_event.get("message", {})
            if message.get("is_echo") or not sender:
                continue

            message_id = message.get("mid")
            if message_id and use_redis and redis_client.exists(f"seen:{message_id}"):
                continue
            if message_id and use_redis:
                redis_client.setex(f"seen:{message_id}", 3600, "1")

            text = (message.get("text") or "").strip()
            if text:
                asyncio.create_task(process_message("messenger", sender, text))
    return {"status": "ok"}

# ---------------------------------------------------------------------------
# Processing & Senders
# ---------------------------------------------------------------------------
async def process_message(platform: str, user_id: str, text: str):
    conv_id = f"{platform}_{user_id}"
    messages = get_conversation(conv_id)
    messages.append({"role": "user", "content": text})
    reply = await make_openrouter_request(messages)
    messages.append({"role": "assistant", "content": reply})
    save_conversation(conv_id, messages)
    await send_reply(platform, user_id, reply)

# ✅ المطلوب: استبدال/إضافة دالة عرض ماسنجر وإعادة بناء send_reply
async def send_messenger_action(recipient_id: str, action: str):
    url = f"https://graph.facebook.com/{FB_GRAPH_VERSION}/me/messages"
    params = {"access_token": FACEBOOK_PAGE_ACCESS_TOKEN}
    payload = {"recipient": {"id": recipient_id}, "sender_action": action}
    async with httpx.AsyncClient() as client:
        r = await client.post(url, params=params, json=payload, timeout=20.0)
        if r.status_code >= 400:
            logger.error(f"[FB action:{action}] {r.status_code} {r.text}")

async def send_reply(platform: str, recipient: str, text: str):
    try:
        if platform == "messenger":
            # اظهر جاري الكتابة + تمّ الاطلاع
            await send_messenger_action(recipient, "mark_seen")
            await send_messenger_action(recipient, "typing_on")

            url = f"https://graph.facebook.com/{FB_GRAPH_VERSION}/me/messages"
            params = {"access_token": FACEBOOK_PAGE_ACCESS_TOKEN}
            payload = {"recipient": {"id": recipient}, "message": {"text": text}}
            async with httpx.AsyncClient() as client:
                r = await client.post(url, json=payload, params=params, timeout=20.0)
                if r.status_code >= 400:
                    logger.error(f"[FB send] {r.status_code} {r.text}")

            await send_messenger_action(recipient, "typing_off")

        elif platform == "whatsapp":
            url = f"https://graph.facebook.com/{FB_GRAPH_VERSION}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
            headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
            payload = {"messaging_product": "whatsapp", "to": recipient, "text": {"body": text}}
            async with httpx.AsyncClient() as client:
                r = await client.post(url, json=payload, headers=headers, timeout=20.0)
                if r.status_code >= 400:
                    logger.error(f"[WA send] {r.status_code} {r.text}")

    except Exception as e:
        logger.exception(f"send_reply error: {e}")

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
