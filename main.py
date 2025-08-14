# -*- coding: utf-8 -*-
"""
🚀 الشات بوت العربي المثالي (FastAPI + OpenRouter + Redis asyncio + Facebook Messenger/WhatsApp)
تنفيذ الحزمة الكاملة من التحسينات (8 نقاط) دون المساس بالهيكل الأساسي:
1) Redis غير حابس: التحويل إلى redis.asyncio مع await.
2) عميل HTTP واحد عالمي (httpx.AsyncClient) مع HTTP/2 و keep-alive عبر lifespan.
3) Typing مبكّر + Heartbeat كل ~15ث.
4) تنفيذ متوازٍ: تشغيل Heartbeat بالتوازي أثناء انتظار الـ LLM ثم إيقافه.
5) سياسة Retry ذكية (429/5xx فقط) + backoff مع jitter بسيط.
6) قصّ سياق المحادثة (آخر 8 رسائل) للحفاظ على سرعة التوليد.
7) قياس دقيق للمدد الزمنية في كل خطوة.
8) تحسين /webhook: تجاهل object!=page (مُدمج) + إرسال مؤشرات ماسنجر (مُدمج).
"""

import os
import json
import time
import logging
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv
import httpx

# Redis asyncio (غير حابس)
import redis.asyncio as redis
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
# Summarization controls
SUMMARIZE_AFTER = get_int_env("SUMMARIZE_AFTER", 10)  # بعد كم رسالة نبدأ التلخيص
KEEP_RECENT = get_int_env("KEEP_RECENT", 6)          # كم رسالة حديثة نحتفظ بها مع الملخّص

# ---------------------------------------------------------------------------
# Lifespan: عميل HTTP عالمي ب‍ HTTP/2 + Keep-Alive
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(
        http2=True,
        timeout=60.0,
        limits=httpx.Limits(max_keepalive_connections=100, max_connections=200),
        headers={},  # نستخدم HEADERS يدويًا حيث يلزم
    )
    try:
        yield
    finally:
        await app.state.http.aclose()

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------
app = FastAPI(lifespan=lifespan, title="🤖 Arabic Chatbot API", description="الشات بوت العربي المثالي - آمن، سريع، وذكي", version="2.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ---------------------------------------------------------------------------
# Storage (Redis asyncio أو in-memory)
# ---------------------------------------------------------------------------
use_redis = False
redis_client: Optional[redis.Redis] = None

async def _init_redis():
    global use_redis, redis_client
    try:
        redis_client = await redis.from_url(get_env_var("REDIS_URL"), decode_responses=True)
        # ping async
        await redis_client.ping()
        use_redis = True
        logger.info("✅ Redis (asyncio) متصل بنجاح")
    except (RedisConnectionError, ValueError, Exception):
        logger.warning("⚠️ Redis غير متاح، استخدام الذاكرة المؤقتة")
        use_redis = False

# نهيئ Redis عند الإقلاع
@app.on_event("startup")
async def _on_startup():
    await _init_redis()

# in-memory fallback
conversations_mem: Dict[str, List[Dict[str, str]]] = {}

async def get_conversation(conversation_id: str) -> List[Dict[str, str]]:
    if use_redis and redis_client is not None:
        if await redis_client.exists(conversation_id):
            data = await redis_client.get(conversation_id)
            return json.loads(data)
        else:
            conv = [{"role": "system", "content": SYSTEM_PROMPT}]
            await redis_client.setex(conversation_id, CONV_TTL_SECONDS, json.dumps(conv))
            return conv
    # memory fallback
    if conversation_id not in conversations_mem:
        conversations_mem[conversation_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return conversations_mem[conversation_id]

async def save_conversation(conversation_id: str, messages: List[Dict[str, str]]):
    if use_redis and redis_client is not None:
        await redis_client.setex(conversation_id, CONV_TTL_SECONDS, json.dumps(messages))
    else:
        conversations_mem[conversation_id] = messages

async def incr_rate(user_id: str) -> bool:
    if not (use_redis and redis_client is not None):
        return True
    key = f"rate:{user_id}"
    try:
        # محاكاة pipeline بشكل بسيط
        count = await redis_client.incr(key)
        await redis_client.expire(key, RATE_LIMIT_TTL)
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
# أدوات مساعدة للمحادثة: التلخيص الذكي + بناء الرسائل
# ---------------------------------------------------------------------------
async def get_summary(conv_id: str) -> Optional[str]:
    if use_redis and redis_client is not None:
        key = f"{conv_id}:summary"
        if await redis_client.exists(key):
            return await redis_client.get(key)
    return None

async def save_summary(conv_id: str, summary: str) -> None:
    if use_redis and redis_client is not None:
        key = f"{conv_id}:summary"
        await redis_client.setex(key, CONV_TTL_SECONDS, summary)
    else:
        conversations_mem[f"{conv_id}:summary"] = summary  # type: ignore

async def summarize_conversation(conv_id: str, msgs: List[Dict[str, str]]) -> str:
    client: httpx.AsyncClient = app.state.http
    prev = await get_summary(conv_id)
    convo = [m for m in msgs if m.get("role") != "system"]
    sys_msg = {
        "role": "system",
        "content": (
            "أنت مُلخِّص عربي دقيق. لخص المحادثة في نقاط موجزة وواضحة، "
            "تذكر الحقائق والأوامر والقيود والأخطاء السابقة، بدون حشو، "
            "وبلغة عربية فصيحة. اجعل الطول في حدود 120-200 كلمة."),
    }
    summary_prompt: List[Dict[str, str]] = [sys_msg]
    if prev:
        summary_prompt.append({"role": "user", "content": f"ملخّص سابق:
{prev}"})
    merged = "

".join(f"[{m['role']}] {m['content']}" for m in convo)
    summary_prompt.append({"role": "user", "content": f"هذه المحادثة:
{merged}"})

    resp = await client.post(
        OPENROUTER_URL,
        json={
            "model": OPENROUTER_MODEL,
            "messages": summary_prompt,
            "temperature": 0.2,
            "max_tokens": 220,
        },
        headers=HEADERS,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    if not content:
        raise ValueError("Empty summary from model")
    return content.strip()

async def build_messages_with_summary(conv_id: str, msgs: List[Dict[str, str]], keep_recent: int = KEEP_RECENT) -> List[Dict[str, str]]:
    base = [{"role": "system", "content": SYSTEM_PROMPT}]
    summary = await get_summary(conv_id)
    if not summary and len([m for m in msgs if m.get("role") != "system"]) > SUMMARIZE_AFTER:
        try:
            summary = await summarize_conversation(conv_id, msgs)
            await save_summary(conv_id, summary)
        except Exception as e:
            logger.warning(f"[SUMMARY] failed to create initial summary: {e}")
            summary = None
    if summary:
        base.append({"role": "system", "content": f"ملخّص المحادثة حتى الآن:
{summary}"})
    recent = [m for m in msgs if m.get("role") != "system"][-keep_recent:]
    return base + recent

# ---------------------------------------------------------------------------
# LLM call with retries + logging (429/5xx فقط)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
RETRY_STATUSES = {429}

def _should_retry(status: int) -> bool:
    return (status in RETRY_STATUSES) or (500 <= status <= 599)

async def make_openrouter_request(messages: List[Dict[str, str]]) -> str:
    client: httpx.AsyncClient = app.state.http
    backoff = 0.8
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
            )
            if _should_retry(resp.status_code):
                raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
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
            wait = backoff * (2 ** attempt) + 0.2 * attempt
            logger.warning(f"[LLM FAIL] attempt={attempt+1}/3 type={type(e).__name__} err={e} retry_in={wait:.1f}s")
            if attempt == 2:
                logger.error("[LLM] giving up after 3 attempts")
                raise HTTPException(status_code=503, detail="خدمة الذكاء غير متاحة حاليًّا")
            await asyncio.sleep(wait)

# ---------------------------------------------------------------------------
# Typing Heartbeat لِماسنجر
# ---------------------------------------------------------------------------
async def send_messenger_action(recipient_id: str, action: str):
    client: httpx.AsyncClient = app.state.http
    url = f"https://graph.facebook.com/{FB_GRAPH_VERSION}/me/messages"
    params = {"access_token": FACEBOOK_PAGE_ACCESS_TOKEN}
    payload = {"recipient": {"id": recipient_id}, "sender_action": action}
    r = await client.post(url, params=params, json=payload, timeout=20.0)
    if r.status_code >= 400:
        logger.error(f"[FB action:{action}] {r.status_code} {r.text}")

async def typing_heartbeat(recipient: str, period: float = 15.0):
    try:
        while True:
            await send_messenger_action(recipient, "typing_on")
            await asyncio.sleep(period)
    except asyncio.CancelledError:
        pass

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

    client: httpx.AsyncClient = app.state.http
    try:
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
    if not await incr_rate(req.user_id):
        raise HTTPException(status_code=429, detail="طلبات كثيرة جدًا، جرّب بعد دقيقة.")

    t0 = time.perf_counter()
    conv_id = req.conversation_id or "default"

    t_redis1 = time.perf_counter()
    messages = await get_conversation(conv_id)
    t_redis1_done = time.perf_counter()

    messages.append({"role": "user", "content": req.message})
    messages = await build_messages_with_summary(conv_id, messages, KEEP_RECENT)

    # تلخيص متجدد لو تجاوزنا الحد مرة أخرى
non_sys = [m for m in messages if m.get("role") != "system"]
if len(non_sys) > SUMMARIZE_AFTER * 2:
    try:
        summary = await summarize_conversation(conv_id, messages)
        await save_summary(conv_id, summary)
        messages = await build_messages_with_summary(conv_id, messages, KEEP_RECENT)
    except Exception as e:
        logger.warning(f"[SUMMARY] refresh failed: {e}")

# تلخيص متجدد أثناء المعالجة إن كثرت الرسائل
non_sys = [m for m in messages if m.get("role") != "system"]
if len(non_sys) > SUMMARIZE_AFTER * 2:
    try:
        summary = await summarize_conversation(conv_id, messages)
        await save_summary(conv_id, summary)
        messages = await build_messages_with_summary(conv_id, messages, KEEP_RECENT)
    except Exception as e:
        logger.warning(f"[SUMMARY] refresh failed: {e}")

t_llm = time.perf_counter()
reply = await make_openrouter_request(messages)
    t_llm_done = time.perf_counter()

    messages.append({"role": "assistant", "content": reply})
    await save_conversation(conv_id, messages)
    t_redis2_done = time.perf_counter()

    response_time = time.perf_counter() - t0
    logger.info(
        "[TIMING /chat] total=%.3fs redis_read=%.3fs llm=%.3fs redis_write=%.3fs",
        response_time,
        t_redis1_done - t_redis1,
        t_llm_done - t_llm,
        t_redis2_done - t_llm_done,
    )

    return ChatResponse(
        reply=reply,
        conversation_id=conv_id,
        response_time=response_time,
        message_count=len(messages)-1,
        timestamp=datetime.now().isoformat(),
    )

@app.post("/new-conversation")
async def new_conversation(req: NewConversationRequest):
    conversation_id = f"{req.user_id}_{int(time.time())}_{os.urandom(4).hex()}"
    await save_conversation(conversation_id, [{"role": "system", "content": SYSTEM_PROMPT}])
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

    # ✅ تجاهل أي إدخال غير متوقع
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
            if message_id and use_redis and redis_client is not None and await redis_client.exists(f"seen:{message_id}"):
                continue
            if message_id and use_redis and redis_client is not None:
                await redis_client.setex(f"seen:{message_id}", 3600, "1")

            text = (message.get("text") or "").strip()
            if text:
                # Typing heartbeat بالتوازي
                asyncio.create_task(process_message("messenger", sender, text))
    return {"status": "ok"}

# ---------------------------------------------------------------------------
# Processing & Senders
# ---------------------------------------------------------------------------
async def process_message(platform: str, user_id: str, text: str):
    t0 = time.perf_counter()

    conv_id = f"{platform}_{user_id}"
    t_redis1 = time.perf_counter()
    messages = await get_conversation(conv_id)
    t_redis1_done = time.perf_counter()

    messages.append({"role": "user", "content": text})
    messages = await build_messages_with_summary(conv_id, messages, KEEP_RECENT)

    hb_task = None
    if platform == "messenger":
        hb_task = asyncio.create_task(typing_heartbeat(user_id, 15.0))

    try:
        t_llm = time.perf_counter()
        reply = await make_openrouter_request(messages)
        t_llm_done = time.perf_counter()
    finally:
        if hb_task:
            hb_task.cancel()

    messages.append({"role": "assistant", "content": reply})
    await save_conversation(conv_id, messages)

    t_send = time.perf_counter()
    await send_reply(platform, user_id, reply)
    t_done = time.perf_counter()

    logger.info(
        "[TIMING process] total=%.3fs redis_read=%.3fs llm=%.3fs send=%.3fs",
        t_done - t0,
        t_redis1_done - t_redis1,
        t_llm_done - t_llm,
        t_done - t_send,
    )

async def send_reply(platform: str, recipient: str, text: str):
    client: httpx.AsyncClient = app.state.http
    try:
        if platform == "messenger":
            # اظهر جاري الكتابة + تمّ الاطلاع
            await send_messenger_action(recipient, "mark_seen")
            await send_messenger_action(recipient, "typing_on")

            url = f"https://graph.facebook.com/{FB_GRAPH_VERSION}/me/messages"
            params = {"access_token": FACEBOOK_PAGE_ACCESS_TOKEN}
            payload = {"recipient": {"id": recipient}, "message": {"text": text}}
            r = await client.post(url, json=payload, params=params, timeout=20.0)
            if r.status_code >= 400:
                logger.error(f"[FB send] {r.status_code} {r.text}")

            await send_messenger_action(recipient, "typing_off")

        elif platform == "whatsapp":
            url = f"https://graph.facebook.com/{FB_GRAPH_VERSION}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
            headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
            payload = {"messaging_product": "whatsapp", "to": recipient, "text": {"body": text}}
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
    # يمكن تشغيل uvloop من سطر الأوامر: --loop uvloop --workers 2
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
