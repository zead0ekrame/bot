# -*- coding: utf-8 -*-
"""
خدمات التطبيق - منطق الأعمال
"""

import json
import asyncio
import logging
from typing import List, Dict, Optional
from datetime import datetime
from time import perf_counter
import threading
from pathlib import Path
import csv

import httpx
import redis
from redis.exceptions import ConnectionError as RedisConnectionError

from config import settings, OPENROUTER_URL, HEADERS
from exceptions import LLMServiceException, RateLimitException, ConversationException

logger = logging.getLogger("arabic-chatbot")

# ---------------------------------------------------------------------------
# Optional LangSmith tracing
# ---------------------------------------------------------------------------
try:
    from langsmith import traceable  # type: ignore
except Exception:
    def traceable(*_args, **_kwargs):  # fallback no-op decorator
        def _decorator(func):
            return func
        return _decorator

# ---------------------------------------------------------------------------
# Optional LangChain RAG (Redis Vector Store + OpenAI Embeddings)
# ---------------------------------------------------------------------------
try:
    from langchain_openai import OpenAIEmbeddings  # type: ignore
    from langchain_community.vectorstores import Redis as RedisVectorStore  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAIEmbeddings = None  # type: ignore
    RedisVectorStore = None  # type: ignore

# ---------------------------------------------------------------------------
# Shared HTTP client (connection pooling, keep-alive)
# ---------------------------------------------------------------------------
_shared_http_client: httpx.AsyncClient | None = None


async def init_http_client() -> None:
    global _shared_http_client
    if _shared_http_client is None:
        limits = httpx.Limits(max_keepalive_connections=100, max_connections=200)
        _shared_http_client = httpx.AsyncClient(timeout=30.0, limits=limits)


async def get_http_client() -> httpx.AsyncClient:
    if _shared_http_client is None:
        await init_http_client()
    assert _shared_http_client is not None
    return _shared_http_client


async def close_http_client() -> None:
    global _shared_http_client
    if _shared_http_client is not None:
        await _shared_http_client.aclose()
        _shared_http_client = None

# ---------------------------------------------------------------------------
# Metrics Service
# ---------------------------------------------------------------------------
class MetricsService:
    """خدمة قياس الأزمنة والمتوسطات"""

    def __init__(self):
        self._lock = threading.Lock()
        self.server_requests_total = 0
        self.server_latency_ms_sum = 0.0

        self.model_requests_total = 0
        self.model_latency_ms_sum = 0.0

        self.bot_requests_total = 0
        self.bot_latency_ms_sum = 0.0

    def record_server_latency_ms(self, duration_ms: float) -> None:
        with self._lock:
            self.server_requests_total += 1
            self.server_latency_ms_sum += float(duration_ms)

    def record_model_latency_ms(self, duration_ms: float) -> None:
        with self._lock:
            self.model_requests_total += 1
            self.model_latency_ms_sum += float(duration_ms)

    def record_bot_latency_ms(self, duration_ms: float) -> None:
        with self._lock:
            self.bot_requests_total += 1
            self.bot_latency_ms_sum += float(duration_ms)

    def get_metrics(self) -> Dict[str, float | int]:
        with self._lock:
            server_avg = (self.server_latency_ms_sum / self.server_requests_total) if self.server_requests_total else 0.0
            model_avg = (self.model_latency_ms_sum / self.model_requests_total) if self.model_requests_total else 0.0
            bot_avg = (self.bot_latency_ms_sum / self.bot_requests_total) if self.bot_requests_total else 0.0

            return {
                "server_requests_total": self.server_requests_total,
                "server_latency_ms_avg": round(server_avg, 2),
                "model_requests_total": self.model_requests_total,
                "model_latency_ms_avg": round(model_avg, 2),
                "bot_requests_total": self.bot_requests_total,
                "bot_latency_ms_avg": round(bot_avg, 2),
            }

# ---------------------------------------------------------------------------
# Storage Service
# ---------------------------------------------------------------------------
class StorageService:
    """خدمة إدارة التخزين (Redis/Memory)"""
    
    def __init__(self):
        self.use_redis = False
        self.redis_client = None
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
        self._init_storage()
    
    def _init_storage(self):
        """تهيئة التخزين"""
        try:
            self.redis_client = redis.from_url(settings.redis_url, decode_responses=True)
            self.redis_client.ping()
            self.use_redis = True
            logger.info("✅ Redis متصل بنجاح")
        except (RedisConnectionError, ValueError, Exception) as e:
            logger.warning(f"⚠️ Redis غير متاح، استخدام الذاكرة المؤقتة: {e}")
            self.use_redis = False
    
    def get_conversation(self, conversation_id: str) -> List[Dict[str, str]]:
        """الحصول على محادثة"""
        if self.use_redis:
            if self.redis_client.exists(conversation_id):
                return json.loads(self.redis_client.get(conversation_id))
            else:
                conv = [{"role": "system", "content": settings.system_prompt}]
                self.redis_client.setex(conversation_id, settings.conv_ttl_seconds, json.dumps(conv))
                return conv
        else:
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = [{"role": "system", "content": settings.system_prompt}]
            return self.conversations[conversation_id]
    
    def save_conversation(self, conversation_id: str, messages: List[Dict[str, str]]):
        """حفظ محادثة"""
        if self.use_redis:
            self.redis_client.setex(conversation_id, settings.conv_ttl_seconds, json.dumps(messages))
        else:
            self.conversations[conversation_id] = messages
    
    def check_rate_limit(self, user_id: str) -> bool:
        """فحص حد الطلبات"""
        if not self.use_redis:
            return True
        
        key = f"rate:{user_id}"
        try:
            with self.redis_client.pipeline() as pipe:
                pipe.incr(key)
                pipe.expire(key, settings.rate_limit_ttl)
                count, _ = pipe.execute()
                return int(count) <= settings.rate_limit_max
        except Exception as e:
            logger.error(f"خطأ في فحص حد الطلبات: {e}")
            return True
    
    def mark_message_seen(self, message_id: str):
        """تحديد الرسالة كمقروءة"""
        if self.use_redis and message_id:
            self.redis_client.setex(f"seen:{message_id}", 3600, "1")
    
    def is_message_seen(self, message_id: str) -> bool:
        """فحص إذا كانت الرسالة مقروءة"""
        if self.use_redis and message_id:
            return self.redis_client.exists(f"seen:{message_id}")
        return False

# ---------------------------------------------------------------------------
# LLM Service
# ---------------------------------------------------------------------------
class LLMService:
    """خدمة الذكاء الاصطناعي"""
    
    @staticmethod
    @traceable(name="llm_generate_response")
    async def generate_response(messages: List[Dict[str, str]]) -> str:
        """توليد رد من الذكاء الاصطناعي"""
        client = await get_http_client()
        for attempt in range(3):
            try:
                t0 = perf_counter()
                resp = await client.post(
                    OPENROUTER_URL,
                    json={
                        "model": settings.openrouter_model,
                        "messages": messages,
                        "temperature": settings.llm_temperature,
                        "max_tokens": settings.llm_max_tokens,
                    },
                    headers=HEADERS,
                )
                resp.raise_for_status()
                result = resp.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content")
                t1 = perf_counter()
                metrics.record_model_latency_ms((t1 - t0) * 1000.0)
                
                if not content:
                    raise ValueError("Empty content from model")
                
                # Logging محسن
                logger.info(f"[LLM OK] attempt={attempt+1}, tokens≈{settings.llm_max_tokens}, model={settings.openrouter_model}")
                logger.info(f"[LLM REPLY] chars={len(content)}")
                
                return content
                
            except Exception as e:
                wait = 1.0 * (2 ** attempt) + (0.2 * attempt)
                logger.warning(f"[LLM FAIL] attempt={attempt+1}/3 type={type(e).__name__} err={e} retry_in={wait:.1f}s")
                
                if attempt == 2:
                    logger.error("[LLM] giving up after 3 attempts")
                    if "429 Too Many Requests" in str(e):
                        raise LLMServiceException(f"OpenRouter API متجاوز حد الطلبات (خطأ 429). انتظر 5-10 دقائق أو ارفع مستوى خطتك.")
                    else:
                        raise LLMServiceException(f"فشل في الاتصال بخدمة الذكاء: {str(e)}")
                
                await asyncio.sleep(wait)

# ---------------------------------------------------------------------------
# RAG Service
# ---------------------------------------------------------------------------
class RAGService:
    """خدمة الاسترجاع المعزز بالمعرفة (RAG). تعمل اختيارياً.

    - تستخدم Redis كـ Vector Store إن توفر، وإلا تعود بلا سياق.
    - تستخدم OpenAI Embeddings إن توفر API Key.
    """

    def __init__(self):
        self.enabled: bool = bool(settings.rag_enabled)
        self.top_k: int = int(settings.rag_top_k)
        self.index_name: str = settings.rag_index_name
        self._retriever = None
        self._init_retriever_safely()

    def _init_retriever_safely(self) -> None:
        if not self.enabled:
            logger.info("RAG disabled by settings")
            return
        try:
            if not (OpenAIEmbeddings and RedisVectorStore):
                logger.warning("LangChain optional deps not available; RAG will be no-op")
                return
            if settings.embedding_provider.lower() != "openai":
                logger.warning("Only 'openai' embeddings are supported in current setup; RAG will be no-op")
                return
            if not settings.openai_api_key:
                logger.warning("OPENAI_API_KEY missing; RAG will be no-op")
                return

            embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
            vs = RedisVectorStore.from_existing_index(
                embedding=embeddings,
                index_name=self.index_name,
                redis_url=settings.redis_url,
            )
            self._retriever = vs.as_retriever(search_kwargs={"k": self.top_k})
            logger.info(f"✅ RAG retriever ready (index={self.index_name}, top_k={self.top_k})")
        except Exception as e:
            logger.warning(f"⚠️ RAG unavailable: {e}")
            self._retriever = None

    def is_ready(self) -> bool:
        return self.enabled and self._retriever is not None

    @traceable(name="rag_retrieve")
    async def retrieve(self, query: str) -> List[str]:
        if not self.is_ready():
            return []
        try:
            loop = asyncio.get_running_loop()
            docs = await loop.run_in_executor(None, self._retriever.get_relevant_documents, query)  # type: ignore
            return [d.page_content for d in (docs or []) if getattr(d, "page_content", None)]
        except Exception as e:
            logger.warning(f"RAG retrieve failed: {e}")
            return []

    def build_context_message(self, docs: List[str]) -> Optional[Dict[str, str]]:
        if not docs:
            return None
        joined = "\n\n---\n\n".join(d.strip()[:1200] for d in docs[: self.top_k])
        content = (
            "معلومات سياقية من قاعدة المعرفة الداخلية. استخدمها للإجابة بدقة، ولا تخترع معلومات.\n\n"
            f"{joined}"
        )
        return {"role": "system", "content": content}

# ---------------------------------------------------------------------------
# Platform Service
# ---------------------------------------------------------------------------
class PlatformService:
    """خدمة منصات التواصل"""
    
    @staticmethod
    @traceable(name="fb_send_action")
    async def send_messenger_action(recipient_id: str, action: str):
        """إرسال إجراء لماسنجر"""
        if not settings.facebook_page_access_token:
            logger.warning("Facebook Page Access Token غير متوفر")
            return
        
        url = f"https://graph.facebook.com/{settings.fb_graph_version}/me/messages"
        params = {"access_token": settings.facebook_page_access_token}
        payload = {"recipient": {"id": recipient_id}, "sender_action": action}
        
        try:
            client = await get_http_client()
            r = await client.post(url, params=params, json=payload, timeout=20.0)
            if r.status_code >= 400:
                logger.error(f"[FB action:{action}] {r.status_code} {r.text}")
        except Exception as e:
            logger.error(f"خطأ في إرسال إجراء ماسنجر: {e}")
    
    @staticmethod
    @traceable(name="fb_send_message")
    async def send_messenger_message(recipient: str, text: str):
        """إرسال رسالة لماسنجر"""
        if not settings.facebook_page_access_token:
            raise ConversationException("Facebook Page Access Token غير متوفر")
        
        url = f"https://graph.facebook.com/{settings.fb_graph_version}/me/messages"
        params = {"access_token": settings.facebook_page_access_token}
        payload = {"recipient": {"id": recipient}, "message": {"text": text}}
        
        try:
            client = await get_http_client()
            r = await client.post(url, json=payload, params=params, timeout=20.0)
            if r.status_code >= 400:
                logger.error(f"[FB send] {r.status_code} {r.text}")
                raise ConversationException(f"فشل في إرسال الرسالة: {r.text}")
        except Exception as e:
            logger.error(f"خطأ في إرسال رسالة ماسنجر: {e}")
            raise ConversationException(f"خطأ في إرسال الرسالة: {str(e)}")
    
    @staticmethod
    @traceable(name="wa_send_message")
    async def send_whatsapp_message(recipient: str, text: str):
        """إرسال رسالة لواتساب"""
        if not settings.whatsapp_token or not settings.whatsapp_phone_number_id:
            raise ConversationException("WhatsApp credentials غير متوفرة")
        
        url = f"https://graph.facebook.com/{settings.fb_graph_version}/{settings.whatsapp_phone_number_id}/messages"
        headers = {"Authorization": f"Bearer {settings.whatsapp_token}", "Content-Type": "application/json"}
        payload = {"messaging_product": "whatsapp", "to": recipient, "text": {"body": text}}
        
        try:
            client = await get_http_client()
            r = await client.post(url, json=payload, headers=headers, timeout=20.0)
            if r.status_code >= 400:
                logger.error(f"[WA send] {r.status_code} {r.text}")
                raise ConversationException(f"فشل في إرسال الرسالة: {r.text}")
        except Exception as e:
            logger.error(f"خطأ في إرسال رسالة واتساب: {e}")
            raise ConversationException(f"خطأ في إرسال الرسالة: {str(e)}")

# ---------------------------------------------------------------------------
# Chat Service
# ---------------------------------------------------------------------------
class ChatService:
    """خدمة المحادثة الرئيسية"""
    
    def __init__(self):
        self.storage = StorageService()
        self.llm = LLMService()
        self.platform = PlatformService()
        self.rag = RAGService()
    
    async def process_message(self, platform: str, user_id: str, text: str):
        """معالجة رسالة من المستخدم"""
        t0 = perf_counter()
        
        # فحص حد الطلبات
        if not self.storage.check_rate_limit(user_id):
            raise RateLimitException()
        
        # الحصول على المحادثة
        conv_id = f"{platform}_{user_id}"
        messages = self.storage.get_conversation(conv_id)
        
        # جمع بيانات الرسالة الواردة
        incoming_message_data = {
            'message_id': f"{user_id}_{int(t0)}",
            'conversation_id': conv_id,
            'user_id': user_id,
            'platform': platform,
            'timestamp': datetime.now().isoformat(),
            'message_type': 'text',
            'content': text,
            'direction': 'in',
            'response_time_ms': 0,
            'tokens_used': 0,
            'sentiment_score': 0,
            'keywords': self._extract_keywords(text),
            'user_engagement': 'medium'
        }
        
        # حفظ بيانات الرسالة الواردة
        chat_data_collector.collect_message_data(incoming_message_data)
        
        # إضافة رسالة المستخدم
        messages.append({"role": "user", "content": text})
        
        # دمج سياق RAG (اختياري)
        messages_for_llm = list(messages)
        try:
            if self.rag.is_ready():
                docs = await self.rag.retrieve(text)
                ctx_msg = self.rag.build_context_message(docs)
                if ctx_msg:
                    messages_for_llm.append(ctx_msg)
        except Exception as e:
            logger.warning(f"RAG context skipped due to error: {e}")

        # توليد الرد
        reply = await self.llm.generate_response(messages_for_llm)
        
        # إضافة رد المساعد
        messages.append({"role": "assistant", "content": reply})
        
        # حفظ المحادثة
        self.storage.save_conversation(conv_id, messages)
        
        # جمع بيانات الرد
        outgoing_message_data = {
            'message_id': f"bot_{user_id}_{int(t0)}",
            'conversation_id': conv_id,
            'user_id': user_id,
            'platform': platform,
            'timestamp': datetime.now().isoformat(),
            'message_type': 'text',
            'content': reply,
            'direction': 'out',
            'response_time_ms': int((perf_counter() - t0) * 1000),
            'tokens_used': len(reply.split()),  # تقدير بسيط
            'sentiment_score': 0,
            'keywords': self._extract_keywords(reply),
            'user_engagement': 'medium'
        }
        
        # حفظ بيانات الرد
        chat_data_collector.collect_message_data(outgoing_message_data)
        
        # تحديث ملف المستخدم
        self._update_user_profile(platform, user_id, text, reply)
        
        # إرسال الرد
        await self.send_reply(platform, user_id, reply)
        t1 = perf_counter()
        metrics.record_bot_latency_ms((t1 - t0) * 1000.0)
    
    async def send_reply(self, platform: str, recipient: str, text: str):
        """إرسال رد للمستخدم"""
        try:
            if platform == "messenger":
                # إظهار مؤشرات الكتابة
                await self.platform.send_messenger_action(recipient, "mark_seen")
                await self.platform.send_messenger_action(recipient, "typing_on")
                
                # إرسال الرسالة
                await self.platform.send_messenger_message(recipient, text)
                
                # إيقاف مؤشر الكتابة
                await self.platform.send_messenger_action(recipient, "typing_off")
                
            elif platform == "whatsapp":
                await self.platform.send_whatsapp_message(recipient, text)
                
        except Exception as e:
            logger.exception(f"خطأ في إرسال الرد: {e}")
            raise ConversationException(f"فشل في إرسال الرد: {str(e)}")
    
    def create_new_conversation(self, user_id: str) -> str:
        """إنشاء محادثة جديدة"""
        import time
        import os
        
        conversation_id = f"{user_id}_{int(time.time())}_{os.urandom(4).hex()}"
        self.storage.save_conversation(conversation_id, [{"role": "system", "content": settings.system_prompt}])
        return conversation_id

    def _extract_keywords(self, text: str) -> str:
        """استخراج كلمات مفتاحية بسيطة"""
        # كلمات مفتاحية للمنتجات الزراعية
        keywords = []
        text_lower = text.lower()
        
        product_keywords = ['بذور', 'سماد', 'تربة', 'أصص', 'أدوات', 'مبيد', 'طماطم', 'خيار', 'فلفل']
        for keyword in product_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
        
        return ', '.join(keywords) if keywords else ''
    
    def _update_user_profile(self, platform: str, user_id: str, user_message: str, bot_reply: str):
        """تحديث ملف المستخدم"""
        try:
            # حساب إحصائيات بسيطة
            message_length = len(user_message)
            
            # تحديد نية المستخدم
            intent = self._determine_user_intent(user_message)
            
            # تحديث ملف المستخدم
            user_data = {
                'user_id': user_id,
                'platform': platform,
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
                'total_conversations': 1,
                'total_messages': 1,
                'avg_message_length': message_length,
                'preferred_topics': intent,
                'conversion_history': 'new_user',
                'customer_segment': 'prospect',
                'notes': f'أول رسالة: {user_message[:50]}...'
            }
            
            chat_data_collector.update_user_profile(user_data)
            
        except Exception as e:
            logger.error(f"خطأ في تحديث ملف المستخدم: {e}")
    
    def _determine_user_intent(self, message: str) -> str:
        """تحديد نية المستخدم"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['أريد شراء', 'كم السعر', 'متوفر']):
            return 'شراء'
        elif any(word in message_lower for word in ['كيف', 'متى', 'أين', 'استفسار']):
            return 'استفسار'
        elif any(word in message_lower for word in ['مشكلة', 'خطأ', 'شكوى']):
            return 'شكوى'
        elif any(word in message_lower for word in ['متابعة', 'أين طلبي']):
            return 'متابعة'
        elif any(word in message_lower for word in ['شكراً', 'ممتاز', 'رائع']):
            return 'شكر'
        else:
            return 'عام'

# ---------------------------------------------------------------------------
# Chat Data Collection Service
# ---------------------------------------------------------------------------
class ChatDataCollectionService:
    """خدمة جمع بيانات الدردشة المتقدمة"""
    
    def __init__(self):
        self.data_dir = Path("chat_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # ملفات البيانات
        self.conversations_file = self.data_dir / "conversations.csv"
        self.messages_file = self.data_dir / "messages.csv"
        self.analytics_file = self.data_dir / "analytics.csv"
        self.user_profiles_file = self.data_dir / "user_profiles.csv"
        
        # تهيئة الملفات
        self._init_data_files()
    
    def _init_data_files(self):
        """تهيئة ملفات CSV لجمع البيانات"""
        
        # ملف المحادثات
        if not self.conversations_file.exists():
            with open(self.conversations_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'conversation_id',
                    'user_id', 
                    'platform',
                    'start_time',
                    'last_message_time',
                    'message_count',
                    'total_chars',
                    'status',
                    'user_intent',
                    'conversion_rate',
                    'notes'
                ])
        
        # ملف الرسائل التفصيلي
        if not self.messages_file.exists():
            with open(self.messages_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'message_id',
                    'conversation_id',
                    'user_id',
                    'platform',
                    'timestamp',
                    'message_type',
                    'content',
                    'direction',
                    'response_time_ms',
                    'tokens_used',
                    'sentiment_score',
                    'keywords',
                    'user_engagement'
                ])
        
        # ملف تحليلات المستخدمين
        if not self.user_profiles_file.exists():
            with open(self.user_profiles_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'user_id',
                    'platform',
                    'first_seen',
                    'last_seen',
                    'total_conversations',
                    'total_messages',
                    'avg_message_length',
                    'preferred_topics',
                    'conversion_history',
                    'customer_segment',
                    'notes'
                ])
        
        # ملف التحليلات اليومية
        if not self.analytics_file.exists():
            with open(self.analytics_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'date',
                    'total_conversations',
                    'total_messages',
                    'unique_users',
                    'avg_response_time',
                    'avg_message_length',
                    'platform_breakdown',
                    'top_intents',
                    'conversion_events',
                    'customer_satisfaction'
                ])
    
    def collect_message_data(self, message_data: Dict):
        """جمع بيانات رسالة جديدة"""
        try:
            with open(self.messages_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    message_data.get('message_id', ''),
                    message_data.get('conversation_id', ''),
                    message_data.get('user_id', ''),
                    message_data.get('platform', ''),
                    message_data.get('timestamp', ''),
                    message_data.get('message_type', 'text'),
                    message_data.get('content', ''),
                    message_data.get('direction', ''),
                    message_data.get('response_time_ms', 0),
                    message_data.get('tokens_used', 0),
                    message_data.get('sentiment_score', 0),
                    message_data.get('keywords', ''),
                    message_data.get('user_engagement', 'medium')
                ])
            
            logger.info(f"تم جمع بيانات رسالة: {message_data.get('message_id', 'unknown')}")
            
        except Exception as e:
            logger.error(f"خطأ في جمع بيانات الرسالة: {e}")
    
    def collect_conversation_data(self, conversation_data: Dict):
        """جمع بيانات محادثة جديدة"""
        try:
            with open(self.conversations_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    conversation_data.get('conversation_id', ''),
                    conversation_data.get('user_id', ''),
                    conversation_data.get('platform', ''),
                    conversation_data.get('start_time', ''),
                    conversation_data.get('last_message_time', ''),
                    conversation_data.get('message_count', 0),
                    conversation_data.get('total_chars', 0),
                    conversation_data.get('status', 'active'),
                    conversation_data.get('user_intent', ''),
                    conversation_data.get('conversion_rate', 0),
                    conversation_data.get('notes', '')
                ])
            
            logger.info(f"تم جمع بيانات محادثة: {conversation_data.get('conversation_id', 'unknown')}")
            
        except Exception as e:
            logger.error(f"خطأ في جمع بيانات المحادثة: {e}")
    
    def update_user_profile(self, user_data: Dict):
        """تحديث ملف المستخدم"""
        try:
            # قراءة الملف الحالي
            profiles = []
            if self.user_profiles_file.exists():
                with open(self.user_profiles_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    profiles = list(reader)
            
            # البحث عن المستخدم
            user_found = False
            for profile in profiles:
                if profile['user_id'] == user_data['user_id'] and profile['platform'] == user_data['platform']:
                    # تحديث البيانات
                    for key, value in user_data.items():
                        if key in profile:
                            profile[key] = value
                    user_found = True
                    break
            
            # إضافة مستخدم جديد إذا لم يوجد
            if not user_found:
                profiles.append(user_data)
            
            # إعادة كتابة الملف
            with open(self.user_profiles_file, 'w', newline='', encoding='utf-8') as f:
                if profiles:
                    writer = csv.DictWriter(f, fieldnames=profiles[0].keys())
                    writer.writeheader()
                    writer.writerows(profiles)
            
            logger.info(f"تم تحديث ملف المستخدم: {user_data.get('user_id', 'unknown')}")
            
        except Exception as e:
            logger.error(f"خطأ في تحديث ملف المستخدم: {e}")
    
    def generate_daily_analytics(self, date: str = None):
        """إنشاء تحليلات يومية"""
        if not date:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # قراءة رسائل اليوم
            daily_messages = []
            with open(self.messages_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['timestamp'].startswith(date):
                        daily_messages.append(row)
            
            if not daily_messages:
                logger.info(f"لا توجد رسائل لليوم {date}")
                return
            
            # حساب الإحصائيات
            total_messages = len(daily_messages)
            unique_users = len(set(m['user_id'] for m in daily_messages))
            unique_conversations = len(set(m['conversation_id'] for m in daily_messages))
            
            # متوسط طول الرسائل
            message_lengths = [len(m.get('content', '')) for m in daily_messages]
            avg_message_length = sum(message_lengths) / len(message_lengths) if message_lengths else 0
            
            # متوسط وقت الاستجابة
            response_times = [float(m.get('response_time_ms', 0)) for m in daily_messages if m.get('response_time_ms')]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            # توزيع المنصات
            platforms = {}
            for msg in daily_messages:
                platform = msg.get('platform', 'unknown')
                platforms[platform] = platforms.get(platform, 0) + 1
            
            platform_breakdown = json.dumps(platforms, ensure_ascii=False)
            
            # تحليل النوايا (بسيط)
            intents = self._analyze_user_intents(daily_messages)
            top_intents = json.dumps(intents[:5], ensure_ascii=False)
            
            # حفظ التحليلات
            with open(self.analytics_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    date,
                    unique_conversations,
                    total_messages,
                    unique_users,
                    round(avg_response_time, 2),
                    round(avg_message_length, 2),
                    platform_breakdown,
                    top_intents,
                    0,  # conversion_events
                    0   # customer_satisfaction
                ])
            
            logger.info(f"تم إنشاء تحليلات يومية لليوم {date}")
            
            # طباعة ملخص
            self._print_daily_summary(date, {
                'total_messages': total_messages,
                'unique_users': unique_users,
                'unique_conversations': unique_conversations,
                'avg_response_time': avg_response_time,
                'avg_message_length': avg_message_length,
                'platforms': platforms,
                'top_intents': intents[:5]
            })
            
        except Exception as e:
            logger.error(f"خطأ في إنشاء التحليلات اليومية: {e}")
    
    def _analyze_user_intents(self, messages: List[Dict]) -> List[tuple]:
        """تحليل نوايا المستخدمين من الرسائل"""
        intent_keywords = {
            'شراء': ['أريد شراء', 'كم السعر', 'متوفر', 'طلب', 'شراء'],
            'استفسار': ['كيف', 'متى', 'أين', 'لماذا', 'استفسار', 'سؤال'],
            'شكوى': ['مشكلة', 'خطأ', 'سيء', 'رديء', 'شكوى', 'استياء'],
            'متابعة': ['متابعة', 'أين طلبي', 'متى يصل', 'تتبع'],
            'شكر': ['شكراً', 'ممتاز', 'رائع', 'أحسنت', 'شكر']
        }
        
        intent_counts = {}
        for msg in messages:
            if msg.get('direction') == 'in':  # رسائل المستخدم فقط
                content = msg.get('content', '').lower()
                for intent, keywords in intent_keywords.items():
                    for keyword in keywords:
                        if keyword in content:
                            intent_counts[intent] = intent_counts.get(intent, 0) + 1
                            break
        
        # ترتيب حسب التكرار
        return sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)
    
    def _print_daily_summary(self, date: str, stats: Dict):
        """طباعة ملخص يومي"""
        print(f"\n التقرير اليومي - {date}")
        print(f"💬 إجمالي الرسائل: {stats['total_messages']}")
        print(f"👥 المستخدمين الفريدين: {stats['unique_users']}")
        print(f"️ المحادثات الفريدة: {stats['unique_conversations']}")
        print(f"⏱️ متوسط وقت الاستجابة: {round(stats['avg_response_time'], 2)} مللي ثانية")
        print(f" متوسط طول الرسالة: {round(stats['avg_message_length'], 2)} حرف")
        print(f"📱 توزيع المنصات: {stats['platforms']}")
        print(f" أهم النوايا: {dict(stats['top_intents'])}")
    
    def export_conversation_to_json(self, conversation_id: str, output_file: str = None):
        """تصدير محادثة إلى JSON"""
        try:
            if not output_file:
                output_file = f"conversation_{conversation_id}.json"
            
            # قراءة الرسائل
            messages = []
            with open(self.messages_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['conversation_id'] == conversation_id:
                        messages.append(row)
            
            if not messages:
                logger.warning(f"لا توجد رسائل للمحادثة {conversation_id}")
                return
            
            # تنظيم البيانات
            conversation_data = {
                'conversation_id': conversation_id,
                'user_id': messages[0]['user_id'],
                'platform': messages[0]['platform'],
                'start_time': min(m['timestamp'] for m in messages),
                'end_time': max(m['timestamp'] for m in messages),
                'message_count': len(messages),
                'messages': messages
            }
            
            # حفظ كملف JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"تم تصدير المحادثة إلى {output_file}")
            
        except Exception as e:
            logger.error(f"خطأ في تصدير المحادثة: {e}")

# إنشاء instance من خدمة جمع البيانات
chat_data_collector = ChatDataCollectionService()

# إنشاء instance واحد من الخدمة
chat_service = ChatService()
metrics = MetricsService()

