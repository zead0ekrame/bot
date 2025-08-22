# -*- coding: utf-8 -*-
"""
إعدادات التطبيق - ملف منفصل للتكوين
"""

import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """إعدادات التطبيق الرئيسية"""
    
    # OpenRouter Configuration
    openrouter_api_key: str = Field(..., env="OPENROUTER_API_KEY")
    openrouter_model: str = Field("qwen/qwen3-30b-a3b:free", env="OPENROUTER_MODEL")
    llm_temperature: float = Field(0.4, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(350, env="LLM_MAX_TOKENS")
    system_prompt: str = Field(
        "أنت موظف مبيعات وخدمة عملاء محترف لصفحة منتجات زراعية.\n"
        "- رحّب بالعميل بأدب، واسأله عن هدفه (شراء/استفسار/متابعة طلب).\n"
        "- اجمع معلومات أساسية عند الحاجة: المنتج/الكمية/الموقع/وسيلة التواصل/موعد التسليم.\n"
        "- قدّم توصيات بديلة عند عدم توفر الصنف، واذكر العروض السارية إن وُجدت.\n"
        "- التزم بالدقة، وإن لم تتوفر معلومة فاطلب الإذن للتحقق وتابع لاحقاً.\n"
        "- اختصر بدون إخلال، وبلهجة عربية مهذبة وواضحة.",
        env="SYSTEM_PROMPT"
    )
    
    # Facebook/Messenger Configuration
    facebook_verify_token: str = Field(..., env="FACEBOOK_VERIFY_TOKEN")
    facebook_page_access_token: str = Field("", env="FACEBOOK_PAGE_ACCESS_TOKEN")
    fb_graph_version: str = Field("v18.0", env="FB_GRAPH_VERSION")
    
    # WhatsApp Configuration
    whatsapp_token: str = Field("", env="WHATSAPP_TOKEN")
    whatsapp_phone_number_id: str = Field("", env="WHATSAPP_PHONE_NUMBER_ID")
    
    # App Configuration
    app_referrer: str = Field("http://localhost", env="APP_REFERRER")
    app_title: str = Field("Arabic Chatbot via OpenRouter", env="APP_TITLE")
    
    # CORS & Security
    allowed_origins: List[str] = Field(
        default=["http://localhost", "http://localhost:3000"]
    )
    
    # Redis Configuration
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    
    # Rate Limiting
    rate_limit_max: int = Field(30, env="RATE_LIMIT_MAX")
    rate_limit_ttl: int = Field(60, env="RATE_LIMIT_TTL")
    
    # Conversation Management
    conv_ttl_seconds: int = Field(86400, env="CONV_TTL_SECONDS")
    
    # Server Configuration
    port: int = Field(8080, env="PORT")
    host: str = Field("0.0.0.0", env="HOST")

    # Modes (multi-bot)
    mode_name: str = Field("retail_agri", env="MODE_NAME")

    # RAG Configuration
    rag_enabled: bool = Field(True, env="RAG_ENABLED")
    rag_vector_store: str = Field("redis", env="RAG_VECTOR_STORE")
    rag_index_name: str = Field("rag:retail_agri", env="RAG_INDEX_NAME")
    rag_top_k: int = Field(4, env="RAG_TOP_K")
    embedding_provider: str = Field("openai", env="EMBEDDING_PROVIDER")  # openai | none
    openai_api_key: str = Field("", env="OPENAI_API_KEY")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    def model_post_init(self, __context) -> None:
        """معالجة post-init للقيم المعقدة"""
        if hasattr(self, 'allowed_origins') and isinstance(self.allowed_origins, str):
            self.allowed_origins = [origin.strip() for origin in self.allowed_origins.split(",")]

# إنشاء instance واحد من الإعدادات
settings = Settings()

# URLs and Headers
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {settings.openrouter_api_key}",
    "Content-Type": "application/json",
    "Accept": "application/json",
    "HTTP-Referer": settings.app_referrer,
    "X-Title": settings.app_title,
}

