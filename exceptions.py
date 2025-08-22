# -*- coding: utf-8 -*-
"""
استثناءات مخصصة للتطبيق
"""

from fastapi import HTTPException
from typing import Optional, Dict, Any


class ChatbotException(Exception):
    """الاستثناء الأساسي للتطبيق"""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class LLMServiceException(ChatbotException):
    """خطأ في خدمة الذكاء الاصطناعي"""
    
    def __init__(self, message: str = "خدمة الذكاء غير متاحة", details: Dict[str, Any] = None):
        super().__init__(message, "LLM_SERVICE_ERROR", details)


class RateLimitException(ChatbotException):
    """خطأ في حد الطلبات"""
    
    def __init__(self, message: str = "طلبات كثيرة جداً", retry_after: int = 60):
        super().__init__(message, "RATE_LIMIT_EXCEEDED", {"retry_after": retry_after})


class ConversationException(ChatbotException):
    """خطأ في إدارة المحادثات"""
    
    def __init__(self, message: str = "خطأ في المحادثة", conversation_id: str = None):
        details = {"conversation_id": conversation_id} if conversation_id else {}
        super().__init__(message, "CONVERSATION_ERROR", details)


class PlatformException(ChatbotException):
    """خطأ في منصة التواصل"""
    
    def __init__(self, platform: str, message: str = "خطأ في المنصة", details: Dict[str, Any] = None):
        details = details or {}
        details["platform"] = platform
        super().__init__(message, "PLATFORM_ERROR", details)


def to_http_exception(exception: ChatbotException) -> HTTPException:
    """تحويل الاستثناء المخصص إلى HTTPException"""
    
    status_codes = {
        "LLM_SERVICE_ERROR": 503,
        "RATE_LIMIT_EXCEEDED": 429,
        "CONVERSATION_ERROR": 400,
        "PLATFORM_ERROR": 502,
    }
    
    status_code = status_codes.get(exception.error_code, 500)
    
    return HTTPException(
        status_code=status_code,
        detail={
            "error": exception.error_code,
            "message": exception.message,
            "details": exception.details
        }
    )

