# -*- coding: utf-8 -*-
import os
import json
import time
import requests
from dotenv import load_dotenv

# ============ الإعدادات ============
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("❌ تأكد من وجود OPENROUTER_API_KEY في ملف .env")

# يمكنك تغيير الموديل هنا (الافتراضي ما وضعته أنت)
MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-30b-a3b:free")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# هيدرز موصى بها من OpenRouter (تعريف بسيط للتطبيق)
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
    # اختياريان لكن مستحسنان:
    "HTTP-Referer": os.getenv("APP_REFERRER", "http://localhost"),
    "X-Title": os.getenv("APP_TITLE", "CLI Chat via OpenRouter"),
}

# باراميترات توليد افتراضية (يمكن تعديلها أثناء التشغيل لاحقًا)
GEN_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 512,  # قلل/زود بحسب حاجتك
}

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "أنت مساعد ذكي وودود ودقيق.")

# ============ أدوات مساعدة ============
def post_with_retry(payload, max_retries=3, stream=False, timeout=60):
    """إرسال الطلب مع محاولات إعادة تلقائية على أخطاء الشبكة/السيرفر."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers=HEADERS,
                json=payload,
                stream=stream,
                timeout=timeout,
            )
            # أخطاء HTTP
            if resp.status_code >= 400:
                # حاول استخراج رسالة الخطأ من JSON إن وجدت
                try:
                    err = resp.json()
                except Exception:
                    err = {"error": {"message": resp.text}}
                raise RuntimeError(f"HTTP {resp.status_code}: {err}")
            return resp
        except Exception as e:
            if attempt == max_retries:
                raise
            # مهلة صغيرة قبل إعادة المحاولة (backoff)
            time.sleep(1.2 * attempt)

def stream_chat(messages, extra_params=None):
    """ستريمنج الرد من الموديل سطرًا بسطر."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
    }
    if extra_params:
        payload.update(extra_params)

    resp = post_with_retry(payload, stream=True)
    full_text = []
    # OpenRouter يبثّ كسطور "data: {...}"
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if raw_line.startswith("data:"):
            data_str = raw_line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                delta = data.get("choices", [{}])[0].get("delta", {})
                chunk = delta.get("content")
                if chunk:
                    print(chunk, end="", flush=True)
                    full_text.append(chunk)
            except json.JSONDecodeError:
                # تجاهل أي سطر غير قابل للفك (نادرًا)
                continue
    print()  # سطر جديد بعد انتهاء الستريمنج
    return "".join(full_text)

def non_stream_chat(messages, extra_params=None):
    """رد غير متدفق (كتلة واحدة)."""
    payload = {
        "model": MODEL,
        "messages": messages,
    }
    if extra_params:
        payload.update(extra_params)

    resp = post_with_retry(payload, stream=False)
    result = resp.json()
    if "choices" not in result:
        raise RuntimeError(f"❌ رد غير متوقع: {result}")
    return result["choices"][0]["message"]["content"], result.get("usage")

def save_transcript(messages, path="chat_transcript.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    print(f"💾 تم حفظ المحادثة في: {path}")

# ============ الحلقة الرئيسية ============
def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print(f"🤖 النموذج: {MODEL}")
    print("💬 اكتب رسالتك (أوامر خاصة: /exit، /reset، /params، /save):")

    while True:
        try:
            user_input = input("👤 أنت: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 تم إنهاء المحادثة.")
            break

        if not user_input:
            continue

        # أوامر الإدارة السريعة:
        if user_input.lower() in ["/exit", "exit", "quit", "خروج"]:
            print("👋 تم إنهاء المحادثة.")
            break

        if user_input.lower() in ["/reset", "reset", "تفريغ"]:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("🔄 تم إعادة ضبط السياق.")
            continue

        if user_input.lower().startswith("/params"):
            # تغيير باراميترات الجيل أثناء التشغيل، مثال: /params temp=0.3 max=200
            try:
                parts = user_input.split()[1:]
                for p in parts:
                    k, v = p.split("=", 1)
                    k = k.strip().lower()
                    v = v.strip()
                    if k in ("temperature", "top_p"):
                        GEN_PARAMS[k] = float(v)
                    elif k in ("max_tokens", "max"):
                        GEN_PARAMS["max_tokens"] = int(v)
                print(f"⚙️ تم التحديث: {GEN_PARAMS}")
            except Exception:
                print("ℹ️ صيغة صحيحة مثال: /params temperature=0.3 max_tokens=256")
            continue

        if user_input.lower().startswith("/save"):
            # حفظ نص المحادثة
            parts = user_input.split(maxsplit=1)
            path = "chat_transcript.json" if len(parts) == 1 else parts[1]
            save_transcript(messages, path)
            continue

        # الرسالة العادية:
        messages.append({"role": "user", "content": user_input})

        # ستريمنج (أنصح به للسرعة الإحساسية)
        try:
            reply_text = stream_chat(messages, GEN_PARAMS)
            messages.append({"role": "assistant", "content": reply_text})
        except Exception as e:
            # لو الستريمنج فشل لسببٍ ما، نحاول طريقة non-stream كخطة بديلة
            print(f"\n⚠️ تعذر الستريمنج، محاولة رد عادي... ({e})")
            try:
                reply_text, usage = non_stream_chat(messages, GEN_PARAMS)
                print("🤖 الموديل:", reply_text)
                messages.append({"role": "assistant", "content": reply_text})
                if usage:
                    print(f"📊 usage: {usage}")
            except Exception as e2:
                print("❌ خطأ في الرد:", e2)

if __name__ == "__main__":
    main()
