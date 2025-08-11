import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("❌ تأكد من وجود OPENROUTER_API_KEY في ملف .env")

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
model = "qwen/qwen3-30b-a3b:free"

print("💬 اكتب رسالتك (اكتب exit للخروج):")
messages = [{"role": "system", "content": "أنت مساعد ذكي وودود."}]

while True:
    user_input = input("👤 أنت: ")
    if user_input.lower() in ["exit", "quit", "خروج"]:
        print("👋 تم إنهاء المحادثة.")
        break
    messages.append({"role": "user", "content": user_input})

    data = {"model": model, "messages": messages}
    response = requests.post(url, headers=headers, json=data)
    result = response.json()

    if "choices" not in result:
        print("❌ خطأ في الرد:", result)
        continue

    reply = result["choices"][0]["message"]["content"]
    print("🤖 الموديل:", reply)
    messages.append({"role": "assistant", "content": reply})
