import time
import requests

# ====== اختبار سرعة OpenRouter ======
def test_openrouter_speed():
    headers = {
        "Authorization": "Bearer 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",
        "Content-Type": "application/json"
    }

    body = {
        "model": "meta-llama/llama-3.1-8b-instruct",
        "messages": [
            {"role": "user", "content": "اختبار السرعة المباشر من OpenRouter"}
        ]
    }

    start_time = time.time()
    response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              headers=headers, json=body)
    total_time = time.time() - start_time

    if response.status_code == 200 and "choices" in response.json():
        print("\n🚀 سرعة OpenRouter:")
        print("⏱ الزمن الكلي:", round(total_time, 3), "ث")
        print("📩 النص المستلم:", response.json()["choices"][0]["message"]["content"])
    else:
        print("\n⚠ خطأ في OpenRouter:", response.status_code, response.text)


# ====== اختبار سرعة Render ======
def test_render_speed():
    url = "https://bot-1-u6t9.onrender.com/ping"  # لازم تضيف /ping في الكود على Render
    start_time = time.time()
    response = requests.get(url)
    total_time = time.time() - start_time

    print("\n🚀 سرعة Render:")
    if response.status_code == 200:
        print("⏱ الزمن الكلي:", round(total_time, 3), "ث")
        print("📩 الرد:", response.json())
    else:
        print("⚠ خطأ:", response.status_code, response.text)


if __name__ == "__main__":
    test_openrouter_speed()
    test_render_speed()
