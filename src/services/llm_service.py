import os
import requests
from src.config.settings import GEMINI_API_KEY, LMSTUDIO_API_URL, LMSTUDIO_MODEL, OPENAI_API_KEY

def get_gemini_model():
    """Khởi tạo và trả về instance của Gemini Model."""
    if not GEMINI_API_KEY:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        return model
    except Exception as e:
        print(f"Lỗi khi khởi tạo Gemini: {e}")
        return None

def get_lmstudio_response(prompt: str):
    """Gửi prompt đến LM Studio API và nhận phản hồi."""
    try:
        url = f"{LMSTUDIO_API_URL}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "model": LMSTUDIO_MODEL,
            "temperature": 0.7,
            "max_tokens": 4000
        }
        
        print(f"Gửi yêu cầu đến LM Studio API: {url}")
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        return "Không nhận được phản hồi từ LM Studio."
    except Exception as e:
        print(f"Lỗi khi gọi LM Studio API: {e}")
        return f"Lỗi kết nối đến LM Studio: {str(e)}"

def get_openai_model():
    """Khởi tạo và trả về client openai chuẩn >=1.0.0, hoặc None nếu thiếu key."""
    try:
        import openai
        if not OPENAI_API_KEY:
            return None
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        return client
    except Exception as e:
        print(f"Lỗi khi khởi tạo OpenAI client: {e}")
        return None