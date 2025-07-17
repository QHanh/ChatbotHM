import os
from dotenv import load_dotenv
import json
import random

# Tải biến môi trường từ file .env
load_dotenv()

# Cấu hình chung
PAGE_SIZE = 5

# API Keys
api_keys = json.loads(os.getenv("GEMINI_API_KEY"))
GEMINI_API_KEY = random.choice(api_keys)
LMSTUDIO_API_URL = os.getenv("LMSTUDIO_API_URL")
LMSTUDIO_MODEL = os.getenv("LMSTUDIO_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# FastAPI Config
APP_CONFIG = {
    "title": "Chatbot Tư Vấn Bán Hàng",
    "description": "RAG Chatbot sử dụng Elasticsearch",
    "version": "1.2.0"
}

# CORS Config
CORS_CONFIG = {
    "allow_origins": ["*"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}