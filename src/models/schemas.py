from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ChatRequest(BaseModel):
    message: str
    model_choice: str = "gemini"  # Mặc định sử dụng Gemini, tùy chọn khác là "lmstudio"

class ImageInfo(BaseModel):
    product_name: str
    image_url: str
    product_link: str


class ChatResponse(BaseModel):
    reply: str
    history: List[Dict[str, str]]
    images: List[ImageInfo]
    has_images: bool


class QueryExtraction(BaseModel):
    product_name: str
    category: str