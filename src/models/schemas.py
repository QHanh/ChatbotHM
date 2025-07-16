from pydantic import BaseModel
from typing import List, Dict, Any, Literal

class ChatRequest(BaseModel):
    message: str
    model_choice: Literal["gemini", "lmstudio", "openai"] = "gemini"

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