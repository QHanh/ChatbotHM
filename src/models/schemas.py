from pydantic import BaseModel
from typing import List, Dict, Any, Literal, Optional

class ChatRequest(BaseModel):
    message: str
    model_choice: Literal["gemini", "lmstudio", "openai"] = "gemini"
    image_url: Optional[str] = None

class ImageInfo(BaseModel):
    product_name: str
    image_url: str
    product_link: str

class PurchaseItem(BaseModel):
    product_name: str
    properties: Optional[str] = None
    quantity: int = 1

class CustomerInfo(BaseModel):
    name: Optional[str] = None
    address: Optional[str] = None
    phone: Optional[str] = None
    items: List[PurchaseItem] = []

class ChatResponse(BaseModel):
    reply: str
    history: List[Dict[str, str]]
    images: List[ImageInfo] = []
    has_images: bool = False
    customer_info: Optional[CustomerInfo] = None
    has_purchase: bool = False
    human_handover_required: bool = False,
    has_negativity: bool = False

class QueryExtraction(BaseModel):
    product_name: str
    category: str