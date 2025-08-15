from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional

class ChatRequest(BaseModel):
    message: str
    model_choice: Literal["gemini", "lmstudio", "openai"] = "gemini"
    image_url: Optional[str] = None

class ControlBotRequest(BaseModel):
    command: str = Field(..., description="Lệnh điều khiển bot, ví dụ: 'start', 'stop'")

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
    phone: Optional[str] = None
    address: Optional[str] = None
    items: List[PurchaseItem]

class Action(BaseModel):
    action: str
    url: str

class ChatResponse(BaseModel):
    reply: str
    history: List[Dict[str, str]]
    images: List[ImageInfo] = []
    has_images: bool = False
    has_purchase: bool = False
    customer_info: Optional[CustomerInfo] = None
    human_handover_required: Optional[bool] = False
    has_negativity: Optional[bool] = False
    action_data: Optional[Action] = None

class QueryExtraction(BaseModel):
    product_name: str
    category: str