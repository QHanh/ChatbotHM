from fastapi import HTTPException, Query
from typing import Dict, Any
import threading

from src.models.schemas import ChatRequest, ChatResponse, ImageInfo
from src.services.intent_service import (
    is_product_search_query, 
    llm_wants_specifications, 
    is_asking_for_images,
    extract_query_from_history,
    resolve_product_for_image,
    llm_wants_inventory
)
from src.services.search_service import search_products
from src.services.response_service import generate_llm_response
from src.utils.helpers import is_asking_for_more
from src.config.settings import PAGE_SIZE

# Global variables for chat history
chat_history: Dict[str, Dict[str, Any]] = {}
chat_history_lock = threading.Lock()

async def chat_endpoint(request: ChatRequest, session_id: str = "default") -> ChatResponse:
    """
    Endpoint chính để tương tác với chatbot.
    """
    user_query = request.message
    model_choice = request.model_choice
    
    if not user_query:
        raise HTTPException(status_code=400, detail="Không có tin nhắn nào được gửi")

    with chat_history_lock:
        session_data = chat_history.get(session_id, {
            "messages": [],
            "last_query": None,
            "offset": 0
        }).copy()
        history = session_data["messages"][-10:].copy()

    # Kiểm tra các ý định của người dùng một lần
    needs_product_search = is_product_search_query(user_query, history, model_choice)
    wants_images = is_asking_for_images(user_query, history, model_choice)
    asking_for_more = is_asking_for_more(user_query)

    print(f"Câu hỏi '{user_query}' cần tìm kiếm sản phẩm: {needs_product_search}")
    print(f"Khách hàng có hỏi về ảnh: {wants_images}")

    if asking_for_more and session_data.get("last_query"):
        response_text, retrieved_data = _handle_more_products(
            user_query, session_data, history, model_choice, wants_images
        )
    else:
        response_text, retrieved_data = _handle_new_query(
            user_query, session_data, history, model_choice, needs_product_search, wants_images
        )

    # Cập nhật lịch sử chat
    _update_chat_history(session_id, user_query, response_text, session_data)

    # Xử lý ảnh
    images = _process_images(wants_images, needs_product_search, retrieved_data, user_query, history, model_choice)

    return ChatResponse(
        reply=response_text,
        history=chat_history[session_id]["messages"].copy(),
        images=images,
        has_images=len(images) > 0
    )

def _handle_more_products(user_query: str, session_data: dict, history: list, model_choice: str, wants_images: bool):
    """Xử lý khi người dùng muốn xem thêm sản phẩm."""
    last_query = session_data["last_query"]
    new_offset = session_data["offset"] + PAGE_SIZE
    
    print(f"Người dùng muốn xem thêm. Tìm lại với query='{last_query}' và offset={new_offset}")
    
    retrieved_data = search_products(
        product_name=last_query["product_name"],
        category=last_query["category"],
        offset=new_offset
    )

    if not retrieved_data:
        response_text = f"Dạ, em đã giới thiệu hết các sản phẩm '{last_query['product_name']}' mà cửa hàng có rồi ạ. Anh/chị có muốn tìm sản phẩm nào khác không?"
    else:
        include_specs = llm_wants_specifications(user_query, history, model_choice)
        response_text = generate_llm_response(
            user_query, retrieved_data, history, include_specs, model_choice, needs_product_search=True, wants_images=wants_images
        )
    
    session_data["offset"] = new_offset
    return response_text, retrieved_data

def _handle_new_query(user_query: str, session_data: dict, history: list, model_choice: str, needs_product_search: bool, wants_images: bool):
    """Xử lý câu hỏi mới."""
    retrieved_data = []
    
    if needs_product_search:
        query_for_es = extract_query_from_history(user_query, history, model_choice)
        retrieved_data = search_products(
            product_name=query_for_es.get("product_name", user_query),
            category=query_for_es.get("category", user_query),
            offset=0
        )
        session_data["last_query"] = query_for_es
        session_data["offset"] = 0
    else:
        session_data["last_query"] = None

    if needs_product_search:
        include_specs = llm_wants_specifications(user_query, history, model_choice)
        include_inventory = llm_wants_inventory(user_query, history, model_choice)
    else:
        include_specs = False
        include_inventory = False

    response_text = generate_llm_response(
        user_query, 
        retrieved_data, 
        history, 
        include_specs=include_specs, 
        model_choice=model_choice,
        needs_product_search=needs_product_search,
        wants_images=wants_images,
        include_inventory=include_inventory
    )
    
    return response_text, retrieved_data
    
def _update_chat_history(session_id: str, user_query: str, response_text: str, session_data: dict):
    """Cập nhật lịch sử chat."""
    with chat_history_lock:
        session_data_to_update = chat_history.get(session_id, {
            "messages": [],
            "last_query": session_data["last_query"],
            "offset": session_data["offset"]
        })
        session_data_to_update["messages"].append({"user": user_query, "bot": response_text})
        session_data_to_update["last_query"] = session_data["last_query"]
        session_data_to_update["offset"] = session_data["offset"]
        
        chat_history[session_id] = session_data_to_update

def _process_images(wants_images: bool, needs_product_search: bool, retrieved_data: list, user_query: str, history: list, model_choice: str) -> list[ImageInfo]:
    """Xử lý và trả về danh sách ảnh nếu cần."""
    images = []
    
    if wants_images and needs_product_search and retrieved_data:
        # Sử dụng LLM để xác định (các) sản phẩm cụ thể mà người dùng muốn xem
        target_product_names = resolve_product_for_image(user_query, history, retrieved_data, model_choice)
        
        print(f"LLM đã xác định các sản phẩm để hiển thị ảnh: {target_product_names}")

        # Tạo một map để tra cứu sản phẩm nhanh hơn
        product_map = {p.get('product_name', ''): p for p in retrieved_data}

        for name in target_product_names:
            product = product_map.get(name)
            if product and product.get('avatar_images'):
                product_link = product.get('link_product', '')
                if not isinstance(product_link, str):
                    product_link = str(product_link) if product_link else ''
                
                images.append(ImageInfo(
                    product_name=product.get('product_name', ''),
                    image_url=product.get('avatar_images', ''),
                    product_link=product_link
                ))

    return images

def health_check():
    """Endpoint kiểm tra trạng thái API."""
    return {"status": "API is running"}