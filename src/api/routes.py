from fastapi import HTTPException, Query
from typing import Dict, Any
import threading

from src.models.schemas import ChatRequest, ChatResponse, ImageInfo
from src.services.intent_service import (
    is_product_search_query, 
    llm_wants_specifications, 
    is_asking_for_images,
    extract_query_from_history
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

    # Kiểm tra xem câu hỏi có cần tìm kiếm sản phẩm hay không
    needs_product_search = is_product_search_query(user_query, history, model_choice)
    print(f"Câu hỏi '{user_query}' cần tìm kiếm sản phẩm: {needs_product_search}")

    if is_asking_for_more(user_query) and session_data.get("last_query"):
        response_text, retrieved_data = _handle_more_products(
            user_query, session_data, history, model_choice
        )
    else:
        response_text, retrieved_data = _handle_new_query(
            user_query, session_data, history, model_choice, needs_product_search
        )

    # Cập nhật lịch sử chat
    _update_chat_history(session_id, user_query, response_text, session_data)

    # Xử lý ảnh
    last_query_info = session_data.get("last_query", {})
    images = _process_images(user_query, history, model_choice, needs_product_search, retrieved_data, last_query_info)

    return ChatResponse(
        reply=response_text,
        history=chat_history[session_id]["messages"].copy(),
        images=images,
        has_images=len(images) > 0
    )

def _handle_more_products(user_query: str, session_data: dict, history: list, model_choice: str):
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
            user_query, retrieved_data, history, include_specs, model_choice, needs_product_search=True
        )
    
    session_data["offset"] = new_offset
    return response_text, retrieved_data

def _handle_new_query(user_query: str, session_data: dict, history: list, model_choice: str, needs_product_search: bool):
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

    include_specs = False
    if needs_product_search:
        include_specs = llm_wants_specifications(user_query, history, model_choice)

    response_text = generate_llm_response(
        user_query, 
        retrieved_data, 
        history, 
        include_specs=include_specs, 
        model_choice=model_choice,
        needs_product_search=needs_product_search
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

def _process_images(user_query: str, history: list, model_choice: str, needs_product_search: bool, retrieved_data: list, last_query_info: dict = None):
    """Xử lý và trả về danh sách ảnh nếu cần."""
    images = []
    wants_images = is_asking_for_images(user_query, history, model_choice)
    print(f"Khách hàng có hỏi về ảnh: {wants_images}")
    
    if wants_images and needs_product_search and retrieved_data:
        # Ưu tiên sử dụng thông tin từ LLM đã trích xuất
        if last_query_info and last_query_info.get('product_name'):
            target_product_name = last_query_info['product_name'].lower()
            print(f"Tên sản phẩm từ LLM: {target_product_name}")
            
            # Lọc sản phẩm dựa trên tên từ LLM
            filtered_products = _filter_by_llm_extracted_name(retrieved_data, target_product_name)
        else:
            # Fallback: Lấy từ khóa sản phẩm từ câu hỏi để lọc chính xác
            product_keywords = _extract_product_keywords_from_query(user_query)
            print(f"Từ khóa sản phẩm được trích xuất: {product_keywords}")
            
            # Lọc sản phẩm phù hợp nhất
            filtered_products = _filter_most_relevant_products(retrieved_data, product_keywords)
        
        print(f"Số sản phẩm sau khi lọc: {len(filtered_products)}")
        
        for item in filtered_products:
            if item.get('avatar_images'):
                # Đảm bảo product_link là string
                product_link = item.get('link_product', '')
                if not isinstance(product_link, str):
                    product_link = str(product_link) if product_link else ''
                
                images.append(ImageInfo(
                    product_name=item.get('product_name', ''),
                    image_url=item.get('avatar_images', ''),
                    product_link=product_link
                ))
    
    return images

def health_check():
    """Endpoint kiểm tra trạng thái API."""
    return {"status": "API is running"}
    
def _extract_product_keywords_from_query(user_query: str) -> list:
    """Trích xuất từ khóa sản phẩm từ câu hỏi của người dùng."""
    # Loại bỏ các từ không cần thiết
    stop_words = ['có', 'ảnh', 'không', 'cho', 'tôi', 'xem', 'show', 'hình', 'của', 'sản', 'phẩm']
    
    # Tách từ và loại bỏ stop words
    words = user_query.lower().split()
    keywords = [word for word in words if word not in stop_words and len(word) > 1]
    
    return keywords

def _filter_most_relevant_products(products: list, keywords: list) -> list:
    """Lọc sản phẩm phù hợp nhất dựa trên từ khóa."""
    if not keywords:
        # Nếu không có từ khóa, chỉ trả về sản phẩm đầu tiên (có score cao nhất từ Elasticsearch)
        return products[:1] if products else []
    
    scored_products = []
    
    for product in products:
        product_name = product.get('product_name', '').lower()
        score = 0
        
        # Tính điểm dựa trên số từ khóa xuất hiện trong tên sản phẩm
        for keyword in keywords:
            if keyword in product_name:
                score += 1
        
        # Thêm điểm bonus nếu tên sản phẩm chứa tất cả từ khóa
        if all(keyword in product_name for keyword in keywords):
            score += 2
        
        if score > 0:
            scored_products.append((product, score))
    
    # Sắp xếp theo điểm giảm dần
    scored_products.sort(key=lambda x: x[1], reverse=True)
    
    # Trả về tối đa 3 sản phẩm có điểm cao nhất
    if scored_products:
        max_score = scored_products[0][1]
        # Chỉ lấy những sản phẩm có điểm cao nhất
        best_products = [product for product, score in scored_products if score == max_score]
        return best_products[:3]  # Tối đa 3 sản phẩm
    
    # Nếu không có sản phẩm nào match, trả về sản phẩm đầu tiên
    return products[:1] if products else []

def _filter_by_llm_extracted_name(products: list, target_product_name: str) -> list:
    """Lọc sản phẩm dựa trên tên sản phẩm được trích xuất bởi LLM."""
    if not target_product_name:
        return products[:1] if products else []
    
    # Tách từ khóa từ tên sản phẩm target
    target_keywords = target_product_name.lower().split()
    
    scored_products = []
    
    for product in products:
        product_name = product.get('product_name', '').lower()
        score = 0
        
        # Tính điểm dựa trên số từ khóa xuất hiện trong tên sản phẩm
        for keyword in target_keywords:
            if keyword in product_name:
                score += 1
        
        # Bonus điểm nếu tên sản phẩm chứa tất cả từ khóa
        if all(keyword in product_name for keyword in target_keywords):
            score += 3
        
        # Bonus điểm nếu tên sản phẩm bắt đầu bằng từ khóa đầu tiên
        if target_keywords and product_name.startswith(target_keywords[0]):
            score += 2
        
        # Bonus điểm nếu tên sản phẩm chứa chính xác chuỗi target
        if target_product_name in product_name:
            score += 5
        
        scored_products.append((product, score))
    
    # Sắp xếp theo điểm giảm dần
    scored_products.sort(key=lambda x: x[1], reverse=True)
    
    # Chỉ lấy sản phẩm có điểm cao nhất
    if scored_products and scored_products[0][1] > 0:
        max_score = scored_products[0][1]
        best_products = [product for product, score in scored_products if score == max_score]
        return best_products[:2]  # Tối đa 2 sản phẩm có điểm cao nhất
    
    # Nếu không có sản phẩm nào match, trả về sản phẩm đầu tiên
    return products[:1] if products else []