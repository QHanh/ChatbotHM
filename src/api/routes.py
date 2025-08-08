from fastapi import HTTPException
from typing import Dict, Any, List, Set
import threading
import io
import requests
from PIL import Image
import google.generativeai as genai

from src.models.schemas import ChatRequest, ChatResponse, ImageInfo, PurchaseItem, CustomerInfo, ControlBotRequest
from src.services.intent_service import analyze_intent_and_extract_entities, extract_customer_info
from src.services.search_service import search_products, search_products_by_image
from src.services.response_service import generate_llm_response
from src.utils.helpers import is_asking_for_more, format_history_text
from src.config.settings import PAGE_SIZE
from src.services.response_service import evaluate_and_choose_product, evaluate_purchase_confirmation, filter_products_with_ai
import time
HANDOVER_TIMEOUT = 900

chat_history: Dict[str, Dict[str, Any]] = {}
chat_history_lock = threading.Lock()
bot_running = True
bot_state_lock = threading.Lock()

def _get_product_key(product: Dict) -> str:
    """Tạo một key định danh duy nhất cho sản phẩm."""
    return f"{product.get('product_name', '')}::{product.get('properties', '')}"

async def chat_endpoint(request: ChatRequest, session_id: str = "default") -> ChatResponse:
    with bot_state_lock:
        if not bot_running:
            return ChatResponse(reply="", history=[], human_handover_required=False)
    
    user_query = request.message
    model_choice = request.model_choice
    image_url = request.image_url

    if not user_query and not image_url:
        raise HTTPException(status_code=400, detail="Không có tin nhắn hoặc hình ảnh nào được gửi")

    with chat_history_lock:
        session_data = chat_history.get(session_id, {
            "messages": [],
            "last_query": None,
            "offset": 0,
            "shown_product_keys": set(),
            "state": None, 
            "pending_purchase_item": None,
            "negativity_score": 0,
            "handover_timestamp": None,
            "collected_customer_info": {}
        }).copy()
        history = session_data["messages"][-8:].copy()

    API_ENDPOINT = "https://embed.doiquanai.vn/embed"
    if image_url:
        print(f"Phát hiện hình ảnh từ URL: {image_url}, bắt đầu xử lý...")
        try:
            # 1. Tải ảnh từ URL
            response = requests.post(API_ENDPOINT, data={"image_url": image_url}, timeout=15)
            response.raise_for_status()
            result = response.json()

            # Kiểm tra có lỗi không
            if "embedding" in result:
                embedding_vector = result["embedding"]
                print(" -> Tạo embedding cho ảnh thành công.")
            else:
                print(" -> Lỗi từ API:", result.get("error", "Không rõ lỗi"))

            retrieved_data = search_products_by_image(embedding_vector)
            if not retrieved_data:
                response_text="Dạ, cửa hàng em không có sản phẩm này ạ."
                _update_chat_history(session_id, user_query, response_text, session_data)
                return ChatResponse(reply=response_text, history=chat_history[session_id]["messages"].copy(), human_handover_required=False)
            
            if not user_query:
                user_query = "Ảnh này là sản phẩm gì vậy shop?"

            response_text = generate_llm_response(
                user_query=user_query,
                search_results=retrieved_data,
                history=history,
                model_choice=model_choice,
                is_image_search=True
            )
            
            _update_chat_history(session_id, user_query, response_text, session_data)
            return ChatResponse(reply=response_text, history=chat_history[session_id]["messages"].copy(), human_handover_required=False)

        except Exception as e:
            print(f"Lỗi nghiêm trọng trong luồng xử lý ảnh: {e}")
            return ChatResponse(reply="Dạ, em xin lỗi, đã có lỗi xảy ra khi xem hình ảnh của mình ạ.", history=history)

    if session_data.get("state") == "stop_bot":
        _update_chat_history(session_id, user_query, "", session_data)
        return ChatResponse(reply="", history=chat_history[session_id]["messages"].copy(), human_handover_required=False)
    
    if user_query.strip().lower() == "/bot":
        session_data["state"] = None
        session_data["negativity_score"] = 0
        response_text = "Dạ, em có thể giúp gì tiếp cho anh/chị ạ?"
        _update_chat_history(session_id, user_query, response_text, session_data)
        return ChatResponse(reply=response_text, history=chat_history[session_id]["messages"].copy(), human_handover_required=False)
        
    if session_data.get("state") == "human_chatting":
        _update_chat_history(session_id, user_query, "", session_data)
        return ChatResponse(reply="", history=chat_history[session_id]["messages"].copy(), human_handover_required=False)
    
    if session_data.get("state") == "human_calling":
        response_text = "Dạ, nhân viên bên em đang vào ngay ạ, anh/chị vui lòng đợi trong giây lát."
        _update_chat_history(session_id, user_query, response_text, session_data)
        return ChatResponse(reply=response_text, history=chat_history[session_id]["messages"].copy(), human_handover_required=False)

    if session_data.get("state") == "awaiting_purchase_confirmation":
        history_text = format_history_text(history, limit=4)
        evaluation = evaluate_purchase_confirmation(user_query, history_text, model_choice)
        decision = evaluation.get("decision")
        if decision == "CONFIRM":
            collected_info = session_data.get("collected_customer_info", {})
            if collected_info.get("name") and collected_info.get("phone") and collected_info.get("address"):
                
                item_data_with_quantity = session_data.get("pending_purchase_item", {})
                item_data = item_data_with_quantity.get("product_data", {})
                quantity = item_data_with_quantity.get("quantity", 1)
                
                purchase_item = PurchaseItem(
                    product_name=item_data.get("product_name", "N/A"),
                    properties=item_data.get("properties"),
                    quantity=quantity
                )
                
                customer_info_obj = CustomerInfo(
                    name=collected_info.get("name"),
                    phone=collected_info.get("phone"),
                    address=collected_info.get("address"),
                    items=[purchase_item]
                )
                
                response_text = f"Dạ em đã nhận được thông tin cho sản phẩm {purchase_item.product_name}. Em sẽ tạo một đơn hàng mới cho mình ạ. Em cảm ơn anh/chị! /-heart"
                
                session_data["state"] = None
                session_data["pending_purchase_item"] = None
                
                _update_chat_history(session_id, user_query, response_text, session_data)
                
                return ChatResponse(
                    reply=response_text,
                    history=chat_history[session_id]["messages"].copy(),
                    customer_info=customer_info_obj,
                    has_purchase=True
                )

            else:
                pending_item = session_data.get("pending_purchase_item", {})
                product_data = pending_item.get("product_data", {})
                product_link = product_data.get("link_product", "#")

                response_text = (
                    f"Dạ vâng ạ. Vậy để đặt đơn hàng, anh/chị có thể vào đường link {product_link} để đặt hàng hoặc đến xem trực tiếp tại cửa hàng chúng em tại số 8 ngõ 117 Thái Hà, Đống Đa, Hà Nội (thời gian mở cửa từ 8h đến 18h).\n"
                    "\nDạ anh/chị vui lòng cho em xin tên, số điện thoại và địa chỉ để em lên đơn cho anh/chị ạ. /-ok\n"
                    "Em cảm ơn anh/chị nhiều ạ. /-heart"
                )
                session_data["state"] = "awaiting_customer_info"
                
                _update_chat_history(session_id, user_query, response_text, session_data)
                return ChatResponse(reply=response_text, history=chat_history[session_id]["messages"].copy(), human_handover_required=False)
        elif decision == "CANCEL":
            response_text = "Dạ, em đã hủy yêu cầu đặt mua sản phẩm, nếu anh/chị muốn mua sản phẩm khác thì báo lại cho em ạ. /-heart"
            session_data["state"] = None
            session_data["pending_purchase_item"] = None
            _update_chat_history(session_id, user_query, response_text, session_data)
            return ChatResponse(reply=response_text, history=chat_history[session_id]["messages"].copy(), human_handover_required=False)
        else:
            session_data["state"] = None
            session_data["pending_purchase_item"] = None

    if session_data.get("state") == "awaiting_customer_info":        
        current_info = session_data.get("collected_customer_info", {})
        extracted_info = extract_customer_info(user_query, model_choice)

        # Cập nhật thông tin thu thập được
        for key, value in extracted_info.items():
            if value and not current_info.get(key):
                current_info[key] = value

        # Kiểm tra những thông tin còn thiếu
        missing_info = []
        if not current_info.get("name"):
            missing_info.append("tên")
        if not current_info.get("phone"):
            missing_info.append("số điện thoại")
        if not current_info.get("address"):
            missing_info.append("địa chỉ")

        # Nếu còn thiếu thông tin, tiếp tục hỏi
        if missing_info:
            response_text = f"Dạ, anh/chị vui lòng cho em xin { ' và '.join(missing_info) } để em lên đơn ạ."
            session_data["collected_customer_info"] = current_info
            _update_chat_history(session_id, user_query, response_text, session_data)
            return ChatResponse(reply=response_text, history=chat_history[session_id]["messages"].copy(), human_handover_required=False)

        # Nếu đã đủ thông tin
        item_data_with_quantity = session_data.get("pending_purchase_item", {})
        item_data = item_data_with_quantity.get("product_data", {})
        quantity = item_data_with_quantity.get("quantity", 1)
        
        props_value = item_data.get("properties")
        final_props = None
        if props_value is not None and str(props_value).strip() not in ['0', '']:
            final_props = str(props_value)
            
        purchase_item = PurchaseItem(
            product_name=item_data.get("product_name", "N/A"),
            properties=final_props,
            quantity=quantity
        )

        customer_info_obj = CustomerInfo(
            name=current_info.get("name"),
            phone=current_info.get("phone"),
            address=current_info.get("address"),
            items=[purchase_item]
        )
        
        response_text = "Dạ em đã nhận được thông tin. Em cảm ơn anh/chị! /-heart"
        session_data["state"] = None
        session_data["pending_purchase_item"] = None
        
        _update_chat_history(session_id, user_query, response_text, session_data)
        
        return ChatResponse(
            reply=response_text,
            history=chat_history[session_id]["messages"].copy(),
            customer_info=customer_info_obj,
            has_purchase=True,
            human_handover_required=False
        )

    analysis_result = analyze_intent_and_extract_entities(user_query, history, model_choice)

    if analysis_result.get("is_negative"):
        session_data["negativity_score"] += 1
        if session_data["negativity_score"] >= 4:
            analysis_result["wants_human_agent"] = True

    if analysis_result.get("wants_store_info"):
        response_text = "Dạ, anh/chị có thể đến xem và mua hàng trực tiếp tại cửa hàng Hoàng Mai Mobile ở địa chỉ:\n👉 Số 8 ngõ 117 Thái Hà, Phường Trung Liệt, Quận Đống Đa, Hà Nội.\n👉 SĐT: 0982153333\n👉 Link google map: https://maps.app.goo.gl/HM9RTi64wpC1GgFp8?g_st=ic"
        map_image_url = "https://s3.hn-1.cloud.cmctelecom.vn/dangbai/hmstore.jpg"
        map_image = [
            ImageInfo(
                product_name="Của hàng Hoàng Mai Mobile",
                image_url=map_image_url,
                product_link=""
            )
        ]
        _update_chat_history(session_id, user_query, response_text, session_data)
        return ChatResponse(
            reply=response_text, 
            history=chat_history[session_id]["messages"].copy(),
            human_handover_required=False,
            has_negativity=False,
            images=map_image,
            has_images=True
        )
    
    if analysis_result.get("wants_human_agent"):
        response_text = "Dạ em đã thông báo lại với nhân viên phụ trách. Anh/chị đợi chút, nhân viên phụ trách bên em sẽ vào trả lời trực tiếp ngay ạ."
        session_data["state"] = "human_calling"
        session_data["handover_timestamp"] = time.time()
        
        _update_chat_history(session_id, user_query, response_text, session_data)
        
        return ChatResponse(
            reply=response_text,
            history=chat_history[session_id]["messages"].copy(),
            human_handover_required=True,
            has_negativity=False
        )

    response_text, retrieved_data, product_images = "", [], []
    asking_for_more = is_asking_for_more(user_query)

    if analysis_result.get("is_add_to_order_intent"):
        response_text = "Dạ vâng, anh/chị muốn mua thêm sản phẩm nào ạ? :b"
        session_data["last_query"] = None

    elif analysis_result.get("is_purchase_intent"):
        search_params = analysis_result["search_params"]
        requested_quantity = search_params.get("quantity", 1)

        last_query = session_data.get("last_query") or {}

        products = search_products(
            product_name=search_params.get("product_name") or last_query.get("product_name", ""),
            category=search_params.get("category") or last_query.get("category", ""),
            properties=search_params.get("properties") or last_query.get("properties", ""),
            # strict_properties=True
        )
        if not products:
            response_text = f"Dạ, em xin lỗi, bên em không có sản phẩm này ạ. :--|"
        else:
            history_text = format_history_text(history, limit=6)
            evaluation = evaluate_and_choose_product(user_query, history_text, products, model_choice)
            
            request_type = evaluation.get("type")
            product_to_check = evaluation.get("product")

            if request_type == "SPECIFIC" and product_to_check:
                requested_quantity = search_params.get("quantity", 1)
                available_stock = product_to_check.get("inventory", 0)
                product_name = product_to_check.get("product_name")
                properties = product_to_check.get("properties")
                available_stock = product_to_check.get("inventory", 0)
                
                full_name = product_name
                if properties and str(properties).strip() not in ['0', '']:
                    full_name = f"{product_name} ({properties})"
                
                if available_stock == 0:
                    response_text = f"Dạ, em xin lỗi, sản phẩm {full_name} bên em hiện đang hết hàng ạ. :--|"
                elif requested_quantity > available_stock:
                    response_text = f"Dạ, em xin lỗi, sản phẩm {full_name} bên em chỉ còn {available_stock} sản phẩm ạ. Anh/chị có muốn lấy số lượng này không ạ? :b"
                else:
                    response_text = f"Dạ, anh/chị muốn đặt mua sản phẩm {full_name} (Số lượng: {requested_quantity}) đúng không ạ? :b"
                    session_data["state"] = "awaiting_purchase_confirmation"
                    session_data["pending_purchase_item"] = {
                        "product_data": product_to_check,
                        "quantity": requested_quantity
                    }
            elif request_type == "GENERAL":
                response_text = (
                    f"Dạ, bên em có nhiều loại {search_params.get('product_name')} ạ.\n"
                    "Anh/chị đang quan tâm đến loại cụ thể nào để em kiểm tra giúp mình ạ? /-heart"
                )
                retrieved_data = products
            else:
                response_text = f"Dạ, em xin lỗi, bên em không có sản phẩm này của mình ạ. :--|"

    elif asking_for_more and session_data.get("last_query"):
        response_text, retrieved_data, product_images = _handle_more_products(
            user_query, session_data, history, model_choice, analysis_result
        )
    else:
        session_data["shown_product_keys"] = set()
        response_text, retrieved_data, product_images = _handle_new_query(
            user_query, session_data, history, model_choice, analysis_result
        )

    _update_chat_history(session_id, user_query, response_text, session_data)
    images = _process_images(analysis_result.get("wants_images", False), retrieved_data, product_images)

    return ChatResponse(
        reply=response_text,
        history=chat_history.get(session_id, {}).get("messages", []).copy(),
        images=images,
        has_images=len(images) > 0,
        has_purchase=analysis_result.get("is_purchase_intent", False),
        human_handover_required=analysis_result.get("human_handover_required", False),
        has_negativity=False
    )

async def control_bot_endpoint(request: ControlBotRequest, session_id: str):
    """
    Điều khiển trạng thái của bot (dừng hoặc tiếp tục).
    """
    with chat_history_lock:
        if session_id not in chat_history:
            # Nếu session_id không tồn tại, tạo mới.
            chat_history[session_id] = {
                "messages": [],
                "last_query": None,
                "offset": 0,
                "shown_product_keys": set(),
                "state": None,
                "pending_purchase_item": None,
                "negativity_score": 0,
                "handover_timestamp": None,
                "collected_customer_info": {}
            }
            print(f"Đã tạo session mới: {session_id} thông qua control endpoint.")

        command = request.command.lower()
        
        if command == "stop":
            # Chuyển bot sang trạng thái stop_bot để tạm dừng
            chat_history[session_id]["state"] = "stop_bot"
            chat_history[session_id]["collected_customer_info"] = {}
            return {"status": "success", "message": f"Bot cho session {session_id} đã được tạm dừng."}
        
        elif command == "start":
            # Kích hoạt lại bot
            if chat_history[session_id].get("state") == "stop_bot":
                chat_history[session_id]["state"] = None
                chat_history[session_id]["negativity_score"] = 0
                chat_history[session_id]["messages"].append({
                    "user": "[SYSTEM]",
                    "bot": "Bot đã được kích hoạt lại bởi quản trị viên."
                })
                return {"status": "success", "message": f"Bot cho session {session_id} đã được kích hoạt lại."}
            else:
                return {"status": "no_change", "message": f"Bot cho session {session_id} đã hoạt động."}
        
        else:
            raise HTTPException(status_code=400, detail="Command không hợp lệ. Chỉ chấp nhận 'start' hoặc 'stop'.")

async def human_chatting_endpoint(session_id: str):
    """
    Chuyển sang trạng thái human_chatting.
    """
    with chat_history_lock:
        if session_id not in chat_history:
            chat_history[session_id] = {
                "messages": [],
                "last_query": None,
                "offset": 0,
                "shown_product_keys": set(),
                "state": None,
                "pending_purchase_item": None,
                "negativity_score": 0,
                "handover_timestamp": None,
                "collected_customer_info": {}
            }
            message = f"Session {session_id} đã được tạo mới và chuyển sang trạng thái human_chatting."
            print(f"Đã tạo session mới: {session_id} thông qua human_chatting endpoint.")
        else:
            message = f"Bot cho session {session_id} đã chuyển sang trạng thái human_chatting."

        chat_history[session_id]["state"] = "human_chatting"
        chat_history[session_id]["handover_timestamp"] = time.time()
        return {"status": "success", "message": message}
 
def _handle_more_products(user_query: str, session_data: dict, history: list, model_choice: str, analysis: dict):
    last_query = session_data["last_query"]
    new_offset = session_data["offset"] + PAGE_SIZE

    retrieved_data = search_products(
        product_name=last_query["product_name"],
        category=last_query["category"],
        properties=last_query["properties"],
        offset=new_offset,
        strict_properties=True,
        strict_category=True
    )

    shown_keys = session_data["shown_product_keys"]
    new_products = [p for p in retrieved_data if _get_product_key(p) not in shown_keys]

    if not new_products:
        response_text = "Dạ, hết rồi ạ."
        session_data["offset"] = new_offset
        return response_text, [], []

    for p in new_products:
        shown_keys.add(_get_product_key(p))

    result = generate_llm_response(
        user_query, new_products, history, analysis["wants_specs"], model_choice, True, analysis["wants_images"]
    )
    
    product_images = []
    if analysis["wants_images"] and isinstance(result, dict):
        response_text = result["answer"].strip()
        product_images = result["product_images"]
        if response_text and product_images:
            response_text = "Dạ đây là hình ảnh sản phẩm em gửi anh/chị tham khảo ạ:\n" + response_text
    else:
        response_text = result

    session_data["offset"] = new_offset
    session_data["shown_product_keys"] = shown_keys
    return response_text, new_products, product_images

def _handle_new_query(user_query: str, session_data: dict, history: list, model_choice: str, analysis: dict):
    retrieved_data = []
    product_images = []

    if analysis["needs_search"]:
        search_params = analysis["search_params"]
        last_query = session_data.get("last_query") or {}
        product_name_to_search = search_params.get("product_name", user_query)
        if not product_name_to_search or product_name_to_search == user_query:
            product_name_to_search = last_query.get("product_name", product_name_to_search)
       
        retrieved_data = search_products(
            product_name=product_name_to_search,
            category=search_params.get("category", user_query),
            properties=search_params.get("properties", None),
            offset=0
        )

        history_text = format_history_text(history, limit=4)
        retrieved_data = filter_products_with_ai(user_query, history_text, retrieved_data)

        session_data["last_query"] = {
            "product_name": product_name_to_search,
            "category": search_params.get("category", user_query),
            "properties": search_params.get("properties", None)
        }
        session_data["offset"] = 0
        session_data["shown_product_keys"] = {_get_product_key(p) for p in retrieved_data}

    result = generate_llm_response(
        user_query, retrieved_data, history, analysis["wants_specs"], model_choice, analysis["needs_search"], analysis["wants_images"]
    )
    
    if analysis["wants_images"] and isinstance(result, dict):
        response_text = result["answer"].strip()
        product_images = result["product_images"]
        if response_text and product_images:
            response_text = "Dạ đây là hình ảnh sản phẩm em gửi anh/chị tham khảo ạ:\n" + response_text
    else:
        response_text = result

    return response_text, retrieved_data, product_images

def _update_chat_history(session_id: str, user_query: str, response_text: str, session_data: dict):
    with chat_history_lock:
        current_session = chat_history.get(session_id, {
            "messages": [], "last_query": None, "offset": 0, "shown_product_keys": set(), "state": None, "pending_purchase_item": None, "handover_timestamp": None, "negativity_score": 0, "collected_customer_info": {}
        })
        current_session["messages"].append({"user": user_query, "bot": response_text})
        current_session["last_query"] = session_data.get("last_query")
        current_session["offset"] = session_data.get("offset")
        current_session["shown_product_keys"] = session_data.get("shown_product_keys", set())
        current_session["state"] = session_data.get("state")
        current_session["pending_purchase_item"] = session_data.get("pending_purchase_item")
        current_session["negativity_score"] = session_data.get("negativity_score", 0)
        current_session["handover_timestamp"] = session_data.get("handover_timestamp")
        current_session["collected_customer_info"] = session_data.get("collected_customer_info", {})
        chat_history[session_id] = current_session

def _process_images(wants_images: bool, retrieved_data: list, product_images_names: list) -> list[ImageInfo]:
    images = []
    if not wants_images or not retrieved_data or not product_images_names:
        return images

    product_map = { f"{p.get('product_name', '')} ({p.get('properties', '')})": p for p in retrieved_data if p.get('product_name')}

    for name in product_images_names:
        product_data = product_map.get(name)
        if product_data:
            image_data = product_data.get('avatar_images')
            if not image_data:
                continue

            primary_image_url = None
            if isinstance(image_data, list) and image_data:
                for url in image_data:
                    if isinstance(url, str) and url.strip():
                        primary_image_url = url
                        break
            elif isinstance(image_data, str) and image_data.strip():
                primary_image_url = image_data

            if primary_image_url:
                images.append(ImageInfo(
                    product_name=product_data.get('product_name', ''),
                    image_url=primary_image_url,
                    product_link=str(product_data.get('link_product', ''))
                ))
    return images

async def power_off_bot_endpoint(request: ControlBotRequest):
    global bot_running
    command = request.command.lower()
    with bot_state_lock:
        if command == "stop":
            bot_running = False
            return {"status": "success", "message": "Bot đã được tạm dừng."}
        elif command == "start":
            bot_running = True
            return {"status": "success", "message": "Bot đã được kích hoạt lại."}
        elif command == "status":
            status_message = "Bot đang chạy" if bot_running else "Bot đã dừng"
            return {"status": "info", "message": status_message}
        else:
            raise HTTPException(status_code=400, detail="Invalid command. Use 'start' or 'stop'.")