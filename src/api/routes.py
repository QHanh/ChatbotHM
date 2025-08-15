from fastapi import HTTPException
from typing import Dict, Any, List, Set
import threading
import io
import requests
from PIL import Image
import google.generativeai as genai
from collections import defaultdict

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
            "collected_customer_info": {},
            "has_past_purchase": False,
            "pending_order": None # Thêm biến để theo dõi giỏ hàng
        }).copy()
        history = session_data["messages"][-8:].copy()

    if session_data.get("state") == "stop_bot":
        _update_chat_history(session_id, user_query, "", session_data)
        return ChatResponse(reply="", history=chat_history[session_id]["messages"].copy(), human_handover_required=False)

    if session_data.get("state") == "human_chatting":
        _update_chat_history(session_id, user_query, "", session_data)
        return ChatResponse(reply="", history=chat_history[session_id]["messages"].copy(), human_handover_required=False)
    
    if session_data.get("state") == "human_calling":
        response_text = "Dạ, nhân viên bên em đang vào ngay ạ, anh/chị vui lòng đợi trong giây lát."
        _update_chat_history(session_id, user_query, response_text, session_data)
        return ChatResponse(reply=response_text, history=chat_history[session_id]["messages"].copy(), human_handover_required=False)
 
    API_ENDPOINT = "https://embed.doiquanai.vn/embed"
    if image_url:
        print(f"Phát hiện hình ảnh từ URL: {image_url}, bắt đầu xử lý...")
        try:
            response = requests.post(API_ENDPOINT, data={"image_url": image_url}, timeout=15)
            response.raise_for_status()
            result = response.json()

            if "embedding" in result:
                embedding_vector = result["embedding"]
                print(" -> Tạo embedding cho ảnh thành công.")
            else:
                print(" -> Lỗi từ API:", result.get("error", "Không rõ lỗi"))

            retrieved_data = search_products_by_image(embedding_vector)
            if not retrieved_data:
                response_text="Dạ, em chưa nhận ra sản phẩm, anh/chị vui lòng cho em tên, thương hiệu hoặc model để tra cứu ạ."
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
    
    analysis_result = analyze_intent_and_extract_entities(user_query, history, model_choice)

    asking_for_more = is_asking_for_more(user_query)

    retrieved_data, product_images = [], []
    response_text = ""

    if user_query.strip().lower() == "/bot":
        session_data["state"] = None
        session_data["negativity_score"] = 0
        response_text = "Dạ, em có thể giúp gì tiếp cho anh/chị ạ?"
        _update_chat_history(session_id, user_query, response_text, session_data)
        return ChatResponse(reply=response_text, history=chat_history[session_id]["messages"].copy(), human_handover_required=False)

    if session_data.get("state") == "awaiting_purchase_confirmation":
        history_text = format_history_text(history, limit=4)
        evaluation = evaluate_purchase_confirmation(user_query, history_text, model_choice)
        decision = evaluation.get("decision")
        if decision == "CONFIRM":
            collected_info = session_data.get("collected_customer_info", {})
            pending_items = session_data.get("pending_purchase_item", [])
            
            if not pending_items:
                response_text = "Dạ có lỗi xảy ra, không tìm thấy sản phẩm cần xác nhận ạ."
                session_data["state"] = None
                _update_chat_history(session_id, user_query, response_text, session_data)
                return ChatResponse(reply=response_text, history=chat_history[session_id]["messages"].copy())

            if collected_info.get("name") and collected_info.get("phone") and collected_info.get("address"):
                purchase_items = []
                for item in pending_items:
                    item_data = item.get("evaluation", {}).get("product", {})
                    quantity = item.get("intent", {}).get("quantity", 1)
                    purchase_items.append(PurchaseItem(
                        product_name=item_data.get("product_name", "N/A"),
                        properties=item_data.get("properties"),
                        quantity=quantity
                    ))
                
                customer_info_obj = CustomerInfo(
                    name=collected_info.get("name"),
                    phone=collected_info.get("phone"),
                    address=collected_info.get("address"),
                    items=purchase_items
                )
                
                confirmed_names = [f"{item.quantity} x {item.product_name}" for item in purchase_items]
                response_text = f"Dạ em đã nhận được thông tin cho các sản phẩm: {', '.join(confirmed_names)}. Em sẽ tạo một đơn hàng mới cho mình ạ. Em cảm ơn anh/chị! /-heart"
                
                session_data["state"] = None
                session_data["pending_purchase_item"] = None
                session_data["has_past_purchase"] = True
                
                _update_chat_history(session_id, user_query, response_text, session_data)
                
                return ChatResponse(
                    reply=response_text,
                    history=chat_history[session_id]["messages"].copy(),
                    human_handover_required=False,
                    customer_info=customer_info_obj,
                    has_purchase=True
                )
            else:
                response_text = (
                    f"Dạ vâng ạ. Vậy để đặt đơn hàng, anh/chị có thể vào đường link sản phẩm để đặt hàng hoặc đến xem trực tiếp tại cửa hàng chúng em tại số 8 ngõ 117 Thái Hà, Đống Đa, Hà Nội (thời gian mở cửa từ 8h đến 18h).\n"
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
        if analysis_result.get("is_purchase_intent") or analysis_result.get("is_add_to_order_intent"):
            new_products_from_intent = analysis_result.get("search_params", {}).get("products", [])
            if new_products_from_intent:
                existing_order_items = session_data.get("pending_purchase_item", [])
                new_order_items = [{"intent": item, "status": "pending", "evaluation": None} for item in new_products_from_intent]
                
                session_data["pending_order"] = existing_order_items + new_order_items
                session_data["state"] = None
                session_data["pending_purchase_item"] = None
                
                # Fall through to re-process the purchase logic below
            else:
                # User wants to add, but didn't say what
                response_text = "Dạ vâng, anh/chị muốn thêm sản phẩm nào vào đơn hàng ạ?"
                _update_chat_history(session_id, user_query, response_text, session_data)
                return ChatResponse(reply=response_text, history=chat_history[session_id]["messages"].copy())
        else:
            current_info = session_data.get("collected_customer_info", {})
            extracted_info = extract_customer_info(user_query, model_choice)

            for key, value in extracted_info.items():
                if value and not current_info.get(key):
                    current_info[key] = value

            missing_info = []
            if not current_info.get("name"):
                missing_info.append("tên")
            if not current_info.get("phone"):
                missing_info.append("số điện thoại")
            if not current_info.get("address"):
                missing_info.append("địa chỉ")

            if missing_info:
                response_text = f"Dạ, anh/chị vui lòng cho em xin { ' và '.join(missing_info) } để em lên đơn ạ."
                session_data["collected_customer_info"] = current_info
                _update_chat_history(session_id, user_query, response_text, session_data)
                return ChatResponse(reply=response_text, history=chat_history[session_id]["messages"].copy(), human_handover_required=False)

            if not missing_info:
                pending_items = session_data.get("pending_purchase_item", [])
                if not pending_items:
                    response_text = "Dạ, anh chị đợi chút, em chưa tìm thấy sản phẩm để đặt hàng ạ. Nhân viên phụ trách bên em sẽ vào trả lời ngay ạ."
                    session_data["state"] = "human_calling"
                    session_data["handover_timestamp"] = time.time()
                    session_data["state"] = None
                    _update_chat_history(session_id, user_query, response_text, session_data)
                    return ChatResponse(reply=response_text, history=chat_history[session_id]["messages"].copy())

                purchase_items_obj = []
                for item in pending_items:
                    item_data = item.get("evaluation", {}).get("product", {})
                    quantity = item.get("intent", {}).get("quantity", 1)
                    props_value = item_data.get("properties")
                    final_props = None
                    if props_value is not None and str(props_value).strip() not in ['0', '']:
                        final_props = str(props_value)
                    
                    purchase_items_obj.append(PurchaseItem(
                        product_name=item_data.get("product_name", "N/A"),
                        properties=final_props,
                        quantity=quantity
                    ))

                customer_info_obj = CustomerInfo(
                    name=current_info.get("name"),
                    phone=current_info.get("phone"),
                    address=current_info.get("address"),
                    items=purchase_items_obj
                )
                
                response_text = "Dạ em đã nhận được đầy đủ thông tin. Em cảm ơn anh/chị! /-heart"
                session_data["state"] = None
                session_data["pending_purchase_item"] = None
                session_data["has_past_purchase"] = True
                
                _update_chat_history(session_id, user_query, response_text, session_data)
                
                return ChatResponse(
                    reply=response_text,
                    history=chat_history[session_id]["messages"].copy(),
                    customer_info=customer_info_obj,
                    has_purchase=True,
                    human_handover_required=False
                )

    retrieved_data, product_images = [], []
    response_text = ""

    if analysis_result.get("is_add_to_order_intent"):
        response_text = "Dạ vâng, anh/chị muốn mua thêm sản phẩm nào ạ?"
        session_data["last_query"] = None

    if analysis_result.get("is_bank_transfer"):
        response_text = "Dạ, anh/chị đợi chút, nhân viên bên em sẽ vào ngay ạ."
        session_data["state"] = "human_calling"
        session_data["handover_timestamp"] = time.time()
        _update_chat_history(session_id, user_query, response_text, session_data)
        return ChatResponse(
            reply=response_text,
            history=chat_history[session_id]["messages"].copy(),
            human_handover_required=True,
            has_negativity=False
        )

    if analysis_result.get("is_negative"):
        session_data["negativity_score"] += 1
        if session_data["negativity_score"] >= 3:
            response_text = "Em đã báo nhân viên phụ trách, anh/chị vui lòng đợi để được hỗ trợ ngay ạ."
            session_data["state"] = "human_calling"
            session_data["handover_timestamp"] = time.time()
            session_data["negativity_score"] = 0
            _update_chat_history(session_id, user_query, response_text, session_data)
            
            return ChatResponse(
                reply=response_text,
                history=chat_history[session_id]["messages"].copy(),
                human_handover_required=False,
                has_negativity=True
            )

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
    
    if analysis_result.get("wants_warranty_service"):
        if session_data.get("has_past_purchase"):
            response_text = "Dạ anh/chị đợi chút, nhân viên phụ trách bảo hành bên em sẽ vào trả lời ngay ạ."
            session_data["state"] = "human_calling"
            session_data["handover_timestamp"] = time.time()
            _update_chat_history(session_id, user_query, response_text, session_data)
            return ChatResponse(
                reply=response_text,
                history=chat_history[session_id]["messages"].copy(),
                human_handover_required=True,
                has_negativity=False
            )

        response_text = "Dạ anh/chị đợi chút, nhân viên phụ trách bảo hành bên em sẽ vào trả lời ngay ạ."
        session_data["state"] = "human_calling"
        session_data["handover_timestamp"] = time.time()
        _update_chat_history(session_id, user_query, response_text, session_data)
        return ChatResponse(
            reply=response_text,
            history=chat_history[session_id]["messages"].copy(),
            human_handover_required=True,
            has_negativity=False
        )
    
    if analysis_result.get("wants_human_agent"):
        response_text = "Em đã báo nhân viên phụ trách, anh/chị vui lòng đợi để được hỗ trợ ngay ạ."
        session_data["state"] = "human_calling"
        session_data["handover_timestamp"] = time.time()
        
        _update_chat_history(session_id, user_query, response_text, session_data)
        
        return ChatResponse(
            reply=response_text,
            history=chat_history[session_id]["messages"].copy(),
            human_handover_required=True,
            has_negativity=False
        )

    if analysis_result.get("is_purchase_intent"):
        

        if "pending_order" not in session_data or session_data["pending_order"] is None:
            products_from_intent = analysis_result.get("search_params", {}).get("products", [])
            if products_from_intent:
                session_data["pending_order"] = [
                    {"intent": item, "status": "pending", "evaluation": None}
                    for item in products_from_intent
                ]

        if "pending_order" in session_data and session_data["pending_order"]:
            history_text = format_history_text(history, limit=6)
            
            for item in session_data["pending_order"]:
                if item["status"] != "confirmed":
                    item_intent = item["intent"]
                    product_name_intent = item_intent.get("product_name")
                    properties_intent = item_intent.get("properties")
                    
                    # Tạo sub_query để AI tập trung vào từng sản phẩm
                    sub_query = f"khách muốn mua {item_intent.get('quantity', 1)} {product_name_intent}"
                    if properties_intent:
                        sub_query += f" loại {properties_intent}"

                    # Xác định query phù hợp để đánh giá
                    query_for_evaluation = user_query
                    # Nếu bot đã từng gợi ý và đây không phải lượt đầu tiên, dùng user_query để hiểu sự đồng ý
                    if not (item.get("evaluation") and item["evaluation"].get("type") == "CLOSE_MATCH"):
                         query_for_evaluation = sub_query


                    best_evaluation = None
                    MAX_SEARCH_PAGES = 5 
                    for page in range(MAX_SEARCH_PAGES):
                        found_products = search_products(
                            product_name=product_name_intent,
                            category=item_intent.get("category"),
                            properties=properties_intent,
                            offset=page * PAGE_SIZE
                        )
                        
                        previous_suggestion = None
                        if item.get("evaluation") and item["evaluation"].get("type") == "CLOSE_MATCH":
                            previous_suggestion = item["evaluation"].get("product")

                        if previous_suggestion:
                            suggestion_key = _get_product_key(previous_suggestion)
                            if not found_products or not any(_get_product_key(p) == suggestion_key for p in found_products):
                                found_products = [previous_suggestion] + (found_products or [])
                        
                        if not found_products and page > 0: break

                        # FIX: Pass the actual user_query to the evaluation function
                        current_evaluation = evaluate_and_choose_product(
                            query_for_evaluation, history_text, found_products, model_choice
                        )

                        if current_evaluation.get("type") == "PERFECT_MATCH":
                            best_evaluation = current_evaluation
                            break
                        
                        if not best_evaluation or current_evaluation.get("score", 0.0) > best_evaluation.get("score", 0.0):
                            best_evaluation = current_evaluation
                        
                        if best_evaluation and best_evaluation.get("score", 0.0) >= 0.8:
                            break
                        
                        if not found_products: break
                    
                    item["evaluation"] = best_evaluation if best_evaluation else {"type": "NO_MATCH"}
                    
                    if item["evaluation"].get("type") == "PERFECT_MATCH":
                        product_data = item["evaluation"]["product"]
                        requested_quantity = item["intent"].get("quantity", 1)
                        try:
                            stock_quantity = int(product_data.get("inventory", 0))
                        except (ValueError, TypeError):
                            stock_quantity = 0

                        if stock_quantity <= 0:
                            item["status"] = "failed"
                            item["failure_reason"] = "out_of_stock"
                        elif stock_quantity < requested_quantity:
                            item["status"] = "failed"
                            item["failure_reason"] = "insufficient_stock"
                        else:
                            item["status"] = "confirmed"
                    else:
                        item["status"] = "failed"

            confirmed_items = [item for item in session_data["pending_order"] if item["status"] == "confirmed"]
            failed_items_list = [item for item in session_data["pending_order"] if item["status"] == "failed"]

            response_parts = []

            if confirmed_items:
                confirmed_names = [
                    f"{item['intent'].get('quantity', 1)} x {item['evaluation']['product'].get('product_name')}"
                    + (f" ({str(props).lower()})" if (props := item['evaluation']['product'].get('properties', 'N/A')) not in [0, '0', None, '', 'N/A'] else '')
                    for item in confirmed_items
                ]
                response_parts.append(f"Dạ, em xác nhận các sản phẩm: {', '.join(confirmed_names)}.\n")

            if failed_items_list:
                not_found_items = [item for item in failed_items_list if item['evaluation'].get('type') == 'NO_MATCH']
                close_match_items = [item for item in failed_items_list if item['evaluation'].get('type') == 'CLOSE_MATCH']
                out_of_stock_items = [item for item in failed_items_list if item.get('failure_reason') == 'out_of_stock']
                insufficient_stock_items = [item for item in failed_items_list if item.get('failure_reason') == 'insufficient_stock']

                if out_of_stock_items:
                    product_names = [item['evaluation']['product'].get('product_name') + (f" ({str(props).lower()})" if (props := item['evaluation']['product'].get('properties', 'N/A')) not in [0, '0', None, '', 'N/A'] else '') for item in out_of_stock_items]
                    response_parts.append(f"Dạ em rất tiếc, các sản phẩm này hiện đang hết hàng rồi ạ: {', '.join(product_names)}.")

                if insufficient_stock_items:
                    messages = []
                    for item in insufficient_stock_items:
                        product_data = item['evaluation']['product']
                        stock = product_data.get('inventory', 0)
                        messages.append(f"{product_data.get('product_name')}" + (f" ({str(props).lower()})" if (props := product_data.get('properties', 'N/A')) not in [0, '0', None, '', 'N/A'] else '') + f" (chỉ còn {stock} sản phẩm)")
                    response_parts.append(f"Dạ, số lượng một số sản phẩm trong kho không đủ ạ: {'; '.join(messages)}. Anh/chị có muốn lấy số lượng này không ạ?")

                if not_found_items:
                    grouped_failures = defaultdict(list)
                    for item in not_found_items:
                        product_name = item.get('intent', {}).get('product_name', 'Sản phẩm không xác định')
                        properties = item.get('intent', {}).get('properties')
                        grouped_failures[product_name].append(properties)
                    
                    failure_messages = []
                    for name, props in grouped_failures.items():
                        clean_props = [p for p in props if p]
                        if clean_props:
                            failure_messages.append(f"{name} (các loại: {', '.join(clean_props)})")
                        else:
                            failure_messages.append(name)
                    response_parts.append(f"Em chưa tìm thấy các sản phẩm: {'; '.join(failure_messages)}.")

                if close_match_items:
                    suggestion_messages = []
                    for item in close_match_items:
                        eval_data = item['evaluation']
                        suggested_prod = eval_data['product']
                        props = suggested_prod.get('properties', 'N/A')
                        full_name = f"{suggested_prod.get('product_name')}" + (f" ({str(props).lower()})" if (props := suggested_prod.get('properties', 'N/A')) not in [0, '0', None, '', 'N/A'] else '')
                        # suggestion_messages.append(f"  - {full_name} - {eval_data['reason']}")
                        suggestion_messages.append(f"  - {full_name}")
                    response_parts.append(f"Em tìm thấy một số sản phẩm gần giống anh chị nói, anh/chị xem có phải không ạ:\n{'\n'.join(suggestion_messages)}")


            if not failed_items_list and confirmed_items:
                session_data["state"] = "awaiting_purchase_confirmation"
                
                session_data["pending_purchase_item"] = confirmed_items
                response_parts.append("Anh/chị có muốn em lên đơn cho những sản phẩm này không ạ?")
                session_data["pending_order"] = None
            
            response_text = " ".join(response_parts)

        else:
            response_text = "Dạ, anh/chị muốn mua sản phẩm nào ạ?"

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

    action_data = None
    is_general_query = not analysis_result.get("is_purchase_intent") and session_data.get("state") is None
    if is_general_query and len(retrieved_data) == 1:
        product = retrieved_data[0]
        product_link = product.get("link_product")
        if product_link and isinstance(product_link, str) and product_link.startswith("http"):
            action_data = {"action": "redirect", "url": product_link}


    return ChatResponse(
        reply=response_text,
        history=chat_history.get(session_id, {}).get("messages", []).copy(),
        images=images,
        has_images=len(images) > 0,
        has_purchase=analysis_result.get("is_purchase_intent", False),
        human_handover_required=analysis_result.get("human_handover_required", False),
        has_negativity=False,
        action_data=action_data
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
                "collected_customer_info": {},
                "has_past_purchase": False,
                "pending_order": None # Thêm biến để theo dõi giỏ hàng
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
                "collected_customer_info": {},
                "has_past_purchase": False,
                "pending_order": None # Thêm biến để theo dõi giỏ hàng
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

    history_text = format_history_text(history, limit=6)
    retrieved_data = filter_products_with_ai(user_query, history_text, retrieved_data)
    
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
        search_params = analysis.get("search_params", {})
        products_list = search_params.get("products", [])
        
        # Logic mới: Lấy thông tin từ sản phẩm đầu tiên trong list
        if products_list:
            first_product = products_list[0]
            product_name_to_search = first_product.get("product_name", user_query)
            category_to_search = first_product.get("category", user_query)
            properties_to_search = first_product.get("properties")

            retrieved_data = search_products(
                product_name=product_name_to_search,
                category=category_to_search,
                properties=properties_to_search,
                offset=0
            )

            history_text = format_history_text(history, limit=6)
            retrieved_data = filter_products_with_ai(user_query, history_text, retrieved_data)

            # Cập nhật last_query theo cấu trúc cũ để _handle_more_products hoạt động
            session_data["last_query"] = {
                "product_name": product_name_to_search,
                "category": category_to_search,
                "properties": properties_to_search
            }
            session_data["offset"] = 0
            session_data["shown_product_keys"] = {_get_product_key(p) for p in retrieved_data}
        else:
            # Fallback nếu không có sản phẩm nào được intent parser trả về
            session_data["last_query"] = None
            session_data["offset"] = 0
            session_data["shown_product_keys"] = set()


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
            "messages": [], "last_query": None, "offset": 0, "shown_product_keys": set(), "state": None, "pending_purchase_item": None, "handover_timestamp": None, "negativity_score": 0, "collected_customer_info": {}, "pending_order": None
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
        current_session["has_past_purchase"] = session_data.get("has_past_purchase", False)
        current_session["pending_order"] = session_data.get("pending_order")
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