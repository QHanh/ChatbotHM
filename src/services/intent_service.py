import json
import re
from typing import Dict, Any

from src.services.llm_service import get_gemini_model, get_lmstudio_response, get_openai_model

def analyze_intent_and_extract_entities(user_query: str, history: list = None, model_choice: str = "gemini") -> Dict[str, Any]:
    """
    Sử dụng một lệnh gọi LLM duy nhất để phân tích ý định của người dùng và trích xuất các thực thể cần thiết.
    """
    history_text = ""
    if history:
        for turn in history[-5:]:
            history_text += f"Khách: {turn['user']}\nBot: {turn['bot']}\n"

    # GỢI Ý: Đã tích hợp logic và ví dụ về category của bạn vào prompt này.
    prompt = f"""
    Bạn là một AI phân tích truy vấn của khách hàng. Dựa vào lịch sử hội thoại và câu hỏi mới nhất, hãy phân tích và trả về một đối tượng JSON.
    QUAN TRỌNG: 
    - Khi câu hỏi của khách hàng là một câu trả lời ngắn gọn cho câu hỏi của bot ở lượt trước, hãy kế thừa ý định từ lượt trước đó.
    - **Kế thừa ý định:** Khi câu hỏi của khách hàng là một câu trả lời ngắn gọn cho câu hỏi của bot ở lượt trước (ví dụ: bot hỏi 'muốn xem loại nào', 'muốn xem ảnh loại nào?', khách trả lời 'tất cả' hoặc 'gửi đi', 'ok'), hãy kế thừa ý định từ lượt trước đó. Nếu ý định trước đó cần tìm kiếm, thì `needs_search` phải là `true` và các `search_params` phải được suy ra từ ngữ cảnh.
    - Nếu câu hỏi của khách hàng quá ngắn, là một lời chào, lời cảm ơn, hoặc không rõ ràng về sản phẩm (ví dụ: "ok", "thanks", "ho", "hi", "uk"), hãy đặt "needs_search" là "false".
    - **Ưu tiên ý định thông tin:** Nếu khách hàng hỏi xin "ảnh", "thông số", thì `is_purchase_intent` PHẢI là `false`.
    - **Ý định mua hàng (`is_purchase_intent`=true):** Chỉ xác định là mua hàng khi khách hàng dùng các từ dứt khoát như "chốt đơn", "lấy cho anh cái này", "đặt mua" và **KHÔNG** đi kèm với yêu cầu xin thông tin. Nếu không nói gì, số lượng mặc định là 1. Khách hàng muốn **hỏi giá hay báo giá** thì is_purchase_intent là `false`.
    - Hãy trích xuất cả số lượng đặt hàng (`quantity`) nếu khách hàng đề cập. Nếu không nói gì, số lượng mặc định là 1.
    - **Phân tích thái độ:** Nếu khách hàng thể hiện sự bực bội, chê bai, phàn nàn, hoặc dùng từ ngữ tiêu cực, hãy đặt `is_negative` là `true`.
    - **Ý định thêm đơn hàng (`is_add_to_order_intent`):** Chỉ là `true` khi khách hàng nói rõ ràng muốn "bổ sung đơn", "thêm đơn hàng". Các câu hỏi như "còn nữa không", "còn nữa chứ" KHÔNG phải là ý định này.
    - **Quy tắc ưu tiên:** `is_add_to_order_intent` và `is_purchase_intent` không thể cùng là `true`. `is_add_to_order_intent` chỉ đúng cho các câu hỏi ban đầu như "mua thêm", "bổ sung đơn". Khi khách hàng đã chỉ định một sản phẩm cụ thể để mua, `is_purchase_intent` sẽ là `true` và `is_add_to_order_intent` phải là `false`.
    - **Ý định muốn biết thông tin cửa hàng:** Nếu khách hàng hỏi về địa chỉ, giờ làm việc, hoặc muốn đến mua trực tiếp, hãy đặt `wants_store_info` là `true`.
    - **Ý định gặp người thật (chat):** Chỉ đặt `wants_human_agent` là `true` khi khách hàng muốn nói chuyện, chat với nhân viên, người thật.
    - **Phân biệt:** "mua trực tiếp" là `wants_store_info`, "tư vấn trực tiếp" là `wants_human_agent`. Lưu ý: `wants_human_agent` là `true` thì `wants_store_info` phải là `false` và ngược lại.
    
    Lịch sử hội thoại gần đây:
    {history_text}

    Câu hỏi mới nhất của khách hàng: "{user_query}"

    Hãy phân tích và điền vào cấu trúc JSON sau:
    {{
      "needs_search": <true nếu cần tìm kiếm thông tin sản phẩm gồm cả giá, ảnh để trả lời, ngược lại false>,
      "is_purchase_intent": <true nếu khách muốn mua/chốt đơn, ví dụ: "cho mình loại này", "chốt đơn", "lấy cho mình cái này", ngược lại false>,
      "is_add_to_order_intent": <true nếu khách muốn mua thêm/thêm đơn, ngược lại false>,
      "wants_images": <true nếu khách hỏi về "ảnh", "hình ảnh", ngược lại false>,
      "wants_specs": <true nếu khách hỏi về "thông số", "chi tiết", "cấu hình", "xuất xứ", "nơi sản xuất", "khách hàng muốn so sánh các sản phẩm", ngược lại false>,
      "wants_human_agent": <true nếu khách muốn gặp người thật, ngược lại false>,
      "wants_store_info": <true nếu khách muốn biết địa chỉ, thời gian làm việc hoặc số hotline của cửa hàng>,
      "is_negative": <true nếu khách hàng có thái độ tiêu cực, ngược lại false>,
      "search_params": {{
        "product_name": "<Tên sản phẩm khách hàng đang đề cập bao gồm luôn cả tên thương hiệu và tên phụ kiện đi kèm>",
        "category": "<Danh mục sản phẩm. Quy tắc: Nếu khách hỏi 'đèn kính hiển vi', category là 'đèn'. Nếu khách hỏi 'kính hiển vi', category là 'kính hiển vi'. Nếu khách hỏi 'kính hiển vi 2 mắt', category là 'kính hiển vi 2 mắt'. Nếu không thể xác định, hãy để category giống product_name.>",
        "properties": "<Các thuộc tính cụ thể như model, màu sắc, loại, combo,... Lưu ý: Tên thương hiệu không phải thuộc tính, ví dụ: máy hàn GVM T210S, GVM H3 thì properties là ''(**không có thuộc tính**). Thuộc tính **chỉ có** khi khách đề cập rõ màu sắc, MODEL, hoặc loại cụ thể.>",
        "quantity": <Số lượng, mặc định là 1>
      }}
    }}

    Ví dụ:
    - Câu hỏi: "shop có đèn kính hiển vi không"
      JSON: {{"needs_search": true, "is_purchase_intent": false, "is_add_to_order_intent": false, "wants_images": false, "wants_specs": false, "wants_human_agent": false, "is_negative": false, "search_params": {{"product_name": "đèn kính hiển vi", "category": "đèn", "properties": "", "quantity": 1}}}}

    - Câu hỏi: "shop có kính hiển vi 2 mắt màu xanh không"
      JSON: {{"needs_search": true, "is_purchase_intent": false, "is_add_to_order_intent": false, "wants_images": false, "wants_specs": false, "wants_human_agent": false, "is_negative": false, "search_params": {{"product_name": "kính hiển vi 2 mắt", "category": "kính hiển vi 2 mắt", "properties": "màu xanh", "quantity": 1}}}}
  
    - Câu hỏi: "cho xem ảnh máy khò kaisi model 8512p"
      JSON: {{"needs_search": true, "is_purchase_intent": false, "is_add_to_order_intent": false, "wants_images": true, "wants_specs": false, "wants_human_agent": false, "is_negative": false, "search_params": {{"product_name": "máy khò kaisi", "category": "Máy khò", "properties": "MODEL:8512P", "quantity": 1}}}}

    - Câu hỏi: "có máy hàn dùng mũi C210 không"
      JSON: {{"needs_search": true, "is_purchase_intent": false, "is_add_to_order_intent": false, "wants_images": false, "wants_specs": false, "wants_human_agent": false, "is_negative": false, "search_params": {{"product_name": "máy hàn dùng mũi C210", "category": "Máy hàn", "properties": "", "quantity": 1}}}}

    - Câu hỏi: "cho mình xin ảnh cái máy hàn GVM T210S và máy hàn GVM H3"
      JSON: {{"needs_search": true, "is_purchase_intent": false, "is_add_to_order_intent": false, "wants_images": true, "wants_specs": false, "wants_human_agent": false, "is_negative": false, "search_params": {{"product_name": "máy hàn GVM T210s H3", "category": "máy hàn", "properties": "", "quantity": 1}}}}
    
    - Câu hỏi: "cho chị loại M6T màu xanh nhé"
      JSON: {{"needs_search": false, "is_purchase_intent": true, "is_add_to_order_intent": false, "wants_images": false, "wants_specs": false, "wants_human_agent": false, "is_negative": false, "search_params": {{"product_name": "kính hiển vi M6T", "category": "kính hiển vi", "properties": "màu xanh", "quantity": 1 }}}}

    - Câu hỏi: "cho tôi gặp anh Hoàng"
      JSON: {{"needs_search": false, "is_purchase_intent": false, "is_add_to_order_intent": false, "wants_images": false, "wants_specs": false, "wants_human_agent": true, "is_negative": false, "search_params": {{...}} }}
    
    - Câu hỏi: "tôi muốn mua trực tiếp sản phẩm"
      JSON: {{"needs_search": false, "is_purchase_intent": false, "is_add_to_order_intent": false, "wants_images": false, "wants_specs": false, "wants_human_agent": false, "wants_store_info": true, "search_params": {{...}} }}
    
    - Câu hỏi: "bot trả lời ngu thế"
      JSON: {{"needs_search": false, "is_purchase_intent": false, "is_add_to_order_intent": false, "wants_images": false, "wants_specs": false, "wants_human_agent": false, "is_negative": true, "search_params": {{...}} }}

    - Câu hỏi: "tôi muốn thêm đơn", "tôi muốn mua thêm", "tôi muốn bổ sung đơn hàng"
      JSON: {{"needs_search": false, "is_purchase_intent": false, "is_add_to_order_intent": true, "wants_images": false, "wants_specs": false, "wants_human_agent": false, "is_negative": false, "search_params": {{...}} }}

    - Bối cảnh: Bot vừa hỏi "Dạ, mình muốn xem ảnh của loại tô vít 2UUL nào ạ?". Khách trả lời: "Tất cả"
      JSON: {{"needs_search": true, "is_purchase_intent": false, "is_add_to_order_intent": false, "wants_images": true, "wants_specs": false, "wants_human_agent": false, "is_negative": false, "search_params": {{"product_name": "tô vít 2UUL", "category": "tô vít", "properties": ""}}}}

    JSON của bạn:
    """

    fallback_response = {
        "needs_search": True,
        "is_purchase_intent": False,
        "is_add_to_order_intent": False,
        "wants_images": "ảnh" in user_query.lower(),
        "wants_specs": "thông số" in user_query.lower(),
        "wants_human_agent": False,
        "wants_store_info": False,
        "is_negative": False, 
        "search_params": { "product_name": user_query, "category": user_query, "properties": "", "quantity": 1 }
    }

    response_text = None
    try:
        if model_choice == "gemini":
            model = get_gemini_model()
            if model:
                response = model.generate_content(prompt)
                response_text = response.text
        elif model_choice == "lmstudio":
            response_text = get_lmstudio_response(prompt)
        elif model_choice == "openai":
            openai = get_openai_model()
            if openai:
                completion = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                response_text = completion.choices[0].message.content
        else:
            return fallback_response

        if not response_text:
            return fallback_response

        print("--- PHÂN TÍCH Ý ĐỊNH & THỰC THỂ ---")
        print(f"Phản hồi thô từ LLM: {response_text}")

        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            cleaned_response = json_match.group(0)
            data = json.loads(cleaned_response)
            if 'needs_search' in data and 'wants_images' in data and 'wants_specs' in data and 'search_params' in data:
                print(f"Kết quả phân tích: {data}")
                print("-----------------------------------")
                return data
        
        print("Không thể parse JSON từ phản hồi LLM, sử dụng fallback.")
        return fallback_response

    except Exception as e:
        print(f"Lỗi trong quá trình phân tích ý định bằng LLM ({model_choice}): {e}")
        return fallback_response
    
def extract_customer_info(user_input: str, model_choice: str = "gemini") -> Dict:
    """
    Sử dụng LLM để bóc tách Tên, SĐT, Địa chỉ từ một chuỗi văn bản.
    """
    prompt = f"""
    Bạn là một AI chuyên bóc tách thông tin. Từ đoạn văn bản dưới đây, hãy trích xuất Tên người (`name`), Số điện thoại (`phone`), và Địa chỉ (`address`) vào một đối tượng JSON.
    Nếu không tìm thấy thông tin nào, hãy để giá trị là null. Chỉ trả về JSON.

    Văn bản: "{user_input}"

    JSON:
    """
    try:
        model = get_gemini_model()
        if model:
            response = model.generate_content(prompt)
            json_text = re.search(r'\{.*\}', response.text, re.DOTALL).group(0)
            return json.loads(json_text)
        return {}
    except Exception as e:
        print(f"Lỗi khi bóc tách thông tin khách hàng: {e}")
        return {}