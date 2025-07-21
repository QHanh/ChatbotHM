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
    - Nếu câu hỏi của khách hàng quá ngắn, là một lời chào, lời cảm ơn, hoặc không rõ ràng về sản phẩm (ví dụ: "ok", "thanks", "ho", "hi", "uk"), hãy đặt "needs_search" là "false".
    Lịch sử hội thoại gần đây:
    {history_text}

    Câu hỏi mới nhất của khách hàng: "{user_query}"

    Hãy phân tích và điền vào cấu trúc JSON sau:
    {{
      "needs_search": <true nếu cần tìm kiếm thông tin sản phẩm để trả lời, ngược lại false>,
      "wants_images": <true nếu khách hỏi về "ảnh", "hình ảnh", ngược lại false>,
      "wants_specs": <true nếu khách hỏi về "thông số", "chi tiết", "cấu hình", ngược lại false>,
      "search_params": {{
        "product_name": "<Tên sản phẩm khách hàng đang đề cập bao gồm luôn cả tên phụ kiện đi kèm>",
        "category": "<Danh mục sản phẩm. Quy tắc: Nếu khách hỏi 'đèn kính hiển vi', category là 'đèn'. Nếu khách hỏi 'kính hiển vi', category là 'kính hiển vi'. Nếu khách hỏi 'kính hiển vi 2 mắt', category là 'kính hiển vi 2 mắt'. Nếu không thể xác định, hãy để category giống product_name.>",
        "properties": "<Các thuộc tính cụ thể như model, màu sắc...>"
      }}
    }}

    Ví dụ:
    - Câu hỏi: "shop có đèn kính hiển vi không"
      JSON: {{"needs_search": true, "wants_images": false, "wants_specs": false, "search_params": {{"product_name": "đèn kính hiển vi", "category": "đèn", "properties": ""}}}}

    - Câu hỏi: "shop có kính hiển vi không"
      JSON: {{"needs_search": true, "wants_images": false, "wants_specs": false, "search_params": {{"product_name": "kính hiển vi", "category": "kính hiển vi", "properties": ""}}}}

    - Câu hỏi: "shop có kính hiển vi 2 mắt màu xanh không"
      JSON: {{"needs_search": true, "wants_images": false, "wants_specs": false, "search_params": {{"product_name": "kính hiển vi 2 mắt", "category": "kính hiển vi 2 mắt", "properties": "màu xanh"}}}}
  
    - Câu hỏi: "cho xem ảnh máy khò kaisi model 8512p"
      JSON: {{"needs_search": true, "wants_images": true, "wants_specs": false, "search_params": {{"product_name": "máy khò kaisi", "category": "Máy khò", "properties": "MODEL:8512P"}}}}

    - Câu hỏi: "có máy hàn dùng mũi C210 không"
      JSON: {{"needs_search": true, "wants_images": false, "wants_specs": false, "search_params": {{"product_name": "máy hàn dùng mũi C210", "category": "Máy hàn", "properties": ""}}}}

    JSON của bạn:
    """

    fallback_response = {
        "needs_search": True,
        "wants_images": "ảnh" in user_query.lower() or "hình" in user_query.lower(),
        "wants_specs": "thông số" in user_query.lower(),
        "search_params": { "product_name": user_query, "category": user_query, "properties": "" }
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