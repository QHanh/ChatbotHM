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

    # GỢI Ý: Thêm chỉ dẫn và ví dụ về cách xử lý các câu trả lời ngắn, nối tiếp trong hội thoại.
    prompt = f"""
    Bạn là một AI phân tích truy vấn của khách hàng. Dựa vào lịch sử hội thoại và câu hỏi mới nhất, hãy phân tích và trả về một đối tượng JSON.
    QUAN TRỌNG: Khi câu hỏi của khách hàng là một câu trả lời ngắn gọn cho câu hỏi của bot ở lượt trước (ví dụ: bot hỏi chọn A hay B, khách trả lời "A"), hãy kế thừa ý định từ lượt trước đó. Nếu ý định trước đó cần thông tin sản phẩm, thì "needs_search" phải là "true".

    Lịch sử hội thoại gần đây:
    {history_text}

    Câu hỏi mới nhất của khách hàng: "{user_query}"

    Hãy phân tích và điền vào cấu trúc JSON sau:
    {{
      "needs_search": <true nếu cần tìm kiếm thông tin sản phẩm để trả lời, ngược lại false>,
      "wants_images": <true nếu khách hỏi về "ảnh", "hình ảnh", ngược lại false>,
      "wants_specs": <true nếu khách hỏi về "thông số", "chi tiết", "cấu hình", ngược lại false>,
      "search_params": {{
        "product_name": "<Tên chính của sản phẩm>",
        "category": "<Danh mục chung của sản phẩm>",
        "properties": "<Các thuộc tính cụ thể như model, màu sắc...>"
      }}
    }}

    Ví dụ:
    - Bối cảnh: Bot vừa hỏi "Anh/chị muốn biết thông số của model 8512P hay 8512D ạ?". Khách trả lời: "8512P"
      JSON: {{"needs_search": true, "wants_images": false, "wants_specs": true, "search_params": {{"product_name": "máy khò hàn", "category": "Máy khò", "properties": "8512P"}}}}

    - Câu hỏi: "shop có bán iphone 15 pro max màu xanh không?"
      JSON: {{"needs_search": true, "wants_images": false, "wants_specs": false, "search_params": {{"product_name": "iphone 15 pro max", "category": "điện thoại", "properties": "màu xanh"}}}}

    - Câu hỏi: "cho xem ảnh máy khò kaisi model 8512p"
      JSON: {{"needs_search": true, "wants_images": true, "wants_specs": false, "search_params": {{"product_name": "máy khò kaisi", "category": "Máy khò", "properties": "MODEL:8512P"}}}}

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