from typing import List, Dict
from src.services.llm_service import get_gemini_model, get_lmstudio_response, get_openai_model
from src.utils.helpers import is_general_query, format_history_text

def generate_llm_response(
    user_query: str, 
    search_results: list, 
    history: list = None, 
    include_specs: bool = False, 
    model_choice: str = "gemini", 
    needs_product_search: bool = True,
    wants_images: bool = False
) -> str:
    """
    Tạo prompt và gọi đến LLM để sinh câu trả lời.
    """
    if is_general_query(user_query):
        if not search_results:
            return {"answer": "Dạ, cửa hàng em chưa có sản phẩm nào để giới thiệu ạ.", "product_images": []} if wants_images else "Dạ, cửa hàng em chưa có sản phẩm nào để giới thiệu ạ."
        product_names = [item.get('product_name', 'N/A') for item in search_results]
        answer = (
            "Hiện tại cửa hàng em đang kinh doanh nhiều loại sản phẩm về các thiết bị điện tử, ví dụ như: "
            + ".\n".join(product_names) + " và nhiều sản phẩm khác nữa"
            + ".\n\nAnh/chị muốn tìm hiểu thêm về sản phẩm nào không ạ?"
        )
        return {"answer": answer, "product_images": []} if wants_images else answer

    # Chuẩn bị context
    context = ""
    if history:
        context += f"Lịch sử hội thoại gần đây:\n{format_history_text(history)}\n"
    
    if needs_product_search:
        if not search_results:
            return {"answer": "Dạ, em xin lỗi, cửa hàng em chưa kinh doanh sản phẩm này ạ.", "product_images": []} if wants_images else "Dạ, em xin lỗi, cửa hàng em chưa kinh doanh sản phẩm này ạ."
        
        context += _build_product_context(search_results, include_specs)

    # Nếu wants_images, truyền danh sách sản phẩm vào prompt
    product_infos = [
        f"{p.get('product_name', '')} ({p.get('properties', '')})"
        for p in search_results if p.get('product_name')
    ] if wants_images else []

    prompt = _build_prompt(user_query, context, needs_product_search, wants_images, product_infos)
    
    print("--- PROMPT GỬI ĐẾN LLM ---")
    print(prompt)
    print("--------------------------")

    llm_response = None
    try:
        if model_choice == "gemini":
            model = get_gemini_model()
            if model:
                response = model.generate_content(prompt)
                llm_response = response.text.strip()
        elif model_choice == "lmstudio":
            llm_response = get_lmstudio_response(prompt)
        elif model_choice == "openai":
            openai = get_openai_model()
            if not openai:
                return {"answer": "Không tìm thấy OpenAI API key.", "product_images": []} if wants_images else "Không tìm thấy OpenAI API key."
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4000
            )
            llm_response = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Lỗi khi gọi LLM: {e}")
        llm_response = None

    if wants_images:
        answer, product_images = _parse_answer_and_images(llm_response, product_infos)
        return {"answer": answer, "product_images": product_images}
    else:
        if llm_response:
            return llm_response
        return _get_fallback_response(search_results, needs_product_search)

def _build_product_context(search_results: List[Dict], include_specs: bool = False) -> str:
    """Xây dựng context thông tin sản phẩm."""
    product_context = "Thông tin sản phẩm tìm thấy:\n"
    for item in search_results:
        product_context += (
            f"Tên: {item.get('product_name', 'N/A')}, "
            f"Danh mục: {item.get('category', 'N/A')}, "
            f"Thuộc tính: {item.get('properties', 'N/A')}, "
            f"Thương hiệu: {item.get('trademark', 'N/A')}, "
            f"Chính sách bảo hành: {item.get('guarantee', 'N/A')}, "
        )
        # Chỉ thêm tồn kho nếu include_inventory hoặc tồn kho = 0
        product_context += f"Tồn kho: {item.get('inventory', 0)}, "
        if include_specs:
            product_context += f"Mô tả: {item.get('specifications', 'N/A')}, "
        product_context += (
            f"Giá: {item.get('lifecare_price', 0):,.0f}đ, "
            f"Link sản phẩm: {item.get('link_product', 'N/A')}\n"
            f"Link ảnh: {item.get('avatar_images', 'N/A')}\n"
        )
    return product_context

def _build_prompt(user_query: str, context: str, needs_product_search: bool, wants_images: bool = False, product_infos: list = None) -> str:
    """Xây dựng prompt cho LLM."""
    base_instructions = """Bạn nên trả lời lễ phép, tôn trọng người dùng, hãy luôn xưng hô là em và gọi khách hàng là anh/chị.
    Nếu gặp các câu hỏi không liên quan đến việc sản phẩm hay tư vấn sản phẩm, bạn nên trả lời là em không rõ, anh/chị có thể hỏi lại câu khác giúp em được không ạ.
    Nếu bạn đã được cung cấp lịch sử hội thoại thì bạn không cần chào lại anh/chị nữa, chỉ cần chào lúc chưa có lịch sử hội thoại."""

    image_instruction = ""
    if wants_images:
        image_instruction = (
            'Lưu ý: Vì hình ảnh sản phẩm sẽ được hiển thị riêng, bạn không cần phải viết link ảnh trong câu trả lời. Thay vào đó, hãy xác nhận rằng bạn đang hiển thị hình ảnh, ví dụ hãy chỉ nói 1 lần duy nhất ở đầu câu trả lời: "Dạ đây là hình ảnh của sản phẩm ạ".'
            'Sau khi trả lời khách hàng, hãy trả về thêm một mục [PRODUCT_IMAGE] gồm tên sản phẩm khách muốn xem ảnh (mỗi tên một dòng, hoặc NONE nếu không có).\n'
            'Cấu trúc trả về:\n[ANSWER]\n<phần trả lời khách hàng>\n[PRODUCT_IMAGE]\n<tên sản phẩm muốn show ảnh, mỗi tên một dòng, hoặc NONE>\n'
            'Chỉ chọn tên sản phẩm trong danh sách sau:\n' + '\n'.join(f'- {info}' for info in (product_infos or []))
        )

    if needs_product_search:
        return f"""Bạn là một nhân viên tư vấn sản phẩm bán hàng cho cửa hàng thiết bị/phụ kiện điện tử tên là Hoàng Mai Mobile có địa chỉ tại Số 8 ngõ 117 Thái Hà, Trung Liệt, Đống Đa, Hà Nội (Làm việc từ 8h00 - 18h00), số Hotline: 0982153333.
    Chỉ nói địa chỉ, thời gian làm việc và số hotline của cửa hàng khi mà khách hàng hỏi về địa chỉ hoặc thời gian làm việc của cửa hàng.
    Hãy sử dụng thông tin được cung cấp dưới đây để trả lời câu hỏi của khách hàng một cách thân thiện, tự nhiên và chính xác. 
    Tuyệt đối không bịa thêm thông tin ngoài dữ liệu được cung cấp.
    
    Thông tin sản phẩm: {context}

    Câu hỏi của khách hàng: \"{user_query}\"

    Quy tắc trả lời:
     - Không tiết lộ số lượng tồn kho chính xác của sản phẩm.
     - Khi khách hàng hỏi về tồn kho:
        + Nếu tồn kho = 0: "Sản phẩm này bên em hiện đang hết hàng ạ."
        + Nếu tồn kho ≥ 1: "Sản phẩm này bên em còn hàng ạ."
     - Khi khách hàng hỏi về giá:
        + Nếu giá sản phẩm = 0đ: "Sản phẩm này hiện tại em chưa cập nhật được giá chính xác, nếu anh/chị chốt mua thì báo em để em kiểm tra lại và gửi giá tốt nhất cho mình ạ."
     - Đối với các sản phẩm có nhiều màu hoặc nhiều thuộc tính, tuyệt đối không chủ động liệt kê tất cả thuộc tính ngay từ đầu. Chỉ khi khách hàng hỏi cụ thể về một sản phẩm, bạn mới trình bày rõ các thuộc tính liên quan.
     - Không tự động cung cấp link ảnh sản phẩm. Chỉ đưa ra khi khách hàng yêu cầu rõ ràng.
     - Bạn cũng nên dựa vào phần lịch sử hội thoại để xác định đúng ý định của khách hàng, nếu thấy chưa hiểu rõ ý khách hàng hãy lịch sự bảo khách hàng có thể hỏi rõ lại.

    {image_instruction}
    {base_instructions}

    Lưu ý quan trọng:
    - KHÔNG được sử dụng bất kỳ định dạng Markdown hoặc HTML nào.
    - KHÔNG dùng các ký tự như `*`, `**`, `_`, `#`...
    - KHÔNG in đậm, in nghiêng, hay làm nổi bật văn bản.
    - Các tên sản phẩm nên dùng dấu gạch ngang "-" ở đầu mỗi tên sản phẩm.
    - Chỉ sử dụng plain text, được phép xuống dòng.

    Câu trả lời của bạn:"""
    else:
        return f"""Bạn là một nhân viên tư vấn sản phẩm bán hàng cho cửa hàng thiết bị/phụ kiện điện tử tên là Hoàng Mai Mobile có địa chỉ tại Số 8 ngõ 117 Thái Hà, Trung Liệt, Đống Đa, Hà Nội (Làm việc từ 8h00 - 18h00), số Hotline: 0982153333.
    Chỉ nói địa chỉ, thời gian làm việc và số hotline của cửa hàng khi mà khách hàng hỏi về địa chỉ hoặc thời gian làm việc của cửa hàng.
    Hãy trả lời câu hỏi của khách hàng một cách thân thiện, lễ phép, tự nhiên và chính xác.
    
    {context}

    Câu hỏi của khách hàng: \"{user_query}\"

    {base_instructions}

    Nếu khách hàng hỏi về sản phẩm cụ thể mà bạn không có thông tin, hãy hỏi họ cung cấp thêm chi tiết về sản phẩm họ đang tìm kiếm.
    
    Lưu ý quan trọng:
    - KHÔNG được sử dụng bất kỳ định dạng Markdown hoặc HTML nào.
    - KHÔNG dùng các ký tự như `*`, `**`, `_`, `#`...
    - KHÔNG in đậm, in nghiêng, hay làm nổi bật văn bản.
    - Chỉ sử dụng plain text, được phép xuống dòng.
    
    Câu trả lời của bạn:"""

def _parse_answer_and_images(llm_response: str, product_infos: list) -> tuple[str, list]:
    """
    Parse kết quả trả về từ LLM dạng:
    [ANSWER]\n<text>\n[PRODUCT_IMAGE]\n<name1>\n<name2>\n...
    """
    if not llm_response:
        return "", []
    answer = ""
    product_images = []
    parts = llm_response.split("[PRODUCT_IMAGE]")
    if len(parts) == 2:
        answer = parts[0].replace("[ANSWER]", "").strip()
        lines = [l.strip() for l in parts[1].split("\n") if l.strip()]
        # Chỉ giữ tên sản phẩm hợp lệ
        product_images = [l for l in lines if l in product_infos and l.upper() != 'NONE']
        if not product_images and lines and lines[0].upper() != 'NONE':
            # fallback: nếu LLM trả về tên không khớp, lấy dòng đầu tiên
            product_images = [lines[0]]
    else:
        answer = llm_response.strip()
    return answer, product_images


def _get_fallback_response(search_results: List[Dict], needs_product_search: bool) -> str:
    """Tạo câu trả lời dự phòng khi LLM không hoạt động."""
    if needs_product_search:
        if not search_results:
            return "Dạ, em xin lỗi, cửa hàng em chưa kinh doanh sản phẩm này ạ."
        first = search_results[0]
        return (
            f"Dạ, sản phẩm {first.get('product_name', 'N/A')} "
            f"giá {first.get('lifecare_price', 0):,.0f}đ, tồn kho {first.get('inventory', 0)}. "
            f"Anh/chị cần tư vấn thêm không ạ?"
        )
    else:
        return "Dạ, em xin lỗi, em không hiểu rõ câu hỏi của anh/chị. Anh/chị có thể hỏi lại không ạ?"