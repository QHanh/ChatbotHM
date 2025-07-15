from typing import List, Dict
from src.services.llm_service import get_gemini_model, get_lmstudio_response
from src.utils.helpers import is_general_query, format_history_text

def generate_llm_response(
    user_query: str, 
    search_results: list, 
    history: list = None, 
    include_specs: bool = False, 
    model_choice: str = "gemini", 
    needs_product_search: bool = True
) -> str:
    """
    Tạo prompt và gọi đến LLM để sinh câu trả lời.
    """
    if is_general_query(user_query):
        if not search_results:
            return "Dạ, cửa hàng em chưa có sản phẩm nào để giới thiệu ạ."
        product_names = [item.get('product_name', 'N/A') for item in search_results]
        return (
            "Hiện tại cửa hàng em đang kinh doanh nhiều loại sản phẩm về các thiết bị điện tử, ví dụ như: "
            + ".\n".join(product_names) + " và nhiều sản phẩm khác nữa"
            + ".\n\nAnh/chị muốn tìm hiểu thêm về sản phẩm nào không ạ?"
        )

    # Chuẩn bị context
    context = ""
    if history:
        context += f"Lịch sử hội thoại gần đây:\n{format_history_text(history)}\n"
    
    if needs_product_search:
        if not search_results:
            return "Dạ, em xin lỗi, cửa hàng em chưa kinh doanh sản phẩm này ạ."
        
        context += _build_product_context(search_results, include_specs)

    # Xây dựng prompt
    prompt = _build_prompt(user_query, context, needs_product_search)
    
    print("--- PROMPT GỬI ĐẾN LLM ---")
    print(prompt)
    print("--------------------------")

    # Gọi LLM
    if model_choice == "gemini":
        model = get_gemini_model()
        if model:
            try:
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                print(f"Lỗi khi gọi Gemini: {e}")
    elif model_choice == "lmstudio":
        try:
            response = get_lmstudio_response(prompt)
            return response
        except Exception as e:
            print(f"Lỗi khi gọi LM Studio: {e}")
    
    # Fallback response
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
            f"Tồn kho: {item.get('inventory', 0)}, "
        )
        if include_specs:
            product_context += f"Mô tả: {item.get('specifications', 'N/A')}, "
        
        product_context += (
            f"Giá: {item.get('lifecare_price', 0):,.0f}đ, "
            f"Link đặt hàng: {item.get('link_product', 'N/A')}\n"
            f"Link ảnh: {item.get('avatar_images', 'N/A')}\n"
        )
    return product_context

def _build_prompt(user_query: str, context: str, needs_product_search: bool) -> str:
    """Xây dựng prompt cho LLM."""
    base_instructions = """Bạn nên trả lời lễ phép, tôn trọng người dùng, hãy luôn xưng hô là em và gọi khách hàng là anh/chị.
    Nếu gặp các câu hỏi không liên quan đến việc sản phẩm hay tư vấn sản phẩm, bạn nên trả lời là em không rõ, vui lòng hỏi lại.
    Nếu bạn đã được cung cấp lịch sử hội thoại thì bạn không cần chào anh/chị nữa, chỉ chào lúc chưa có lịch sử hội thoại."""

    if needs_product_search:
        return f"""Bạn là một trợ lý tư vấn sản phẩm bán hàng cho cửa hàng thiết bị/phụ kiện của bạn. 
    Hãy sử dụng thông tin được cung cấp dưới đây để trả lời câu hỏi của khách hàng một cách thân thiện, tự nhiên và chính xác. 
    Không tự bịa thêm thông tin không có trong ngữ cảnh.
    
    {context}

    Câu hỏi của khách hàng: \"{user_query}\"

    Chỉ nói cửa hàng còn bao nhiêu sản phẩm khi mà người dùng hỏi về tồn kho, nếu không hỏi thì không cần nói.
    Nếu một sản phẩm được cung cấp có giá là 0đ, hãy nói với khách hàng rằng giá sản phẩm này là "Liên hệ".
    Không tự động cung cấp link đặt hàng hay link ảnh, chỉ đưa ra khi khách hàng yêu cầu một cách cụ thể.
    {base_instructions}
    Câu trả lời của bạn:"""
    else:
        return f"""Bạn là một trợ lý tư vấn bán hàng cho cửa hàng thiết bị/phụ kiện của bạn.
    Hãy trả lời câu hỏi của khách hàng một cách thân thiện, tự nhiên và chính xác.
    
    {context}

    Câu hỏi của khách hàng: \"{user_query}\"

    {base_instructions}
    Nếu khách hàng hỏi về sản phẩm cụ thể mà bạn không có thông tin, hãy đề nghị họ cung cấp thêm chi tiết về sản phẩm họ đang tìm kiếm.
    Câu trả lời của bạn:"""

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