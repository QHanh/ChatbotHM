import re
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

    context = ""
    if history:
        context += f"Lịch sử hội thoại gần đây:\n{format_history_text(history)}\n"

    if needs_product_search:
        if not search_results:
            return {"answer": "Dạ, em xin lỗi, cửa hàng em chưa kinh doanh sản phẩm này ạ.", "product_images": []} if wants_images else "Dạ, em xin lỗi, cửa hàng em chưa kinh doanh sản phẩm này ạ."

        context += _build_product_context(search_results, include_specs)

    product_infos = [
        f"{p.get('product_name', '')} ({p.get('properties', '')})"
        for p in search_results if p.get('product_name')
    ] if wants_images else []

    prompt = _build_prompt(user_query, context, needs_product_search, wants_images, product_infos)

    print("--- PROMPT GỬI ĐẾN LLM (CẬP NHẬT QUY TẮC CHỌN ẢNH) ---")
    print(prompt)
    print("--------------------------------------------------")

    llm_response = None
    try:
        if model_choice == "gemini":
            model = get_gemini_model()
            if model:
                response = model.generate_content(prompt, safety_settings={'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE'})
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
                temperature=0.5,
                max_tokens=4000
            )
            llm_response = response.choices[0].message.content.strip()
            usage = response.usage
            print(f"📊 Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens}, Total: {usage.total_tokens}")
            cost = (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.6) / 1_000_000
            print(f"💰 Estimated cost (GPT-4o-mini): ${cost:.6f}")

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
    product_context = "Dữ liệu sản phẩm tìm thấy:\n"
    for item in search_results:
        product_context += f"- Tên: {item.get('product_name', 'N/A')}\n"
        product_context += f"  Thuộc tính: {item.get('properties', 'N/A')}\n"
        product_context += f"  Giá: {item.get('lifecare_price', 0):,.0f}đ\n"
        product_context += f"  Tồn kho: {item.get('inventory', 0)}\n"
        if include_specs:
            product_context += f"  Mô tả: {item.get('specifications', 'N/A')}\n"
    return product_context


def _build_prompt(user_query: str, context: str, needs_product_search: bool, wants_images: bool = False, product_infos: list = None) -> str:
    image_instruction = ""
    if wants_images:
        product_list_str = '\n'.join(f'- {info}' for info in product_infos or [])
        # GỢI Ý: Đã thêm một quy tắc mới, nghiêm ngặt hơn về cách chọn ảnh.
        image_instruction = f"""## HƯỚNG DẪN ĐẶC BIỆT KHI CUNG CẤP HÌNH ẢNH ##
- Khi khách muốn xem ảnh, câu trả lời PHẢI có 2 phần: [ANSWER] và [PRODUCT_IMAGE].
- Phần [ANSWER]: Bắt đầu bằng "Dạ đây là hình ảnh sản phẩm ạ." và nội dung tư vấn.
- Phần [PRODUCT_IMAGE]: Liệt kê tên sản phẩm từ danh sách dưới đây. Mỗi tên một dòng.

- **QUY TẮC CHỌN ẢNH (RẤT QUAN TRỌNG):** Phải đối chiếu chính xác từng chi tiết trong câu hỏi của khách (bao gồm cả model, thuộc tính) với "Danh sách sản phẩm". Chỉ chọn những dòng khớp **chính xác 100%**. Nếu khách hỏi về "MODEL:8512P", bạn chỉ được phép chọn dòng có chứa "MODEL:8512P". Không được suy diễn hay chọn các model tương tự.

- Danh sách sản phẩm có thể dùng cho [PRODUCT_IMAGE]:
{product_list_str}
"""

    if not needs_product_search:
        return f"""## BỐI CẢNH ##
- Bạn là "Mai", một nhân viên tư vấn thân thiện và chuyên nghiệp của cửa hàng "Hoàng Mai Mobile".
- Địa chỉ: Số 8 ngõ 117 Thái Hà, Trung Liệt, Đống Đa, Hà Nội.
- Giờ làm việc: 8h00 - 18h00.
- Hotline: 0982153333.
- Lịch sử trò chuyện:
{context}
## NHIỆM VỤ ##
- Trả lời câu hỏi của khách hàng: "{user_query}"
- Luôn xưng "em" và gọi khách là "anh/chị".
- CHỈ cung cấp thông tin cửa hàng (địa chỉ, giờ làm việc, hotline) khi khách hỏi trực tiếp.
- Nếu không biết, hãy nói: "Dạ, vấn đề này em không rõ. Anh/chị vui lòng hỏi giúp em câu khác liên quan đến sản phẩm được không ạ?"
## QUY TẮC ĐỊNH DẠNG ##
- KHÔNG dùng Markdown (*, #, _). Chỉ dùng text thuần.
## CÂU TRẢ LỜI CỦA BẠN: ##
"""

    return f"""## BỐI CẢNH ##
- Bạn là "Mai", một nhân viên tư vấn chuyên nghiệp của cửa hàng "Hoàng Mai Mobile".
- Dưới đây là lịch sử trò chuyện và dữ liệu về các sản phẩm liên quan đến câu hỏi của khách.

## NHIỆM VỤ ##
- Phân tích **toàn bộ lịch sử trò chuyện** và **câu hỏi mới nhất** của khách hàng để hiểu đúng ý định.
- Trả lời câu hỏi của khách hàng: "{user_query}"
- TUYỆT ĐỐI chỉ sử dụng thông tin trong phần "DỮ LIỆU CUNG CẤP" dưới đây.

## DỮ LIỆU CUNG CẤP ##
{context}

{image_instruction}

## QUY TẮC BẮT BUỘC ##
- **Xử lý câu hỏi chung về danh mục:**
    - Nếu câu hỏi của khách hàng chỉ là để xác nhận sự tồn tại của một loại sản phẩm chung (ví dụ: "shop có bán máy hàn không?"), **KHÔNG liệt kê tất cả sản phẩm ra ngay**.
    - Thay vào đó, hãy xác nhận là có bán và hỏi lại để làm rõ nhu cầu của khách.
    - **VÍ DỤ:**
        - Khách hỏi: "bên shop có bán máy hàn không ạ"
        - Trả lời đúng: "Dạ bên em có bán nhiều loại máy hàn ạ. Không biết anh/chị đang cần tìm loại máy hàn nào ạ?"

- **Phân tích ngữ cảnh:**
    - **BẮT BUỘC** phải xem lại "Lịch sử hội thoại gần đây" để hiểu đúng ý của khách.
    - **VÍ DỤ:** Nếu khách vừa hỏi về "máy hàn", sau đó hỏi "còn loại nào không", bạn phải hiểu là khách muốn xem các loại **máy hàn khác**.

- **Xem thêm / Loại khác:**
    - Khi khách hỏi xem thêm, hãy giới thiệu các sản phẩm có trong "DỮ LIỆU CUNG CẤP".

- **Tồn kho:**
    - Áp dụng khi khách hỏi về **tình trạng có sẵn** của một sản phẩm cụ thể (ví dụ: "máy hàn A còn hàng không?").
    - Trả lời "Dạ sản phẩm này bên em còn hàng ạ" (nếu tồn kho > 0) hoặc "Dạ sản phẩm này bên em hiện đang hết hàng ạ" (nếu tồn kho = 0).
    - KHÔNG tự động nói về tồn kho và KHÔNG nói số lượng cụ thể.

- **Giá sản phẩm:**
    - Nếu một sản phẩm có giá lớn hơn 0, bạn có thể chủ động nói giá khi giới thiệu.
    - **Nếu một sản phẩm có giá là 0đ, TUYỆT ĐỐI KHÔNG tự động nói ra giá.**
    - **CHỈ KHI** khách hàng hỏi cụ thể về giá của sản phẩm có giá 0đ, hãy trả lời: "Dạ sản phẩm này em chưa có giá chính xác, nếu anh/chị muốn mua thì em sẽ xem lại và báo cho anh/chị giá chính xác sản phẩm này ạ."

- **Xưng hô và Định dạng:**
    - Luôn xưng "em", gọi khách là "anh/chị".
    - KHÔNG dùng Markdown (*, #, _), không in đậm, in nghiêng. Chỉ dùng text thuần.
    - Khi liệt kê sản phẩm, dùng dấu gạch ngang "-" ở đầu dòng.

## CÂU TRẢ LỜI CỦA BẠN: ##
"""

def _parse_answer_and_images(llm_response: str, product_infos: list) -> tuple[str, list]:
    if not llm_response:
        return "Dạ em xin lỗi, có lỗi xảy ra trong quá trình tạo câu trả lời.", []

    answer = ""
    product_images = []
    parts = re.split(r'\[PRODUCT_IMAGE\]', llm_response, flags=re.IGNORECASE)

    def clean_name(name: str) -> str:
        return re.sub(r"^[-\s*•+]+", "", name.strip())

    if len(parts) == 2:
        answer = re.sub(r'\[ANSWER\]', '', parts[0], flags=re.IGNORECASE).strip()
        image_lines = [clean_name(l) for l in re.split(r'[\n,]+', parts[1]) if l.strip()]
        valid_product_names = set(product_infos)
        product_images = [line for line in image_lines if line in valid_product_names and line.upper() != 'NONE']

        if not product_images and image_lines:
             for line in image_lines:
                 for valid_name in valid_product_names:
                     if line in valid_name:
                         product_images.append(valid_name)
                         break
    else:
        answer = llm_response.strip()

    if not answer and product_images:
        answer = "Dạ đây là hình ảnh sản phẩm em gửi anh/chị tham khảo ạ."

    return answer, product_images


def _get_fallback_response(search_results: List[Dict], needs_product_search: bool) -> str:
    if needs_product_search:
        if not search_results:
            return "Dạ, em xin lỗi, cửa hàng em chưa kinh doanh sản phẩm này ạ."
        first = search_results[0]
        return (
            f"Dạ, sản phẩm {first.get('product_name', 'N/A')} "
            f"hiện đang có giá {first.get('lifecare_price', 0):,.0f}đ. "
            f"Anh/chị cần tư vấn thêm không ạ?"
        )
    else:
        return "Dạ, em xin lỗi, em không hiểu rõ câu hỏi của anh/chị. Anh/chị có thể hỏi lại không ạ?"