import json
import re
from collections import defaultdict
from typing import List, Dict, Optional
from src.services.llm_service import get_gemini_model, get_lmstudio_response, get_openai_model
from src.utils.helpers import is_general_query, format_history_text

def generate_llm_response(
    user_query: str,
    search_results: list,
    history: list = None,
    include_specs: bool = False,
    model_choice: str = "gemini",
    needs_product_search: bool = True,
    wants_images: bool = False,
    is_image_search: bool = False
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
    has_history = bool(history)
    if has_history:
        context += f"Lịch sử hội thoại gần đây:\n{format_history_text(history)}\n"
    else:
        context += "Lịch sử hội thoại gần đây:\n(Đây là tin nhắn đầu tiên)\n"

    if needs_product_search:
        # if not search_results:
        #     return {"answer": "Dạ, em xin lỗi, cửa hàng em chưa kinh doanh sản phẩm này ạ.", "product_images": []} if wants_images else "Dạ, em xin lỗi, cửa hàng em chưa kinh doanh sản phẩm này ạ."

        context += _build_product_context(search_results, include_specs)

    product_infos = [
        f"{p.get('product_name', '')} ({p.get('properties', '')})"
        for p in search_results if p.get('product_name')
    ] if wants_images else []

    prompt = _build_prompt(user_query, context, needs_product_search, wants_images, product_infos, has_history, is_image_search)

    print("--- PROMPT GỬI ĐẾN LLM ---")
    print(prompt)
    print("--------------------------")

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
    """
    Xây dựng context thông tin sản phẩm, nhóm các sản phẩm cùng tên lại với nhau.
    """
    product_groups = defaultdict(list)
    
    for item in search_results:
        product_groups[item.get('product_name', 'N/A')].append(item)

    product_context = "Dữ liệu sản phẩm tìm thấy:\n"
    
    for name, items in product_groups.items():
        product_context += f"- Tên: {name}\n"

        sorted_items = sorted(items, key=lambda x: str(x.get('properties', '')))

        if len(sorted_items) == 1:
            item = sorted_items[0]
            prop = item.get('properties')
            if prop and str(prop).strip() != '0':
                product_context += f"  Thuộc tính: {prop}\n"
            
            price = item.get('lifecare_price', 0)
            price_str = f"{price:,.0f}đ" if price > 0 else "Liên hệ"
            product_context += f"  Giá: {price_str}\n"
            inventory = item.get('inventory', 0)
            if inventory > 0:
                product_context += f"  Tình trạng: Còn hàng ({inventory} sản phẩm)\n"
            else:
                product_context += "  Tình trạng: Hết hàng\n"
        else:
            product_context += "  Lưu ý: Sản phẩm này có nhiều thuộc tính khác nhau (ví dụ: loại, cỡ, model, màu,...). Các phiên bản có sẵn:\n"
            for item in sorted_items:
                prop = item.get('properties', 'N/A')
                price = item.get('lifecare_price', 0)
                inventory = item.get('inventory', 0)
                price_str = f"{price:,.0f}đ" if price > 0 else "Liên hệ"
                stock_str = f"Còn hàng ({inventory})" if inventory > 0 else "Hết hàng"
                product_context += f"    + {prop} - Giá: {price_str} - Tình trạng: {stock_str}\n"

        
        if include_specs:
            product_context += f"  Mô tả: {sorted_items[0].get('specifications', 'N/A')}\n"
    return product_context


def _build_prompt(user_query: str, context: str, needs_product_search: bool, wants_images: bool = False, product_infos: list = None, has_history: bool = None, is_image_search: bool = False) -> str:
    """
    Xây dựng prompt cho LLM với các quy tắc hội thoại nâng cao.
    """
    image_instruction = ""
    if wants_images:
        product_list_str = '\n'.join(f'- {info}' for info in product_infos or [])
        image_instruction = f"""## HƯỚNG DẪN ĐẶC BIỆT KHI CUNG CẤP HÌNH ẢNH ##
- Nhiệm vụ của bạn là tạo ra một danh sách các sản phẩm kèm ảnh dựa trên "DỮ LIỆU CUNG CẤP".
- **KHÔNG** được hỏi lại khách hàng. **KHÔNG** thêm bất kỳ lời thoại nào khác.
- Câu trả lời của bạn **BẮT BUỘC** phải có 2 phần: `[ANSWER]` và `[PRODUCT_IMAGE]`.

- **Phần [ANSWER]:**
    - **KHÔNG** thêm bất kỳ lời chào hay câu giới thiệu nào.
    - **Chỉ liệt kê** lại các sản phẩm mà khách muốn xem ảnh.
    - **Mỗi sản phẩm phải nằm trên một dòng riêng**, **không được** cách dòng quá 1 dòng, bắt đầu bằng dấu gạch ngang (-).
    - Ghi rõ Tên và Giá của sản phẩm.
    - **VÍ DỤ ĐỊNH DẠNG PHẦN ANSWER:**
        - Máy hàn OSSTEAM T210 - giá 145,000đ
        - Máy hàn MECHANIC A210 - giá 780,000đ

- **Phần [PRODUCT_IMAGE]:**
    - Liệt kê CHÍNH XÁC tên định danh (có dạng Tên (Thuộc tính)) của các sản phẩm đã liệt kê trong phần [ANSWER].
    - **Mỗi tên một dòng và phải theo đúng thứ tự** đã liệt kê ở phần [ANSWER].

- **QUY TẮC CHỌN ẢNH:** Phải đối chiếu chính xác từng chi tiết trong câu hỏi của khách (bao gồm cả model, thuộc tính) với "Danh sách sản phẩm". Chỉ chọn những dòng khớp **chính xác 100%**.

- Danh sách sản phẩm có thể dùng cho [PRODUCT_IMAGE]:
{product_list_str}
"""

    store_info = """- Tên cửa hàng: Hoàng Mai Mobile
- Địa chỉ: Số 8 ngõ 117 Thái Hà, Đống Đa, Hà Nội
- Giờ làm việc: 8h00 - 18h00
- Hotline: 0982153333"""

    greeting_rule = ""
    
    if not wants_images:
        if not has_history:
            greeting_rule = '- **Chào hỏi:** Bắt đầu câu trả lời bằng lời chào đầy đủ "Dạ, em chào anh/chị ạ." vì đây là tin nhắn đầu tiên.'
        else:
            greeting_rule = '- **Chào hỏi:** KHÔNG chào hỏi đầy đủ. Bắt đầu câu trả lời trực tiếp bằng "Dạ,".'

    image_search_priority_rule = ""
    if is_image_search:
        image_search_priority_rule = """
**QUY TẮC ƯU TIÊN TUYỆT ĐỐI (TÌM KIẾM BẰNG HÌNH ẢNH):**
- Cuộc trò chuyện này bắt đầu bằng việc khách hàng gửi một hình ảnh để tìm kiếm.
- Nhiệm vụ của bạn là trả lời câu hỏi của khách hàng DỰA HOÀN TOÀN vào "DỮ LIỆU SẢN PHẨM TÌM THẤY".
- **TUYỆT ĐỐI BỎ QUA** lịch sử trò chuyện cũ và không được liệt kê các sản phẩm khác không có trong dữ liệu tìm thấy.
"""

    if not needs_product_search:
        return f"""## BỐI CẢNH ##
- Bạn là một nhân viên tư vấn chuyên nghiệp của cửa hàng.
- Thông tin cửa hàng:
{store_info}
- Dưới đây là lịch sử trò chuyện.

## NHIỆM VỤ (RẤT QUAN TRỌNG) ##
- Trả lời câu hỏi của sau khách hàng: "{user_query}"
- **BẠN PHẢI TRẢ LỜI DỰA TRÊN NGỮ CẢNH CỦA LỊCH SỬ HỘI THOẠI.**
- **TUYỆT ĐỐI KHÔNG ĐƯỢC THAY ĐỔI CHỦ ĐỀ.** Ví dụ: nếu cuộc trò chuyện đang về "sản phẩm A", câu trả lời của bạn cũng phải về "sản phẩm A", không được tự ý chuyển sang "sản phẩm B".
- Hãy trả lời một cách thân thiện và lễ phép.
- **Nếu tin nhắn cuối cùng trong lịch sử là bot nói về việc chuyển cho nhân viên, và câu hỏi mới của khách là một lời chào chung chung (ví dụ: "Hi", "hello", "chào shop"), HÃY bỏ qua ngữ cảnh cũ và chào lại một cách bình thường như một cuộc trò chuyện mới.** Ví dụ: "Dạ, em chào anh/chị. Em có thể giúp gì cho mình ạ?"

## QUY TẮC ##

1.  **Sử dụng Emoji (Zalo):**
    - Để làm cho cuộc trò chuyện thân thiện hơn, hãy sử dụng các mã emoji sau một cách hợp lý.
    - **Danh sách mã emoji có thể dùng:** :d :( :~ :b :') 8-) :-(( :$ :3 :z :(( &-( :p :o :( ;-) --b :)) :-* ;-d /-showlove ;d ;o :--| 8* /-heart /-strong _()_ $-) /-break /-ok
    - **Quy tắc:** Chỉ sử dụng 1-2 emoji trong một câu trả lời để thể hiện cảm xúc tích cực hoặc thân thiện. Ví dụ: "Dạ, bên em có sản phẩm này ạ :d .", "Em cảm ơn anh/chị nhiều ạ /-heart .". Không lạm dụng. Lưu ý: **SAU EMOJI PHẢI CÓ MỘT KHOẢNG TRẮNG (SPACE)**.

2.  {greeting_rule}

3. Nếu khách hàng hỏi những từ hoặc câu bạn không hiểu hãy nói: "Dạ em không hiểu ý của anh/chị ạ."

## DỮ LIỆU CUNG CẤP ##
{context}

## CÂU TRẢ LỜI CỦA BẠN: ##
"""

    return f"""## BỐI CẢNH ##
- Bạn là một nhân viên tư vấn chuyên nghiệp, thông minh và khéo léo.
- Dưới đây là lịch sử trò chuyện và dữ liệu về các sản phẩm liên quan.

## NHIỆM VỤ ##
- Phân tích ngữ cảnh và câu hỏi của khách hàng để trả lời một cách chính xác và tự nhiên như người thật.
- **Ưu tiên hàng đầu: Luôn trả lời trực tiếp vào câu hỏi của khách hàng trước, sau đó mới áp dụng các quy tắc khác.**
- Câu hỏi của khách hàng: "{user_query}"
- TUYỆT ĐỐI chỉ sử dụng thông tin trong phần "DỮ LIỆU CUNG CẤP".

## DỮ LIỆU CUNG CẤP ##
{context}

{image_instruction}

## QUY TẮC HỘI THOẠI BẮT BUỘC ##

1.  {image_search_priority_rule}

2.  {greeting_rule}

3.  **Lọc và giữ vững chủ đề (QUAN TRỌNG NHẤT):**
    - Phải xác định **chủ đề chính** của cuộc trò chuyện (ví dụ: "máy hàn", "kính hiển vi RELIFE").
    - **TUYỆT ĐỐI KHÔNG** giới thiệu sản phẩm không thuộc chủ đề chính, ví dụ khách hỏi về máy hàn Quick thì **không giới thiệu** máy khò Quick, tay hàn Quick, mũi hàn Quick.
    - Nếu khách hỏi một sản phẩm không có trong dữ liệu cung cấp, hãy trả lời rằng: "Dạ, bên em không bán 'tên_sản_phẩm_khách_hỏi' ạ."

4.  **Sản phẩm có nhiều model, combo, cỡ, màu sắc,... (tùy thuộc tính):**
    - Khi giới thiệu lần đầu, chỉ nói tên sản phẩm chính và hãy thông báo có nhiều màu hoặc có nhiều model hoặc có nhiều cỡ,... (tùy vào thuộc tính của sản phẩm).
    - **Khi khách hỏi trực tiếp về số lượng** (ví dụ: "chỉ có 3 màu thôi à?"), bạn phải trả lời thẳng vào câu hỏi.

5.  **Xử lý câu hỏi chung về danh mục:**
    - Nếu khách hỏi "shop có bán máy hàn không?, có kính hiển vi không?", **KHÔNG liệt kê sản phẩm ra ngay**. Hãy xác nhận là có bán và có thể nói ra một số đặc điểm riêng biệt như thương hiệu, hãng có trong dữ liệu cung cấp và hỏi lại để làm rõ nhu cầu lựa chọn.

6.  **Liệt kê sản phẩm:**
    - Khi khách hàng yêu cầu liệt kê các sản phẩm (ví dụ: "có những loại nào", "kể hết ra đi"), bạn **PHẢI** trình bày câu trả lời dưới dạng một danh sách rõ ràng.
    - **Mỗi sản phẩm phải nằm trên một dòng riêng**, bắt đầu bằng dấu gạch ngang (-).
    - **KHÔNG** được gộp tất cả các tên sản phẩm vào trong một đoạn văn.

7.  **Xem thêm / Loại khác:**
    - Áp dụng khi khách hỏi "còn không?", "còn loại nào nữa không?" hoặc có thể là "tiếp đi" (tùy vào ngữ cảnh cuộc trò chuyện). Hiểu rằng khách muốn xem thêm sản phẩm khác (cùng chủ đề), **không phải hỏi tồn kho**.

8.  **Tồn kho:**
    - **KHÔNG** liệt kê các sản phẩm hoặc các phiên bản sản phẩm có "Tình trạng: Hết hàng".
    - **KHÔNG** tự động nói ra số lượng tồn kho chính xác.
    
9.  **Giá sản phẩm:**
    - **Các sản phẩm có giá là **Liên hệ** thì **KHÔNG ĐƯỢC** nói ra giá, chỉ nói tên sản phẩm KHÔNG KÈM GIÁ.
    - **Các sản phẩm có giá **KHÁC** **Liên hệ** thì hãy luôn nói kèm giá khi liệt kê.
    - **CHỈ KHI** khách hàng hỏi giá của sản phẩm có giá "Liên hệ" thì hãy nói "Sản phẩm này em chưa có giá chính xác, nếu anh/chị muốn mua thì em sẽ xem lại và báo lại cho anh chị một mức giá hợp lý".

10.  **Xưng hô và Định dạng:**
    - Luôn xưng "em", gọi khách là "anh/chị".
    - KHÔNG dùng Markdown. Chỉ dùng text thuần.

11.  **Link sản phẩm**
    - Bạn cứ gửi kèm link sản phẩm khi liệt kê các sản phẩm ra và chỉ cần gắn link vào cuối tên sản phẩm **không cần thêm gì hết**.

12.  **Với các câu hỏi bao quát khi khách hàng mới hỏi**
    - Ví dụ: "Shop bạn bán những mặt hàng gì", "Bên bạn có những sản phẩm gi?", hãy trả lời rằng: "Dạ, bên em chuyên kinh doanh các dụng cụ sửa chữa, thiết bị điện tử như máy hàn, kính hiển vi,... Anh/chị đang quan tâm mặt hàng nào để em tư vấn ạ."

13.  **Xử lý lời đồng ý:**
    - Nếu bot ở lượt trước vừa hỏi một câu hỏi Yes/No để đề nghị cung cấp thông tin (ví dụ: "Anh/chị có muốn xem chi tiết không?") và câu hỏi mới nhất của khách là một lời đồng ý (ví dụ: "có", "vâng", "ok"), HÃY thực hiện hành động đã đề nghị.
    - Trong trường hợp này, hãy liệt kê các sản phẩm có trong "DỮ LIỆU CUNG CẤP" theo đúng định dạng danh sách.

14.  **Sử dụng Emoji (Zalo):**
    - Để làm cho cuộc trò chuyện thân thiện hơn, hãy sử dụng các mã emoji sau một cách hợp lý.
    - **Danh sách mã emoji có thể dùng:** :d :( :~ :b :') 8-) :-(( :$ :3 :z :(( &-( :p :o :( ;-) --b :)) :-* ;-d /-showlove ;d ;o :--| 8* /-heart /-strong _()_ $-) /-break /-ok
    - **Quy tắc:** Chỉ sử dụng 1-2 emoji trong một câu trả lời để thể hiện cảm xúc tích cực hoặc thân thiện. Ví dụ: "Dạ, bên em có sản phẩm này ạ :d ", "Em cảm ơn anh/chị nhiều ạ /-heart ". Không lạm dụng. Lưu ý: **SAU EMOJI PHẢI CÓ MỘT KHOẢNG TRẮNG (SPACE)**.

## CÂU TRẢ LỜI CỦA BẠN: ##
"""

def _parse_answer_and_images(llm_response: str, product_infos: list) -> tuple[str, list]:
    """
    Parse kết quả trả về từ LLM.
    """
    if not llm_response:
        return "Dạ em xin lỗi, có lỗi xảy ra trong quá trình tạo câu trả lời.", []

    answer = ""
    product_images = []
    parts = re.split(r'\[PRODUCT_IMAGE\]', llm_response, flags=re.IGNORECASE)

    def clean_name(name: str) -> str:
        return re.sub(r"^[-\s*•+]+", "", name.strip())

    if len(parts) == 2:
        answer = re.sub(r'\[ANSWER\]', '', parts[0], flags=re.IGNORECASE).strip()
        image_lines = [clean_name(l) for l in re.split(r'[\n]+', parts[1]) if l.strip()]
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
    """Tạo câu trả lời dự phòng khi LLM không hoạt động."""
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
    
def evaluate_and_choose_product(user_query: str, history_text: str, product_candidates: List[Dict], model_choice: str = "gemini") -> Dict:
    """
    Sử dụng một lệnh gọi AI duy nhất để vừa đánh giá độ cụ thể của yêu cầu,
    vừa chọn ra sản phẩm phù hợp nhất nếu có thể.
    Trả về một dictionary: {'type': 'GENERAL'/'SPECIFIC'/'NONE', 'product': product_dict/None}
    """
    if not product_candidates:
        return {'type': 'NONE', 'product': None}

    prompt_list = ""
    for i, product in enumerate(product_candidates):
        name = product.get("product_name", "")
        props = product.get("properties", "")
        full_name = f"{name} ({props})" if props and str(props) != '0' else name
        prompt_list += f"{i}: {full_name}\n"

    prompt = f"""
    Bạn là một AI chuyên phân tích và chọn lựa sản phẩm. Dựa vào lịch sử hội thoại, yêu cầu mua của khách hàng và danh sách sản phẩm, hãy thực hiện 2 nhiệm vụ sau:
    1.  Đánh giá xem yêu cầu của khách là "GENERAL" (hỏi chung chung về một loại) hay "SPECIFIC" (chỉ đến một sản phẩm cụ thể).
    2.  Nếu yêu cầu là "SPECIFIC", hãy chọn ra sản phẩm phù hợp nhất.

    Lưu ý:
    - Bạn sẽ trả về "GENERAL" nếu thấy ngữ cảnh lịch sử chat và yêu cầu của khách chưa đủ để phân biệt được sản phẩm cụ thể trong danh sách sản phẩm để chọn bên dưới. 
      Ví dụ: khách hỏi "bán cho mình chiếc kính hiển vi 2 mắt" và trong lịch sử chat cũng không thấy khách đang đề cập rõ đến loại hay brand nào, nhưng trong danh sách sản phẩm để chọn lại có 2 loại brand kính hiển vi 2 mắt khác nhau, cho nên chưa xác định được sản phẩm cụ thể nào được chọn.

    Hãy trả về kết quả dưới dạng JSON với cấu trúc: {{"type": "GENERAL" | "SPECIFIC", "index": SỐ_THỨ_TỰ | null}}
    - Nếu yêu cầu là "GENERAL", "index" sẽ là null.
    - Nếu không có sản phẩm nào phù hợp, hãy trả về {{"type": "NONE", "index": null}}

    Lịch sử hội thoại:
    {history_text}
    Yêu cầu mới nhất của khách hàng: "{user_query}"
    Danh sách sản phẩm để chọn:
    {prompt_list}

    JSON kết quả:
    """

    try:
        model = get_gemini_model()
        if model:
            response = model.generate_content(prompt)
            json_text = re.search(r'\{.*\}', response.text, re.DOTALL).group(0)
            data = json.loads(json_text)
            
            request_type = data.get("type", "NONE")
            index = data.get("index")

            if request_type == "SPECIFIC" and index is not None and 0 <= index < len(product_candidates):
                print(f"AI đánh giá: SPECIFIC, chọn index: {index}")
                return {'type': 'SPECIFIC', 'product': product_candidates[index]}
            
            print(f"AI đánh giá: {request_type}")
            return {'type': request_type, 'product': None}

    except Exception as e:
        print(f"Lỗi khi AI đánh giá và chọn sản phẩm: {e}")

    # Fallback an toàn
    return {'type': 'NONE', 'product': None}