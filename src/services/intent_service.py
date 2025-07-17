import json
import re
from typing import List, Dict, Optional
from src.services.llm_service import get_gemini_model, get_lmstudio_response, get_openai_model

def is_product_search_query(user_query: str, history: list = None, model_choice: str = "gemini") -> bool:
    """
    Sử dụng LLM để xác định xem câu hỏi của khách hàng có cần tìm kiếm thông tin sản phẩm hay không.
    """
    if model_choice == "gemini":
        model = get_gemini_model()
        if not model:
            return True
    elif model_choice != "lmstudio" and model_choice != "openai":
        return True

    history_text = ""
    if history:
        for turn in history[-3:]:
            history_text += f"Khách: {turn['user']}\nBot: {turn['bot']}\n"

    prompt = f"""Bạn là một AI phân loại ý định. Hãy đọc câu hỏi của khách hàng trong bối cảnh cuộc trò chuyện và quyết định xem họ có đang hỏi về thông tin sản phẩm cụ thể hay không. 

Câu hỏi cần tìm kiếm sản phẩm là những câu hỏi như:
- Hỏi về một sản phẩm cụ thể (ví dụ: "shop có bán điện thoại iPhone không?", "có bán laptop Dell không?") 
- Hỏi về giá cả, tính năng, thông số kỹ thuật của sản phẩm
- Hỏi về tồn kho, hàng có sẵn
- So sánh các sản phẩm

Câu hỏi KHÔNG cần tìm kiếm sản phẩm là những câu như:
- Chào hỏi đơn thuần ("chào", "xin chào")
- Câu hỏi chung chung không liên quan đến sản phẩm ("bạn là ai?", "thời tiết hôm nay thế nào?")
- Câu hỏi về chính sách, thông tin cửa hàng mà không hỏi về sản phẩm cụ thể
- Cảm ơn, tạm biệt

Chỉ trả lời 'CÓ' nếu câu hỏi cần tìm kiếm sản phẩm hoặc 'KHÔNG' nếu không cần.

Bối cảnh hội thoại gần đây:
{history_text}

Câu hỏi của khách hàng: "{user_query}"

Câu hỏi này có cần tìm kiếm thông tin sản phẩm không? (CÓ/KHÔNG):"""

    if model_choice == "gemini":
        try:
            model = get_gemini_model()
            response = model.generate_content(prompt)
            answer = response.text.strip().upper()
            print(f"--- KIỂM TRA CẦN TÌM KIẾM SẢN PHẨM (GEMINI) ---")
            print(f"Câu trả lời của LLM: {answer}")
            print("------------------------------------------")
            return "CÓ" in answer
        except Exception as e:
            print(f"Lỗi khi kiểm tra ý định tìm kiếm bằng Gemini: {e}")
            return True
    elif model_choice == "lmstudio":
        try:
            response = get_lmstudio_response(prompt)
            answer = response.strip().upper()
            print(f"--- KIỂM TRA CẦN TÌM KIẾM SẢN PHẨM (LM STUDIO) ---")
            print(f"Câu trả lời của LLM: {answer}")
            print("------------------------------------------")
            return "CÓ" in answer
        except Exception as e:
            print(f"Lỗi khi kiểm tra ý định tìm kiếm bằng LM Studio: {e}")
            return True
    elif model_choice == "openai":
        try:
            openai = get_openai_model()
            if not openai:
                return True
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4000
            )
            answer = response.choices[0].message.content.strip().upper()
            print(f"--- KIỂM TRA CẦN TÌM KIẾM SẢN PHẨM (OPENAI) ---")
            print(f"Câu trả lời của LLM: {answer}")
            print("------------------------------------------")
            return "CÓ" in answer
        except Exception as e:
            print(f"Lỗi khi kiểm tra ý định tìm kiếm bằng OpenAI: {e}")
            return True
    return True

def llm_wants_specifications(user_query: str, history: list = None, model_choice: str = "gemini") -> bool:
    """
    Sử dụng LLM để xác định xem người dùng có muốn biết thông số kỹ thuật/chi tiết sản phẩm hay không.
    """
    if model_choice == "gemini":
        model = get_gemini_model()
        if not model:
            return False
    elif model_choice != "lmstudio" and model_choice != "openai":
        return False

    history_text = ""
    if history:
        for turn in history[-3:]:
            history_text += f"Khách: {turn['user']}\nBot: {turn['bot']}\n"

    prompt = f"""Bạn là một AI phân loại ý định. Hãy đọc câu hỏi của khách hàng trong bối cảnh cuộc trò chuyện và quyết định xem họ có đang hỏi về thông số kỹ thuật, chi tiết, đặc điểm, hay tính năng của một sản phẩm hay không (chú ý: họ hỏi ảnh thì không phải là hỏi thông số kỹ thuật). Chỉ trả lời 'CÓ' hoặc 'KHÔNG'.

Bối cảnh hội thoại gần đây:
{history_text}

Câu hỏi của khách hàng: "{user_query}"

Khách hàng có hỏi về thông số/chi tiết sản phẩm không? (CÓ/KHÔNG):"""

    if model_choice == "gemini":
        try:
            model = get_gemini_model()
            response = model.generate_content(prompt)
            answer = response.text.strip().upper()
            print(f"--- KIỂM TRA Ý ĐỊNH XEM THÔNG SỐ (GEMINI) ---")
            print(f"Câu trả lời của LLM: {answer}")
            print("------------------------------------------")
            return "CÓ" in answer
        except Exception as e:
            print(f"Lỗi khi kiểm tra ý định bằng Gemini: {e}")
            return False
    elif model_choice == "lmstudio":
        try:
            response = get_lmstudio_response(prompt)
            answer = response.strip().upper()
            print(f"--- KIỂM TRA Ý ĐỊNH XEM THÔNG SỐ (LM STUDIO) ---")
            print(f"Câu trả lời của LLM: {answer}")
            print("------------------------------------------")
            return "CÓ" in answer
        except Exception as e:
            print(f"Lỗi khi kiểm tra ý định bằng LM Studio: {e}")
            return False
    elif model_choice == "openai":
        try:
            openai = get_openai_model()
            if not openai:
                return False
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4000
            )
            answer = response.choices[0].message.content.strip().upper()
            print(f"--- KIỂM TRA Ý ĐỊNH XEM THÔNG SỐ (OPENAI) ---")
            print(f"Câu trả lời của LLM: {answer}")
            print("------------------------------------------")
            return "CÓ" in answer
        except Exception as e:
            print(f"Lỗi khi kiểm tra ý định bằng OpenAI: {e}")
            return False
    return False

def is_asking_for_images(user_query: str, history: list = None, model_choice: str = "gemini") -> bool:
    """
    Sử dụng LLM để xác định xem người dùng có hỏi về ảnh sản phẩm hay không.
    """
    if model_choice == "gemini":
        model = get_gemini_model()
        if not model:
            return False
    elif model_choice != "lmstudio" and model_choice != "openai":
        return False

    history_text = ""
    if history:
        for turn in history[-3:]:
            history_text += f"Khách: {turn['user']}\nBot: {turn['bot']}\n"

    prompt = f"""Bạn là một AI phân loại ý định. Hãy đọc câu hỏi của khách hàng trong bối cảnh cuộc trò chuyện và quyết định xem họ có đang hỏi về ảnh, hình ảnh của sản phẩm hay không.

Câu hỏi về ảnh sản phẩm bao gồm:
- "cho tôi xem ảnh", "có ảnh không", "hình ảnh sản phẩm"
- "show ảnh", "xem hình", "có hình không"
- "ảnh như thế nào", "trông như thế nào"

Chỉ trả lời 'CÓ' nếu câu hỏi hỏi về ảnh/hình ảnh hoặc 'KHÔNG' nếu không hỏi về ảnh.

Bối cảnh hội thoại gần đây:
{history_text}

Câu hỏi của khách hàng: "{user_query}"

Khách hàng có hỏi về ảnh sản phẩm không? (CÓ/KHÔNG):"""

    if model_choice == "gemini":
        try:
            model = get_gemini_model()
            response = model.generate_content(prompt)
            answer = response.text.strip().upper()
            print(f"--- KIỂM TRA Ý ĐỊNH XEM ẢNH (GEMINI) ---")
            print(f"Câu trả lời của LLM: {answer}")
            print("------------------------------------------")
            return "CÓ" in answer
        except Exception as e:
            print(f"Lỗi khi kiểm tra ý định xem ảnh bằng Gemini: {e}")
            return False
    elif model_choice == "lmstudio":
        try:
            response = get_lmstudio_response(prompt)
            answer = response.strip().upper()
            print(f"--- KIỂM TRA Ý ĐỊNH XEM ẢNH (LM STUDIO) ---")
            print(f"Câu trả lời của LLM: {answer}")
            print("------------------------------------------")
            return "CÓ" in answer
        except Exception as e:
            print(f"Lỗi khi kiểm tra ý định xem ảnh bằng LM Studio: {e}")
            return False
    elif model_choice == "openai":
        try:
            openai = get_openai_model()
            if not openai:
                return False
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4000
            )
            answer = response.choices[0].message.content.strip().upper()
            print(f"--- KIỂM TRA Ý ĐỊNH XEM ẢNH (OPENAI) ---")
            print(f"Câu trả lời của LLM: {answer}")
            print("------------------------------------------")
            return "CÓ" in answer
        except Exception as e:
            print(f"Lỗi khi kiểm tra ý định xem ảnh bằng OpenAI: {e}")
            return False
    return False
    
def extract_query_from_history(user_query: str, history: list = None, model_choice: str = "gemini") -> Dict[str, str]:
    """
    Dùng LLM để phân tích lịch sử hội thoại và sinh ra từ khóa/truy vấn phù hợp cho Elasticsearch.
    """
    history_text = ""
    if history:
        for turn in history[-5:]:
            history_text += f"Khách: {turn['user']}\nBot: {turn['bot']}\n"
    
    prompt = (
        "Bạn là một trợ lý AI hỗ trợ tìm kiếm sản phẩm cho cửa hàng thiết bị/phụ kiện. Dưới đây là lịch sử hội thoại giữa khách hàng và bot:\n"
        f"{history_text}"
        f"Khách: {user_query}\n"
        "\n"
        "Nhiệm vụ của bạn:\n"
        "- Phân tích ý định của khách hàng và xác định rõ product_name (tên sản phẩm) và category (danh mục sản phẩm) phù hợp nhất để tìm kiếm trong kho hàng.\n"
        "- Nếu khách hàng hỏi 'shop có đèn kính hiển vi không' thì product_name là 'đèn kính hiển vi', category là 'đèn'.\n"
        "- Nếu khách hàng hỏi 'shop có kính hiển vi không' thì cả product_name và category đều là 'kính hiển vi'.\n"
        "- Nếu không xác định được category, hãy để category giống product_name.\n"
        "- Chỉ trả về kết quả ở dạng JSON với 2 trường: product_name và category. Không giải thích thêm, không thêm bất kỳ ký tự nào ngoài JSON.\n"
        "\n"
        "Ví dụ:\n"
        "Khách: shop có đèn kính hiển vi không\n"
        "Trả về: {\"product_name\": \"đèn kính hiển vi\", \"category\": \"đèn\"}\n"
        "Khách: shop có kính hiển vi không\n"
        "Trả về: {\"product_name\": \"kính hiển vi\", \"category\": \"kính hiển vi\"}\n"
    )
    
    if model_choice == "gemini":
        model = get_gemini_model()
        if model:
            try:
                response = model.generate_content(prompt)
                print("--- RESPONSE để tìm kiếm ---")
                print(response.text.strip())
                cleaned_response = response.text.strip().removeprefix("```json").removesuffix("```").strip()
                data = json.loads(cleaned_response)
                if 'product_name' in data and 'category' in data:
                    return data
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Lỗi khi parse JSON từ Gemini (extract_query): {e}")
            except Exception as e:
                print(f"Lỗi khi gọi Gemini (extract_query): {e}")
    elif model_choice == "lmstudio":
        try:
            response = get_lmstudio_response(prompt)
            print("--- RESPONSE để tìm kiếm (LM Studio) ---")
            print(response)
            # Tìm và trích xuất phần JSON từ phản hồi
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                cleaned_response = json_match.group(0)
                data = json.loads(cleaned_response)
                if 'product_name' in data and 'category' in data:
                    return data
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Lỗi khi parse JSON từ LM Studio (extract_query): {e}")
        except Exception as e:
            print(f"Lỗi khi gọi LM Studio (extract_query): {e}")
    elif model_choice == "openai":
        try:
            openai = get_openai_model()
            if not openai:
                return {"product_name": user_query, "category": user_query}
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4000
            )
            response_text = response.choices[0].message.content.strip()
            print("--- RESPONSE để tìm kiếm (OpenAI) ---")
            print(response_text)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                cleaned_response = json_match.group(0)
                data = json.loads(cleaned_response)
                if 'product_name' in data and 'category' in data:
                    return data
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Lỗi khi parse JSON từ OpenAI (extract_query): {e}")
        except Exception as e:
            print(f"Lỗi khi gọi OpenAI (extract_query): {e}")
        return {"product_name": user_query, "category": user_query}
    
    return {"product_name": user_query, "category": user_query}

def resolve_product_for_image(user_query: str, history: list, products: list, model_choice: str = "gemini") -> List[str]:
    """
    Sử dụng LLM để xác định (các) sản phẩm cụ thể mà người dùng muốn xem ảnh
    dựa trên ngữ cảnh và danh sách sản phẩm được cung cấp.
    """
    if not products:
        return []

    # Lấy danh sách tên sản phẩm kèm properties
    product_infos = [
        f"{p.get('product_name', '')} ({p.get('properties', '')})"
        for p in products
        if p.get('product_name')
    ]

    if not product_infos:
        return []

    # Xây dựng lịch sử hội thoại gần đây
    history_text = ""
    if history:
        for turn in history[-3:]:
            history_text += f"Khách: {turn['user']}\nBot: {turn['bot']}\n"

    # Tạo prompt đầu ra
    prompt = (
        "Bạn là một AI phân tích hội thoại khách hàng. Dưới đây là lịch sử hội thoại gần đây và danh sách các sản phẩm mà cửa hàng có. "
        "Nhiệm vụ của bạn: xác định khách hàng muốn xem ảnh của sản phẩm nào trong danh sách này. "
        "Chỉ trả về tên sản phẩm (mỗi tên trên một dòng), không giải thích gì thêm. Nếu không có sản phẩm nào phù hợp, trả về 'NONE'.\n"
        f"\nBối cảnh hội thoại gần đây:\n{history_text}"
        f"Câu hỏi của khách hàng: \"{user_query}\"\n"
        f"\nDanh sách sản phẩm:\n" + "\n".join(f"- {info}" for info in product_infos) + "\n"
        "\nTrả về tên sản phẩm muốn xem ảnh (mỗi tên một dòng, hoặc 'NONE'):"
    )

    response_text = None
    try:
        if model_choice == "gemini":
            model = get_gemini_model()
            if not model:
                return [product_infos[0]]
            response = model.generate_content(prompt)
            response_text = response.text.strip()
        elif model_choice == "lmstudio":
            response_text = get_lmstudio_response(prompt).strip()
        elif model_choice == "openai":
            openai = get_openai_model()
            if not openai:
                return [product_infos[0]]
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4000
            )
            response_text = response.choices[0].message.content.strip()
        else:
            return [product_infos[0]]
    except Exception as e:
        print(f"Lỗi khi gọi LLM (resolve_product_for_image): {e}")
        return [product_infos[0]]

    print(f"--- XÁC ĐỊNH SẢN PHẨM ĐỂ XEM ẢNH (LLM) ---")
    print(f"Prompt: {prompt}")
    print(f"LLM Response: {response_text}")
    print("------------------------------------------")

    if not response_text or response_text.strip().upper() == 'NONE':
        return []

    # Tách các dòng, loại bỏ dòng trống, chỉ giữ tên sản phẩm có trong danh sách
    resolved_names = [name.strip() for name in response_text.split('\n') if name.strip() in product_infos]
    # Nếu LLM trả về tên không khớp, fallback sản phẩm đầu tiên
    if not resolved_names:
        return [product_infos[0]]
    return resolved_names