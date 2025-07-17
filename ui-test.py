import streamlit as st
import requests
import json
import uuid
import os
from dotenv import load_dotenv

# Tải biến môi trường từ file .env
load_dotenv()

# --- Responsive & Safari-friendly UI ---
st.set_page_config(page_title="Chatbot Bán Hàng", page_icon="🤖", layout="wide")

# safari_friendly_css = """
# <style>
# /* Safari iOS fix input bị che */
# html, body {
#     height: 100%;
#     overflow-x: hidden;
#     -webkit-overflow-scrolling: touch;
#     position: relative;
# }

# .block-container {
#     display: flex;
#     flex-direction: column;
#     height: 100%;
#     padding-bottom: 180px !important; /* Tăng padding để tránh nội dung bị che */
# }

# section.main {
#     flex-grow: 1;
#     overflow-y: auto;
#     padding-bottom: 180px !important; /* Tăng padding để tránh nội dung bị che */
#     margin-bottom: 80px !important; /* Thêm margin để đảm bảo khoảng cách */
# }

# /* Đảm bảo input hiển thị nổi ở cuối và không bị che */
# .stChatInput {
#     position: fixed !important;
#     bottom: 0 !important;
#     left: 0;
#     right: 0;
#     z-index: 99999 !important; /* Tăng z-index lên cao hơn */
#     background: white;
#     padding: 1rem !important;
#     box-shadow: 0 -2px 6px rgba(0,0,0,0.1);
#     max-height: 80px;
# }

# /* Style cho input để dễ gõ */
# .stChatInput input {
#     font-size: 1.1rem !important;
#     padding: 15px !important;
#     border-radius: 16px !important;
#     width: 100% !important;
#     box-sizing: border-box !important;
#     background: #fff !important;
#     -webkit-appearance: none !important;
#     border: 1px solid #ccc;
#     position: relative !important;
#     z-index: 100000 !important; /* Tăng z-index cho input */
# }

# /* Fix cho Safari iOS */
# @supports (-webkit-touch-callout: none) {
#     .stChatInput {
#         padding-bottom: calc(1rem + env(safe-area-inset-bottom)) !important;
#         bottom: 0 !important;
#     }
    
#     body {
#         padding-bottom: env(safe-area-inset-bottom);
#     }
# }
# </style>
# """
# st.markdown(safari_friendly_css, unsafe_allow_html=True)

# --- Unique session ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- Model choice ---
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "gemini"

# --- Config ---
FASTAPI_URL = "http://127.0.0.1:8111/chat"

# --- Header ---
st.title("🤖 Chatbot Tư Vấn Bán Hàng")
st.caption("Trợ lý ảo thông minh cho cửa hàng của bạn")

# --- Model Selection ---
st.sidebar.title("Cài đặt")
model_choice = st.sidebar.radio(
    "Chọn mô hình AI:",
    ["Gemini", "LM Studio", "OpenAI"],
    index=0,
    help="Chọn mô hình AI để xử lý câu hỏi của bạn"
)

# Cập nhật lựa chọn mô hình
if model_choice == "Gemini":
    st.session_state.model_choice = "gemini"
elif model_choice == "LM Studio":
    st.session_state.model_choice = "lmstudio"
else:
    st.session_state.model_choice = "openai"

# Hiển thị thông tin về mô hình đang sử dụng
model_info = {
    "gemini": "Gemini",
    "lmstudio": "LM Studio",
    "openai": "OpenAI"
}[st.session_state.model_choice]
st.sidebar.info(f"Đang sử dụng: {model_info}")

# Hiển thị trạng thái kết nối
if st.session_state.model_choice == "lmstudio":
    try:
        # Lấy URL từ biến môi trường hoặc sử dụng giá trị mặc định
        lmstudio_url = os.getenv("LMSTUDIO_API_URL")
        response = requests.get(f"{lmstudio_url}/v1/models", timeout=2)
        if response.status_code == 200:
            st.sidebar.success(f"✅ LM Studio đã kết nối tại {lmstudio_url}")
        else:
            st.sidebar.error("❌ LM Studio không phản hồi")
    except Exception as e:
        st.sidebar.error(f"❌ Không thể kết nối đến LM Studio: {str(e)}")
elif st.session_state.model_choice == "openai":
    try:
        import openai
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.sidebar.error("❌ Thiếu OPENAI_API_KEY trong .env")
        else:
            client = openai.OpenAI(api_key=api_key)
            # Thử gọi models.list để kiểm tra kết nối
            models = client.models.list()
            st.sidebar.success("✅ OpenAI API key hợp lệ và kết nối thành công")
    except Exception as e:
        st.sidebar.error(f"❌ Không thể kết nối đến OpenAI: {str(e)}")

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Chào anh/chị, em có thể giúp gì cho anh/chị không ạ?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat input handler ---
if prompt := st.chat_input("Bạn cần tìm sản phẩm nào?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        model_name = "Gemini" if st.session_state.model_choice == "gemini" else "LM Studio"
        message_placeholder.markdown(f"Cửa hàng đang tìm kiếm thông tin với {model_name}... ⏳")

        try:
            payload = {
                "message": prompt, 
                "session_id": st.session_state.session_id,
                "model_choice": st.session_state.model_choice
            }
            response = requests.post(FASTAPI_URL, json=payload)
            response.raise_for_status()

            bot_reply = response.json().get("reply", "Xin lỗi, đã có lỗi xảy ra.")
            message_placeholder.markdown(bot_reply)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})

        except requests.exceptions.RequestException as e:
            model_name = "Gemini" if st.session_state.model_choice == "gemini" else "LM Studio"
            error_message = f"Lỗi kết nối đến backend khi sử dụng {model_name}: {e}"
            st.error(error_message)
            error_content = f"Rất xin lỗi, hệ thống đang gặp sự cố kết nối khi sử dụng {model_name}. Vui lòng thử lại sau hoặc chọn mô hình khác."
            st.session_state.messages.append({"role": "assistant", "content": error_content})
        except json.JSONDecodeError:
            model_name = "Gemini" if st.session_state.model_choice == "gemini" else "LM Studio"
            st.error(f"Lỗi: Không thể giải mã phản hồi từ server khi sử dụng {model_name}.")
            error_content = f"Rất xin lỗi, hệ thống đang gặp sự cố dữ liệu khi sử dụng {model_name}. Vui lòng thử lại sau hoặc chọn mô hình khác."
            st.session_state.messages.append({"role": "assistant", "content": error_content})
