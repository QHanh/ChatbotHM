# Chatbot Tư Vấn Bán Hàng

Ứng dụng chatbot tư vấn bán hàng sử dụng Elasticsearch và hai mô hình AI: Google Gemini và LM Studio (Gemma-3n).

## Tính năng

- Tìm kiếm sản phẩm thông minh với Elasticsearch
- Hỗ trợ hai mô hình AI: Google Gemini và LM Studio (Gemma-3n)
- Giao diện người dùng thân thiện với Streamlit
- Lưu trữ lịch sử hội thoại
- Phân tích ý định người dùng

## Cài đặt

1. Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

2. Cấu hình API key trong file `.env`:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

## Chạy ứng dụng

1. Khởi động backend FastAPI:

```bash
python app.py
```

2. Khởi động frontend Streamlit:

```bash
streamlit run ui-test.py
```

## Sử dụng LM Studio

1. Tải và cài đặt LM Studio từ [trang chủ](https://lmstudio.ai/)
2. Tải mô hình Gemma-3n-e4b-it-text
3. Chạy mô hình với API server (mặc định: 192.168.1.73:1234)
4. Cấu hình URL và model trong file `.env`:

```
LMSTUDIO_API_URL=http://192.168.1.73:1234
LMSTUDIO_MODEL=gemma-3n-e4b-it-text
```

5. Trong giao diện Streamlit, chọn "LM Studio (Gemma-3n)" trong sidebar

## Cấu trúc dự án

- `app.py`: Backend FastAPI
- `ui-test.py`: Frontend Streamlit
- `elastic_search_push_data.py`: Kết nối và đẩy dữ liệu vào Elasticsearch
- `requirements.txt`: Danh sách thư viện cần thiết

## Lưu ý

- Đảm bảo Elasticsearch đang chạy trước khi khởi động ứng dụng
- Khi sử dụng LM Studio, đảm bảo API server đang chạy ở địa chỉ 192.168.1.73:1234