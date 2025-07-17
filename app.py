"""
File app.py mới - sử dụng cấu trúc module đã được tổ chức lại
Chạy file này thay vì file app.py cũ
"""

from src.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8118, reload=True)