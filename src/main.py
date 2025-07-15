from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import APP_CONFIG, CORS_CONFIG
from src.models.schemas import ChatRequest
from src.api.routes import chat_endpoint, health_check

# Khởi tạo FastAPI app
app = FastAPI(**APP_CONFIG)

# Thêm CORS middleware
app.add_middleware(CORSMiddleware, **CORS_CONFIG)

# Định nghĩa các routes
@app.post("/chat", summary="Gửi tin nhắn đến chatbot")
async def chat(request: ChatRequest, session_id: str = Query("default", description="ID phiên chat")):
    """
    Endpoint chính để tương tác với chatbot.
    - **message**: Câu hỏi của người dùng.
    - **session_id**: ID phiên chat (mặc định là 'default')
    """
    return await chat_endpoint(request, session_id)

@app.get("/", summary="Kiểm tra trạng thái API")
def read_root():
    return health_check()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="127.0.0.1", port=8111, reload=True)