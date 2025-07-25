from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import threading
import time

from src.config.settings import APP_CONFIG, CORS_CONFIG
from src.models.schemas import ChatRequest
from src.api.routes import chat_endpoint, chat_history, chat_history_lock, HANDOVER_TIMEOUT

# Khởi tạo FastAPI app
app = FastAPI(**APP_CONFIG)

# Thêm CORS middleware
app.add_middleware(CORSMiddleware, **CORS_CONFIG)

def session_timeout_scanner():
    """
    Quét và reset các session bị timeout trong một luồng nền.
    """
    while True:
        print("Chạy tác vụ nền: Quét các session timeout...")
        with chat_history_lock:
            current_time = time.time()
            sessions_to_reactivate = []
            for session_id, session_data in chat_history.items():
                if session_data.get("state") == "human_handover":
                    handover_time = session_data.get("handover_timestamp", 0)
                    if (current_time - handover_time) > HANDOVER_TIMEOUT:
                        sessions_to_reactivate.append(session_id)
            
            for session_id in sessions_to_reactivate:
                print(f"Session {session_id} đã quá hạn. Kích hoạt lại bot.")
                chat_history[session_id]["state"] = None
                chat_history[session_id]["negativity_score"] = 0
                chat_history[session_id]["messages"].append({
                    "user": "[SYSTEM]",
                    "bot": "Bot đã được tự động kích hoạt lại do không có hoạt động."
                })
        
        time.sleep(300)


# Định nghĩa các routes
@app.on_event("startup")
async def startup_event():
    """
    Tạo luồng nền để quét các session bị timeout.
    """
    scanner_thread = threading.Thread(target=session_timeout_scanner, daemon=True)
    scanner_thread.start()
    print("Đã khởi động tác vụ nền để quét session timeout.")

@app.post("/chat", summary="Gửi tin nhắn đến chatbot")
async def chat(request: ChatRequest, session_id: str = Query("default", description="ID phiên chat")):
    """
    Endpoint chính để tương tác với chatbot.
    - **message**: Câu hỏi của người dùng.
    - **session_id**: ID phiên chat (mặc định là 'default')
    """
    return await chat_endpoint(request, session_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="127.0.0.1", port=8111, reload=True)