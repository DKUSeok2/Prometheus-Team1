from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from datetime import datetime
import logging

# 챗봇 클래스 import
from final_chatbot import JejuTravelChatbot, ChatResponse as ChatbotResponse

# FastAPI 앱 생성
app = FastAPI(
    title="Jeju Travel Chatbot Service",
    description="제주도 여행 AI 챗봇 마이크로서비스",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용, 프로덕션에서는 제한 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 챗봇 인스턴스 생성
chatbot = JejuTravelChatbot()

# Request/Response 스키마
class ChatRequest(BaseModel):
    content: str
    session_id: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = []
    user_profile: Dict[str, Any] = {}
    profile_completion: int = 0

class ChatServiceResponse(BaseModel):
    response: str
    session_id: str
    needs_more_info: bool
    profile_completion: float
    follow_up_questions: List[str]
    user_profile: Dict[str, Any]
    analysis_confidence: float
    timestamp: str

# 서비스 시작 시 로그
@app.on_event("startup")
async def startup_event():
    logger.info("🤖 Jeju Travel Chatbot Service started!")
    logger.info("🔗 Service running on: http://localhost:8001")
    logger.info("📚 API Docs: http://localhost:8001/docs")

# 챗봇 엔드포인트
@app.post("/chat", response_model=ChatServiceResponse)
async def chat_endpoint(request: ChatRequest):
    """
    챗봇과 대화하기 API
    """
    try:
        logger.info(f"💬 New chat request: {request.content[:50]}...")
        
        # 챗봇 호출 (추가 정보 포함)
        response: ChatbotResponse = chatbot.chat_with_context(
            user_message=request.content,
            session_id=request.session_id or "default",
            conversation_history=request.conversation_history,
            existing_profile=request.user_profile,
            profile_completion=request.profile_completion
        )
        
        # 응답 변환 (session_id 추가)
        session_id_used = request.session_id or "default"
        service_response = ChatServiceResponse(
            response=response.response,
            session_id=session_id_used,
            needs_more_info=response.needs_more_info,
            profile_completion=response.profile_completion,
            follow_up_questions=response.follow_up_questions,
            user_profile=response.user_profile,
            analysis_confidence=response.analysis_confidence,
            timestamp=response.timestamp
        )
        
        logger.info(f"✅ Chat response generated for session: {session_id_used}")
        return service_response
        
    except Exception as e:
        logger.error(f"❌ Error in chat service: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="챗봇 서비스에서 오류가 발생했습니다."
        )

# 건강 상태 확인
@app.get("/health")
async def health_check():
    """챗봇 서비스 상태 확인"""
    return {
        "status": "healthy",
        "service": "jeju-chatbot",
        "timestamp": datetime.now().isoformat()
    }

# 사용자 프로필 조회
@app.get("/profile/{session_id}")
async def get_user_profile(session_id: str):
    """세션별 사용자 프로필 조회"""
    try:
        profile = chatbot.get_user_profile(session_id)
        return {
            "session_id": session_id,
            "profile": profile
        }
    except Exception as e:
        logger.error(f"Error getting profile: {str(e)}")
        raise HTTPException(status_code=404, detail="프로필을 찾을 수 없습니다.")

# 세션 리셋
@app.delete("/session/{session_id}")
async def reset_session(session_id: str):
    """세션 리셋"""
    try:
        chatbot.reset_session(session_id)
        return {"message": f"세션 {session_id}이 리셋되었습니다."}
    except Exception as e:
        logger.error(f"Error resetting session: {str(e)}")
        raise HTTPException(status_code=500, detail="세션 리셋 중 오류가 발생했습니다.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "chatbot_service:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    ) 