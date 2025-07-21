from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import uuid
from datetime import datetime
import httpx
import logging

from database import get_db
from models import User, ChatSession, ChatMessage
from schemas import (
    ChatRequest, 
    ChatResponse, 
    ChatSessionWithMessages,
    UserProfileUpdate,
    ProfileResponse,
    MessageResponse,
    ChatSessionTitleUpdate
)
from utils.auth import get_current_active_user
from config import settings

# 로깅 설정
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# 챗봇 서비스는 별도로 통합 예정
# from jeju_chatbot.final_chatbot import JejuTravelChatbot

@router.post("", response_model=ChatResponse)
async def chat_with_bot(
    chat_request: ChatRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    챗봇과 대화하기 API
    """
    try:
        # Get or create session
        session_id = chat_request.session_id or str(uuid.uuid4())
        
        # Find existing session or create new one
        db_session = db.query(ChatSession).filter(
            ChatSession.session_id == session_id,
            ChatSession.user_id == current_user.id
        ).first()
        
        if not db_session:
            # 새로운 세션 생성 시 기본 제목 설정
            base_title = "새로운 채팅"
            existing_titles = db.query(ChatSession.title).filter(
                ChatSession.user_id == current_user.id,
                ChatSession.title.like(f"{base_title}%")
            ).all()
            
            new_title = base_title
            if existing_titles:
                new_title = f"{base_title} {len(existing_titles) + 1}"

            db_session = ChatSession(
                session_id=session_id,
                user_id=current_user.id,
                title=new_title, # <-- title 추가
                user_profile={},
                profile_completion=0
            )
            db.add(db_session)
            db.commit()
            db.refresh(db_session)
        
        # Save user message
        user_message = ChatMessage(
            session_id=session_id,
            message_type="user",
            content=chat_request.content,
            message_metadata={}
        )
        db.add(user_message)
        db.commit()
        logger.info(f"✅ 사용자 메시지 저장됨 - Session: {session_id}, Content: {chat_request.content[:50]}...")
        
        # 대화 히스토리 가져오기 (최근 10개 메시지)
        recent_messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.created_at.desc()).limit(10).all()
        
        # 대화 히스토리를 올바른 순서로 정렬
        conversation_history = []
        for msg in reversed(recent_messages):
            conversation_history.append({
                "role": "user" if msg.message_type == "user" else "assistant",
                "content": msg.content,
                "timestamp": msg.created_at.isoformat()
            })
        
        # 챗봇 서비스 호출
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                chatbot_response = await client.post(
                    f"{settings.CHATBOT_SERVICE_URL}/chat",
                    json={
                        "content": chat_request.content,
                        "session_id": session_id,
                        "conversation_history": conversation_history,
                        "user_profile": db_session.user_profile or {},
                        "profile_completion": db_session.profile_completion
                    }
                )
                chatbot_response.raise_for_status()
                chatbot_data = chatbot_response.json()
            
            # 챗봇 응답을 ChatResponse로 변환
            ai_response = ChatResponse(
                response=chatbot_data["response"],
                session_id=chatbot_data["session_id"],
                needs_more_info=chatbot_data["needs_more_info"],
                profile_completion=chatbot_data["profile_completion"],
                follow_up_questions=chatbot_data["follow_up_questions"],
                user_profile=chatbot_data["user_profile"],
                analysis_confidence=chatbot_data["analysis_confidence"],
                timestamp=chatbot_data["timestamp"]
            )
            
        except httpx.RequestError as e:
            logger.error(f"챗봇 서비스 연결 실패: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="챗봇 서비스에 연결할 수 없습니다. 서비스가 실행 중인지 확인해주세요."
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"챗봇 서비스 오류: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="챗봇 서비스에서 오류가 발생했습니다."
            )
        except Exception as e:
            logger.error(f"예상치 못한 오류: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="챗봇 처리 중 오류가 발생했습니다."
            )
        
        # Save bot response
        bot_message = ChatMessage(
            session_id=session_id,
            message_type="bot",
            content=ai_response.response,
            message_metadata={
                "needs_more_info": ai_response.needs_more_info,
                "profile_completion": ai_response.profile_completion,
                "analysis_confidence": ai_response.analysis_confidence
            }
        )
        db.add(bot_message)
        logger.info(f"✅ AI 응답 저장됨 - Session: {session_id}, Content: {ai_response.response[:50]}...")
        
        # Update session profile with actual data from chatbot
        db_session.profile_completion = int(ai_response.profile_completion * 100)
        db_session.user_profile = ai_response.user_profile
        logger.info(f"✅ 세션 업데이트 - Profile: {ai_response.profile_completion}%")
        
        db.commit()
        logger.info(f"✅ DB 커밋 완료 - Session: {session_id}")
        
        return ai_response
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="채팅 처리 중 오류가 발생했습니다."
        )

@router.get("/profile/{session_id}", response_model=ProfileResponse)
async def get_session_profile(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    세션 프로필 조회 API
    """
    db_session = db.query(ChatSession).filter(
        ChatSession.session_id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    
    if not db_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="세션을 찾을 수 없습니다."
        )
    
    return ProfileResponse(
        session_id=session_id,
        profile=db_session.user_profile,
        completion=db_session.profile_completion / 100.0
    )

@router.put("/profile/{session_id}", response_model=ProfileResponse)
async def update_session_profile(
    session_id: str,
    profile_update: UserProfileUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    세션 프로필 업데이트 API
    """
    db_session = db.query(ChatSession).filter(
        ChatSession.session_id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    
    if not db_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="세션을 찾을 수 없습니다."
        )
    
    # Update profile with provided data
    current_profile = db_session.user_profile or {}
    update_data = profile_update.dict(exclude_unset=True)
    current_profile.update(update_data)
    
    db_session.user_profile = current_profile
    
    # Calculate completion percentage (simplified logic)
    required_fields = ["companions", "duration", "budget", "interests", "transportation"]
    completed_fields = sum(1 for field in required_fields if current_profile.get(field))
    completion = (completed_fields / len(required_fields)) * 100
    db_session.profile_completion = int(completion)
    
    db.commit()
    
    return ProfileResponse(
        session_id=session_id,
        profile=current_profile,
        completion=completion / 100.0
    )

@router.delete("/session/{session_id}", response_model=MessageResponse)
async def reset_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    세션 리셋 API
    """
    # Delete all messages for this session
    db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
    
    # Reset session profile
    db_session = db.query(ChatSession).filter(
        ChatSession.session_id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    
    if db_session:
        db_session.user_profile = {}
        db_session.profile_completion = 0
    
    db.commit()
    
    return MessageResponse(message="세션이 성공적으로 리셋되었습니다.")

@router.get("/history/{session_id}", response_model=ChatSessionWithMessages)
async def get_chat_history(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    채팅 히스토리 조회 API
    """
    db_session = db.query(ChatSession).filter(
        ChatSession.session_id == session_id,
        ChatSession.user_id == current_user.id
    ).first()
    
    if not db_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="세션을 찾을 수 없습니다."
        )
    
    # Fetch messages for this session
    messages = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.created_at.asc()).all()
    
    logger.info(f"✅ 히스토리 조회 - Session: {session_id}, Messages: {len(messages)}개")
    
    # Convert to response format
    return ChatSessionWithMessages(
        session_id=db_session.session_id,
        user_id=db_session.user_id,
        title=db_session.title,  # title 필드 추가
        user_profile=db_session.user_profile,
        profile_completion=db_session.profile_completion,
        is_active=db_session.is_active,
        created_at=db_session.created_at,
        updated_at=db_session.updated_at,
        messages=messages
    )

@router.get("/sessions")
async def get_chat_sessions(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    사용자의 모든 채팅 세션 목록 가져오기
    """
    try:
        # Get all sessions for current user
        sessions = db.query(ChatSession).filter(
            ChatSession.user_id == current_user.id,
            ChatSession.is_active == True
        ).order_by(ChatSession.updated_at.desc()).all()
        
        session_list = []
        for session in sessions:
            # Get last message for preview
            last_message = db.query(ChatMessage).filter(
                ChatMessage.session_id == session.session_id
            ).order_by(ChatMessage.created_at.desc()).first()
            
            session_list.append({
                "session_id": session.session_id,
                "title": session.title, # <-- DB에서 실제 title 가져오기
                "last_message": last_message.content if last_message else "대화를 시작해보세요",
                "timestamp": session.updated_at.isoformat(),
                "profile_completion": session.profile_completion,
                "is_current": False  # Will be set by frontend
            })
        
        logger.info(f"✅ 세션 목록 조회 - User: {current_user.email}, Sessions: {len(session_list)}개")
        
        return {
            "sessions": session_list
        }
        
    except Exception as e:
        logger.error(f"세션 목록 조회 실패: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="세션 목록을 불러올 수 없습니다."
        )

# --- 채팅방 이름 변경 API 추가 ---
@router.put("/sessions/{session_id}/title", response_model=MessageResponse)
async def update_chat_session_title(
    session_id: str,
    title_update: ChatSessionTitleUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    채팅 세션의 제목을 변경합니다.
    """
    db_session = db.query(ChatSession).filter(
        ChatSession.session_id == session_id,
        ChatSession.user_id == current_user.id
    ).first()

    if not db_session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="세션을 찾을 수 없습니다."
        )

    db_session.title = title_update.title
    db.commit()

    return MessageResponse(message="채팅방 이름이 성공적으로 변경되었습니다.") 