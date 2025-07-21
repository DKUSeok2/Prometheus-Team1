from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, List, Any
from datetime import datetime

# ============ Auth Schemas ============
class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserLogin(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    email: Optional[str] = None

# ============ Chat Schemas ============
class ChatMessageBase(BaseModel):
    content: str

class ChatRequest(ChatMessageBase):
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    needs_more_info: bool
    profile_completion: float
    follow_up_questions: List[str]
    user_profile: Dict[str, Any]
    analysis_confidence: float
    timestamp: str

class ChatMessage(ChatMessageBase):
    id: int
    session_id: str
    message_type: str  # "user" or "bot"
    message_metadata: Dict[str, Any]
    created_at: datetime
    
    class Config:
        from_attributes = True

# ============ Session Schemas ============
class ChatSessionBase(BaseModel):
    session_id: str
    user_id: int
    title: str  # <-- title 필드 추가
    user_profile: Dict[str, Any] = {}
    profile_completion: int = 0
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class ChatSessionWithMessages(ChatSessionBase):
    messages: List[ChatMessage] = []

class ChatSessionTitleUpdate(BaseModel): # <-- 제목 변경용 스키마 추가
    title: str

# ============ Profile Schemas ============
class UserProfileUpdate(BaseModel):
    companions: Optional[str] = None
    duration: Optional[str] = None
    budget: Optional[str] = None
    interests: Optional[List[str]] = None
    transportation: Optional[str] = None
    accommodation_preference: Optional[str] = None

class ProfileResponse(BaseModel):
    session_id: str
    profile: Dict[str, Any]
    completion: float

# ============ Generic Response Schemas ============
class MessageResponse(BaseModel):
    message: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

class EmailCheckResponse(BaseModel):
    available: bool
    message: str 