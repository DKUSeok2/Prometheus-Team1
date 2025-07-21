from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session

from database import get_db
from models import User
from schemas import UserCreate, UserLogin, User as UserSchema, Token, MessageResponse, EmailCheckResponse
from utils.auth import (
    authenticate_user, 
    create_access_token, 
    get_password_hash,
    get_user_by_email,
    get_current_active_user
)
from config import settings

router = APIRouter(prefix="/auth", tags=["authentication"])
security = HTTPBearer()

@router.post("/register", response_model=UserSchema)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    회원가입 API
    """
    # Check if user already exists
    existing_user = get_user_by_email(db, email=user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="이미 등록된 이메일입니다."
        )
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        email=user_data.email,
        hashed_password=hashed_password,
        is_active=True
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

@router.get("/check-email", response_model=EmailCheckResponse)
async def check_email_availability(email: str, db: Session = Depends(get_db)):
    """
    이메일 중복 확인 API
    """
    existing_user = get_user_by_email(db, email=email)
    if existing_user:
        return EmailCheckResponse(
            available=False,
            message="이미 사용 중인 이메일입니다."
        )
    else:
        return EmailCheckResponse(
            available=True,
            message="사용 가능한 이메일입니다."
        )

@router.post("/login", response_model=Token)
async def login_user(login_data: UserLogin, db: Session = Depends(get_db)):
    """
    로그인 API
    """
    user = authenticate_user(db, login_data.email, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="이메일 또는 비밀번호가 틀렸습니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return Token(access_token=access_token, token_type="bearer")

@router.get("/validate", response_model=UserSchema)
async def validate_token(current_user: User = Depends(get_current_active_user)):
    """
    토큰 검증 API
    """
    return current_user

@router.get("/me", response_model=UserSchema) 
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """
    현재 사용자 정보 조회 API
    """
    return current_user 