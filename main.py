from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime

from config import settings
from database import create_tables, get_db
from routers import auth, chat
from schemas import HealthResponse

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="제주도 여행 AI 챗봇 API",
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Create database tables on startup
@app.on_event("startup")
async def startup_event():
    """Create database tables on startup"""
    create_tables()
    print("🚀 FastAPI server started!")
    print(f"📊 Database: Connected to PostgreSQL")
    print(f"🔗 API Documentation: http://localhost:8000/docs")

# Include routers
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(chat.router, prefix=settings.API_V1_STR)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "제주도 여행 AI 챗봇 API",
        "version": settings.PROJECT_VERSION,
        "docs": "/docs",
        "status": "running"
    }

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version=settings.PROJECT_VERSION
    )

# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "요청하신 리소스를 찾을 수 없습니다."}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "서버 내부 오류가 발생했습니다."}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    ) 