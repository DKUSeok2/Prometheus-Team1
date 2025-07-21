#!/usr/bin/env python3
"""
데이터베이스 테이블을 재생성하는 스크립트
기존 테이블을 삭제하고 새로운 스키마로 다시 생성합니다.
"""

from database import drop_tables, create_tables
import models  # 모델을 import해야 Base.metadata에 등록됨

def reset_database():
    """데이터베이스 테이블 재생성"""
    print("🗑️  기존 테이블 삭제 중...")
    drop_tables()
    
    print("🔨 새로운 테이블 생성 중...")
    create_tables()
    
    print("✅ 데이터베이스 재생성 완료!")

if __name__ == "__main__":
    reset_database() 