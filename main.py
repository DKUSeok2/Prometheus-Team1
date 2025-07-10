#!/usr/bin/env python3
"""
오르다 - 제주도 여행 추천 챗봇
메인 애플리케이션 진입점

실행 방법:
streamlit run main.py
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import settings
from src.ui import streamlit_app

def main():
    """메인 함수"""
    try:
        # 설정 검증
        if not settings.validate_settings():
            print("❌ 설정 검증 실패. 환경 변수를 확인해주세요.")
            print("\n필요한 환경 변수:")
            print("- UPSTAGE_API_KEY: Upstage API 키 (필수)")
            print("- KAKAO_API_KEY: 카카오 REST API 키 (선택사항)")
            print("- KAKAO_JS_KEY: 카카오 JavaScript 키 (지도 표시용, 선택사항)")
            print("\n.env 파일을 생성하고 위 변수들을 설정해주세요.")
            return
        
        print("🚀 오르다 챗봇을 시작합니다...")
        print(f"📍 ChromaDB 경로: {settings.chromadb_path}")
        print(f"💬 채팅 기록 경로: {settings.chat_history_path}")
        print(f"🤖 Ollama 모델: {settings.ollama_model}")
        
        # Streamlit 앱 실행
        streamlit_app.run()
        
    except KeyboardInterrupt:
        print("\n👋 애플리케이션을 종료합니다.")
    except Exception as e:
        print(f"❌ 애플리케이션 실행 중 오류 발생: {e}")
        print("문제가 지속되면 GitHub 이슈를 등록해주세요.")

if __name__ == "__main__":
    main()
