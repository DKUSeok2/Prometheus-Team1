#!/usr/bin/env python3
"""
제주도 데이터 로딩 스크립트

사용법:
python scripts/load_data.py [data_directory]
"""

import sys
import os
import argparse
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database import chromadb_manager
from src.config import settings

def load_jeju_data(data_directory: str):
    """제주도 데이터 로딩"""
    
    # 파일-카테고리 매핑 (README.md에서 참조)
    category_map = {
        "visitjeju_food.json": "음식",
        "visitjeju_hotel.json": "숙소", 
        "visitjeju_tour.json": "관광지",
        "visitjeju_event.json": "행사"
    }
    
    data_path = Path(data_directory)
    
    if not data_path.exists():
        print(f"❌ 데이터 디렉토리가 존재하지 않습니다: {data_directory}")
        return False
    
    print("🚀 제주도 데이터 로딩을 시작합니다...")
    print(f"📂 데이터 경로: {data_directory}")
    print(f"📍 ChromaDB 경로: {settings.chromadb_path}")
    
    success_count = 0
    total_files = len(category_map)
    
    for filename, category in category_map.items():
        file_path = data_path / filename
        
        if not file_path.exists():
            print(f"⚠️  파일을 찾을 수 없습니다: {filename}")
            continue
        
        print(f"\n📝 {filename} 로딩 중... (카테고리: {category})")
        
        try:
            success = chromadb_manager.load_data_from_json(str(file_path), category)
            
            if success:
                print(f"✅ {filename} 로딩 완료")
                success_count += 1
            else:
                print(f"❌ {filename} 로딩 실패")
                
        except Exception as e:
            print(f"❌ {filename} 로딩 중 오류: {e}")
    
    print(f"\n📊 로딩 완료: {success_count}/{total_files} 파일 성공")
    
    # 최종 통계
    try:
        stats = chromadb_manager.get_collection_stats()
        print(f"📈 총 저장된 문서 수: {stats['total_documents']}")
    except Exception as e:
        print(f"⚠️  통계 조회 중 오류: {e}")
    
    return success_count == total_files

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="제주도 데이터 로딩 스크립트")
    parser.add_argument(
        "data_directory",
        nargs="?",
        default="./data",
        help="JSON 데이터 파일들이 있는 디렉토리 경로 (기본값: ./data)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 데이터를 삭제하고 새로 로딩"
    )
    
    args = parser.parse_args()
    
    try:
        # 설정 검증
        if not settings.validate_settings():
            print("❌ 설정 검증 실패. 환경 변수를 확인해주세요.")
            return 1
        
        # 기존 데이터 초기화 (옵션)
        if args.reset:
            print("🔄 기존 데이터를 초기화합니다...")
            # 여기에 컬렉션 삭제 로직 추가 가능
        
        # 데이터 로딩
        success = load_jeju_data(args.data_directory)
        
        if success:
            print("\n🎉 모든 데이터 로딩이 완료되었습니다!")
            return 0
        else:
            print("\n⚠️  일부 데이터 로딩에 실패했습니다.")
            return 1
            
    except KeyboardInterrupt:
        print("\n👋 데이터 로딩을 중단합니다.")
        return 1
    except Exception as e:
        print(f"❌ 데이터 로딩 중 오류 발생: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
