#!/usr/bin/env python3
"""
제주도 여행 챗봇 데이터베이스 초기화 스크립트
ChromaDB에 제주도 여행 데이터를 로드합니다.
"""

import json
import os
import chromadb
from langchain_upstage import UpstageEmbeddings
from typing import List, Dict
from config import (
    UPSTAGE_API_KEY, EMBEDDING_MODEL,
    CHROMA_DB_PATH, COLLECTION_NAME
)

def load_json_data(file_path: str) -> List[Dict]:
    """JSON 파일에서 데이터를 로드합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print(f"✅ {file_path} 로드 완료: {len(data)}개 항목")
            return data
    except Exception as e:
        print(f"❌ {file_path} 로드 실패: {e}")
        return []

def setup_chroma_collection():
    """ChromaDB 컬렉션을 초기화합니다."""
    try:
        # ChromaDB 클라이언트 생성
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # 기존 컬렉션이 있다면 삭제
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print(f"🗑️ 기존 컬렉션 '{COLLECTION_NAME}' 삭제됨")
        except:
            pass
        
        # 임베딩 모델 초기화
        embeddings = UpstageEmbeddings(
            api_key=UPSTAGE_API_KEY,
            model=EMBEDDING_MODEL
        )
        
        # 새 컬렉션 생성
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"✅ 컬렉션 '{COLLECTION_NAME}' 생성 완료")
        return collection, embeddings
        
    except Exception as e:
        print(f"❌ ChromaDB 초기화 실패: {e}")
        return None, None

def process_and_embed_data(collection, embeddings, data_list: List[Dict], category: str):
    """데이터를 처리하고 임베딩하여 컬렉션에 추가합니다."""
    documents = []
    metadatas = []
    ids = []
    
    for i, item in enumerate(data_list):
        try:
            # 텍스트 내용 생성
            if category in ['food', 'hotel']:
                text_content = f"""
                이름: {item.get('name', '')}
                주소: {item.get('address', '')}
                전화번호: {item.get('phone', '')}
                태그: {item.get('tags', '')}
                설명: {item.get('description', '')}
                """
            else:  # event, tour
                text_content = f"""
                이름: {item.get('name', '')}
                주소: {item.get('address', '')}
                전화번호: {item.get('phone', '')}
                설명: {item.get('overview', item.get('description', ''))}
                """
            
            # 메타데이터 생성
            metadata = {
                'category': category,
                'name': item.get('name', ''),
                'address': item.get('address', ''),
                'phone': item.get('phone', ''),
                'tags': item.get('tags', ''),
            }
            
            documents.append(text_content.strip())
            metadatas.append(metadata)
            ids.append(f"{category}_{i}")
            
        except Exception as e:
            print(f"⚠️ 데이터 처리 오류 (항목 {i}): {e}")
            continue
    
    if documents:
        try:
            # 임베딩 생성
            print(f"🔄 {category} 데이터 임베딩 중... ({len(documents)}개 항목)")
            embeddings_list = embeddings.embed_documents(documents)
            
            # 컬렉션에 추가
            collection.add(
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"✅ {category} 데이터 {len(documents)}개 항목 추가 완료")
            
        except Exception as e:
            print(f"❌ {category} 데이터 임베딩 실패: {e}")

def main():
    """메인 함수"""
    print("🌴 제주도 여행 챗봇 데이터베이스 초기화 시작")
    print("=" * 60)
    
    # API 키 확인
    if not UPSTAGE_API_KEY or UPSTAGE_API_KEY == "your_upstage_api_key_here":
        print("❌ UPSTAGE_API_KEY가 설정되지 않았습니다.")
        print("💡 .env 파일을 생성하고 API 키를 설정해주세요.")
        return
    
    # ChromaDB 컬렉션 초기화
    collection, embeddings = setup_chroma_collection()
    if not collection or not embeddings:
        return
    
    # 데이터 파일 경로
    data_files = {
        'food': 'data/visitjeju_food.json',
        'hotel': 'data/visitjeju_hotel.json',
        'event': 'data/visitjeju_event.json',
        'tour': 'data/visitjeju_tour.json'
    }
    
    # 각 카테고리별 데이터 로드 및 처리
    total_items = 0
    for category, file_path in data_files.items():
        if os.path.exists(file_path):
            data = load_json_data(file_path)
            if data:
                process_and_embed_data(collection, embeddings, data, category)
                total_items += len(data)
        else:
            print(f"⚠️ 파일을 찾을 수 없습니다: {file_path}")
    
    print("=" * 60)
    print(f"🎉 데이터베이스 초기화 완료!")
    print(f"📊 총 {total_items}개 항목이 로드되었습니다.")
    print(f"💾 데이터베이스 위치: {CHROMA_DB_PATH}")
    print(f"🏷️ 컬렉션 이름: {COLLECTION_NAME}")
    
    # 테스트 검색
    print("\n🧪 테스트 검색 실행...")
    try:
        test_results = collection.query(
            query_texts=["제주도 맛집"],
            n_results=3
        )
        print(f"✅ 테스트 검색 성공: {len(test_results['documents'][0])}개 결과")
    except Exception as e:
        print(f"⚠️ 테스트 검색 실패: {e}")
    
    print("\n🚀 이제 interactive_demo.py를 실행할 수 있습니다!")

if __name__ == "__main__":
    main() 