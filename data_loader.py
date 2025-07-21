"""
제주도 여행 데이터를 ChromaDB에 로드하는 스크립트
"""

import json
import os
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict

class JejuDataLoader:
    def __init__(self, db_path="./chroma_db"):
        """ChromaDB 클라이언트 초기화"""
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
    def load_json_data(self, file_path: str) -> List[Dict]:
        """JSON 파일 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def create_collection(self, name: str):
        """컬렉션 생성 (이미 존재하면 삭제 후 재생성)"""
        try:
            self.client.delete_collection(name)
        except:
            pass
        
        return self.client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def process_tourism_data(self, data: List[Dict]) -> tuple:
        """관광지 데이터 처리"""
        documents = []
        metadatas = []
        ids = []
        
        for idx, item in enumerate(data):
            # 문서 텍스트 생성 (검색용)
            doc_text = f"{item['이름']}: {item['소개']} 위치: {item['주소']} 태그: {item.get('태그', '')}"
            documents.append(doc_text)
            
            # 메타데이터 생성
            metadata = {
                "name": item['이름'],
                "address": item['주소'],
                "description": item['소개'],
                "tags": item.get('태그', ''),
                "phone": item.get('전화번호', ''),
                "category": "tourism"
            }
            metadatas.append(metadata)
            ids.append(f"tourism_{idx}")
            
        return documents, metadatas, ids
    
    def process_hotel_data(self, data: List[Dict]) -> tuple:
        """숙박 데이터 처리"""
        documents = []
        metadatas = []
        ids = []
        
        for idx, item in enumerate(data):
            doc_text = f"{item['이름']}: {item['소개']} 위치: {item['주소']} 태그: {item.get('태그', '')}"
            documents.append(doc_text)
            
            metadata = {
                "name": item['이름'],
                "address": item['주소'],
                "description": item['소개'],
                "tags": item.get('태그', ''),
                "phone": item.get('전화번호', ''),
                "category": "accommodation"
            }
            metadatas.append(metadata)
            ids.append(f"hotel_{idx}")
            
        return documents, metadatas, ids
    
    def process_food_data(self, data: List[Dict]) -> tuple:
        """음식점 데이터 처리"""
        documents = []
        metadatas = []
        ids = []
        
        for idx, item in enumerate(data):
            doc_text = f"{item['이름']}: {item['소개']} 위치: {item['주소']} 태그: {item.get('태그', '')}"
            documents.append(doc_text)
            
            metadata = {
                "name": item['이름'],
                "address": item['주소'],
                "description": item['소개'],
                "tags": item.get('태그', ''),
                "phone": item.get('전화번호', ''),
                "category": "restaurant"
            }
            metadatas.append(metadata)
            ids.append(f"food_{idx}")
            
        return documents, metadatas, ids
    
    def process_event_data(self, data: List[Dict]) -> tuple:
        """이벤트 데이터 처리"""
        documents = []
        metadatas = []
        ids = []
        
        for idx, item in enumerate(data):
            doc_text = f"{item['이름']}: {item['소개']} 위치: {item['주소']} 태그: {item.get('태그', '')}"
            documents.append(doc_text)
            
            metadata = {
                "name": item['이름'],
                "address": item['주소'],
                "description": item['소개'],
                "tags": item.get('태그', ''),
                "phone": item.get('전화번호', ''),
                "category": "event"
            }
            metadatas.append(metadata)
            ids.append(f"event_{idx}")
            
        return documents, metadatas, ids
    
    def load_all_data(self):
        """모든 데이터를 ChromaDB에 로드"""
        print("🚀 제주도 여행 데이터 로딩 시작...")
        
        # 통합 컬렉션 생성
        collection = self.create_collection("jeju_travel")
        
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        # 각 데이터 타입별 처리
        data_files = {
            "tourism": "./data/visitjeju_tour.json",
            "hotel": "./data/visitjeju_hotel.json", 
            "food": "./data/visitjeju_food.json",
            "event": "./data/visitjeju_event.json"
        }
        
        processors = {
            "tourism": self.process_tourism_data,
            "hotel": self.process_hotel_data,
            "food": self.process_food_data,
            "event": self.process_event_data
        }
        
        for data_type, file_path in data_files.items():
            if os.path.exists(file_path):
                print(f"📊 {data_type} 데이터 처리 중...")
                data = self.load_json_data(file_path)
                processor = processors[data_type]
                docs, metas, ids = processor(data)
                
                all_documents.extend(docs)
                all_metadatas.extend(metas)
                all_ids.extend(ids)
                
                print(f"✅ {data_type}: {len(docs)}개 항목 처리 완료")
        
        # ChromaDB에 모든 데이터 추가
        print(f"💾 총 {len(all_documents)}개 항목을 ChromaDB에 저장 중...")
        
        # 배치 사이즈로 나누어 저장 (ChromaDB 제한 고려)
        batch_size = 1000
        for i in range(0, len(all_documents), batch_size):
            batch_docs = all_documents[i:i+batch_size]
            batch_metas = all_metadatas[i:i+batch_size]
            batch_ids = all_ids[i:i+batch_size]
            
            collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
            print(f"📦 배치 {i//batch_size + 1} 저장 완료")
        
        print("🎉 모든 데이터 로딩 완료!")
        return collection
    
    def test_search(self, query: str = "바다 근처 맛집"):
        """검색 테스트"""
        collection = self.client.get_collection("jeju_travel")
        results = collection.query(
            query_texts=[query],
            n_results=5
        )
        
        print(f"\n🔍 검색어: '{query}'")
        print("검색 결과:")
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            print(f"\n{i+1}. {metadata['name']} ({metadata['category']})")
            print(f"   주소: {metadata['address']}")
            print(f"   설명: {metadata['description'][:100]}...")

if __name__ == "__main__":
    loader = JejuDataLoader()
    
    # 데이터 로드
    collection = loader.load_all_data()
    
    # 검색 테스트
    loader.test_search("바다 근처 맛집")
    loader.test_search("가족 여행 숙소")
    loader.test_search("아이들 체험 관광지") 