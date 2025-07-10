"""
ChromaDB 클라이언트 및 데이터 관리 모듈
"""
import os
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from langchain_upstage import UpstageEmbeddings

from ..config import settings

class UpstageEmbeddingFunction(EmbeddingFunction):
    """ChromaDB에서 사용할 Upstage 임베딩 함수 래퍼"""
    
    def __init__(self, embedder):
        self.embedder = embedder

    def __call__(self, texts):
        try:
            return [self.embedder.embed_query(text) for text in texts]
        except Exception as e:
            raise RuntimeError(f"임베딩 생성 중 오류 발생: {e}")

class ChromaDBManager:
    """ChromaDB 관리 클래스"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.query_embedder = None
        self.passage_embedder = None
        self._initialize_embedders()
        self._initialize_client()
    
    def _initialize_embedders(self):
        """임베딩 모델 초기화"""
        try:
            self.query_embedder = UpstageEmbeddings(
                model=settings.query_embedding_model,
                api_key=settings.upstage_api_key
            )
            self.passage_embedder = UpstageEmbeddings(
                model=settings.passage_embedding_model,
                api_key=settings.upstage_api_key
            )
        except Exception as e:
            raise RuntimeError(f"Upstage 임베딩 로드 중 오류 발생: {e}")
    
    def _initialize_client(self):
        """ChromaDB 클라이언트 초기화"""
        try:
            self.client = chromadb.PersistentClient(path=settings.chromadb_path)
            self.collection = self.client.get_or_create_collection(
                name=settings.collection_name,
                embedding_function=UpstageEmbeddingFunction(self.passage_embedder)
            )
        except Exception as e:
            raise RuntimeError(f"ChromaDB 초기화 중 오류 발생: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """컬렉션 통계 정보 반환"""
        try:
            data = self.collection.get()
            return {
                "total_documents": len(data['ids']),
                "collection_name": settings.collection_name
            }
        except Exception as e:
            print(f"통계 정보 조회 중 오류: {e}")
            return {"total_documents": 0, "collection_name": settings.collection_name}
    
    def load_data_from_json(self, file_path: str, category: str) -> bool:
        """JSON 파일에서 데이터 로드"""
        try:
            df = pd.read_json(file_path)
        except Exception as e:
            print(f"❌ {file_path} 로딩 실패: {e}")
            return False
        
        ids, documents, metadatas = [], [], []
        
        for i, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"📂 {file_path} 처리 중"):
            try:
                metadata = row.to_dict()
                content = self._create_document_content(row, category)
                
                ids.append(f"{category}_{i}")
                documents.append(content)
                metadatas.append({
                    **metadata, 
                    "category": category,
                    **{key: value if value is not None else '' for key, value in metadata.items()}
                })
            except Exception as e:
                print(f"{i}번째 행 처리 중 오류: {e}")
                continue
        
        # 배치로 저장
        return self._save_batch(ids, documents, metadatas, file_path)
    
    def _create_document_content(self, row: pd.Series, category: str) -> str:
        """카테고리별 문서 내용 생성"""
        if category == "음식":
            return (
                f"카테고리: 음식 "
                f"이름: {row.get('이름', '')} "
                f"주소: {row.get('주소', '')} "
                f"소개: {row.get('소개', '')} "
                f"태그: {row.get('태그', '')} "
            )
        elif category == "숙소":
            return (
                f"카테고리: 숙소 "
                f"이름: {row.get('이름', '')} "
                f"주소: {row.get('주소', '')} "
                f"전화번호: {row.get('전화번호', '')} "
                f"소개: {row.get('소개', '')}"
                f"태그: {row.get('태그', '')} "
            )
        elif category == "관광지":
            return (
                f"카테고리: 관광지 "
                f"이름: {row.get('이름', '')} "
                f"주소: {row.get('주소', '')} "
                f"전화번호: {row.get('전화번호', '')} "
                f"소개: {row.get('소개', '')}"
                f"태그: {row.get('태그', '')} "
            )
        elif category == "행사":
            return (
                f"카테고리: 행사 "
                f"이름: {row.get('title', '')} "
                f"주소: {row.get('roadaddress', '')} "
                f"태그: {row.get('alltag', '')} "
                f"소개: {row.get('introduction', '')}"
            )
        else:
            return "카테고리 정보 없음"
    
    def _save_batch(self, ids: List[str], documents: List[str], metadatas: List[Dict], file_path: str) -> bool:
        """배치 단위로 데이터 저장"""
        batch_size = 100
        success = True
        
        for batch_start in range(0, len(ids), batch_size):
            batch_end = batch_start + batch_size
            try:
                self.collection.add(
                    ids=ids[batch_start:batch_end],
                    documents=documents[batch_start:batch_end],
                    metadatas=metadatas[batch_start:batch_end]
                )
                print(f"✅ {file_path} → {batch_start}~{batch_end}번 저장 완료")
            except Exception as e:
                print(f"❌ {file_path} → {batch_start}~{batch_end} 저장 실패: {e}")
                success = False
        
        return success
    
    def search(self, query: str, n_results: int = None, category_filter: str = None) -> Dict[str, Any]:
        """벡터 검색 실행"""
        if n_results is None:
            n_results = settings.max_results
        
        try:
            query_embedding = self.query_embedder.embed_query(query)
            
            # 카테고리 필터 적용
            where_clause = {"category": category_filter} if category_filter else None
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause
            )
            
            return results
        except Exception as e:
            raise RuntimeError(f"ChromaDB 검색 중 오류 발생: {e}")
    
    def search_by_category(self, query: str, categories: List[str], n_results: int = None) -> Dict[str, List[Dict]]:
        """카테고리별 검색"""
        results = {}
        
        for category in categories:
            try:
                category_results = self.search(query, n_results, category)
                results[category] = self._format_search_results(category_results)
            except Exception as e:
                print(f"카테고리 {category} 검색 중 오류: {e}")
                results[category] = []
        
        return results
    
    def _format_search_results(self, results: Dict[str, Any]) -> List[Dict]:
        """검색 결과 포맷팅"""
        formatted_results = []
        
        metadatas = results.get('metadatas', [[]])[0]
        distances = results.get('distances', [[]])[0]
        documents = results.get('documents', [[]])[0]
        
        for i, (metadata, distance, document) in enumerate(zip(metadatas, distances, documents)):
            formatted_results.append({
                "metadata": metadata,
                "distance": distance,
                "document": document,
                "rank": i + 1
            })
        
        return formatted_results

# 전역 ChromaDB 매니저 인스턴스
chromadb_manager = ChromaDBManager()
