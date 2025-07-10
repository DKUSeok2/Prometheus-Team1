"""
대화 기록 관리 모듈
"""
import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from ..config import settings

class ChatHistoryManager:
    """대화 기록 관리 클래스"""
    
    def __init__(self):
        self.history_path = settings.chat_history_path
        self.ensure_directory_exists()
    
    def ensure_directory_exists(self):
        """히스토리 디렉토리 생성"""
        os.makedirs(self.history_path, exist_ok=True)
    
    def create_new_session(self) -> str:
        """새로운 채팅 세션 생성"""
        session_id = str(uuid.uuid4())
        session_data = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "title": "새로운 대화",
            "messages": []
        }
        
        self._save_session(session_id, session_data)
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """세션에 메시지 추가"""
        try:
            session_data = self.load_session(session_id)
            if not session_data:
                return False
            
            message = {
                "role": role,  # "user" or "assistant"
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            session_data["messages"].append(message)
            session_data["updated_at"] = datetime.now().isoformat()
            
            # 첫 번째 사용자 메시지를 제목으로 설정
            if role == "user" and len(session_data["messages"]) == 1:
                session_data["title"] = content[:50] + ("..." if len(content) > 50 else "")
            
            return self._save_session(session_id, session_data)
            
        except Exception as e:
            print(f"메시지 추가 오류: {e}")
            return False
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 로드"""
        try:
            file_path = os.path.join(self.history_path, f"{session_id}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"세션 로드 오류: {e}")
        
        return None
    
    def _save_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """세션 저장"""
        try:
            file_path = os.path.join(self.history_path, f"{session_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"세션 저장 오류: {e}")
            return False
    
    def get_session_list(self) -> List[Dict[str, Any]]:
        """저장된 세션 목록 반환"""
        sessions = []
        
        try:
            for filename in os.listdir(self.history_path):
                if filename.endswith('.json'):
                    session_id = filename[:-5]  # .json 제거
                    session_data = self.load_session(session_id)
                    
                    if session_data:
                        sessions.append({
                            "session_id": session_id,
                            "title": session_data.get("title", "제목 없음"),
                            "created_at": session_data.get("created_at", ""),
                            "updated_at": session_data.get("updated_at", ""),
                            "message_count": len(session_data.get("messages", []))
                        })
            
            # 업데이트 시간순으로 정렬 (최신순)
            sessions.sort(key=lambda x: x["updated_at"], reverse=True)
            
        except Exception as e:
            print(f"세션 목록 조회 오류: {e}")
        
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """세션 삭제"""
        try:
            file_path = os.path.join(self.history_path, f"{session_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except Exception as e:
            print(f"세션 삭제 오류: {e}")
        
        return False
    
    def get_chat_history(self, session_id: str, limit: int = None) -> List[Dict[str, Any]]:
        """채팅 히스토리 반환 (최근 메시지부터)"""
        session_data = self.load_session(session_id)
        if not session_data:
            return []
        
        messages = session_data.get("messages", [])
        
        if limit:
            return messages[-limit:]
        
        return messages
    
    def search_sessions(self, query: str) -> List[Dict[str, Any]]:
        """메시지 내용으로 세션 검색"""
        matching_sessions = []
        
        try:
            for filename in os.listdir(self.history_path):
                if filename.endswith('.json'):
                    session_id = filename[:-5]
                    session_data = self.load_session(session_id)
                    
                    if session_data:
                        # 제목이나 메시지 내용에서 검색
                        title = session_data.get("title", "").lower()
                        if query.lower() in title:
                            matching_sessions.append({
                                "session_id": session_id,
                                "title": session_data.get("title", "제목 없음"),
                                "created_at": session_data.get("created_at", ""),
                                "updated_at": session_data.get("updated_at", ""),
                                "message_count": len(session_data.get("messages", [])),
                                "match_type": "title"
                            })
                            continue
                        
                        # 메시지 내용에서 검색
                        for message in session_data.get("messages", []):
                            if query.lower() in message.get("content", "").lower():
                                matching_sessions.append({
                                    "session_id": session_id,
                                    "title": session_data.get("title", "제목 없음"),
                                    "created_at": session_data.get("created_at", ""),
                                    "updated_at": session_data.get("updated_at", ""),
                                    "message_count": len(session_data.get("messages", [])),
                                    "match_type": "content"
                                })
                                break
            
            # 업데이트 시간순으로 정렬
            matching_sessions.sort(key=lambda x: x["updated_at"], reverse=True)
            
        except Exception as e:
            print(f"세션 검색 오류: {e}")
        
        return matching_sessions
    
    def export_session(self, session_id: str) -> Optional[str]:
        """세션을 텍스트 형태로 내보내기"""
        session_data = self.load_session(session_id)
        if not session_data:
            return None
        
        try:
            export_text = f"채팅 제목: {session_data.get('title', '제목 없음')}\n"
            export_text += f"생성일: {session_data.get('created_at', '')}\n"
            export_text += f"수정일: {session_data.get('updated_at', '')}\n"
            export_text += "=" * 50 + "\n\n"
            
            for message in session_data.get("messages", []):
                role = "사용자" if message["role"] == "user" else "오르다"
                timestamp = message.get("timestamp", "")
                content = message.get("content", "")
                
                export_text += f"[{timestamp}] {role}:\n{content}\n\n"
            
            return export_text
            
        except Exception as e:
            print(f"세션 내보내기 오류: {e}")
            return None
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """세션 통계 정보"""
        try:
            sessions = self.get_session_list()
            
            total_sessions = len(sessions)
            total_messages = sum(session["message_count"] for session in sessions)
            
            if sessions:
                latest_session = sessions[0]["updated_at"]
                oldest_session = sessions[-1]["created_at"]
            else:
                latest_session = oldest_session = "없음"
            
            return {
                "total_sessions": total_sessions,
                "total_messages": total_messages,
                "latest_session": latest_session,
                "oldest_session": oldest_session,
                "average_messages_per_session": total_messages / total_sessions if total_sessions > 0 else 0
            }
            
        except Exception as e:
            print(f"통계 조회 오류: {e}")
            return {
                "total_sessions": 0,
                "total_messages": 0,
                "latest_session": "없음",
                "oldest_session": "없음",
                "average_messages_per_session": 0
            }

# 전역 채팅 히스토리 매니저 인스턴스
chat_history_manager = ChatHistoryManager()
