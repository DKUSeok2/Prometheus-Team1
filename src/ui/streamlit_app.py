"""
Streamlit 기반 메인 UI
"""
import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, Any, List
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import settings
from src.agents import workflow_manager
from src.chat import chat_history_manager
from src.maps import map_visualizer
from src.database import chromadb_manager

class StreamlitApp:
    """Streamlit 애플리케이션 클래스"""
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """세션 상태 초기화"""
        if "session_id" not in st.session_state:
            st.session_state.session_id = None
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_recommendations" not in st.session_state:
            st.session_state.current_recommendations = []
        if "active_tab" not in st.session_state:
            st.session_state.active_tab = "💬 채팅"  # 기본 탭
    
    def run(self):
        """메인 애플리케이션 실행"""
        st.set_page_config(
            page_title=settings.app_title,
            page_icon="🗾",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 사이드바 구성
        self.render_sidebar()
        
        # 메인 컨텐츠
        self.render_main_content()
        
        # 채팅 입력 (탭 밖에 위치)
        self.render_chat_input()
    
    def render_sidebar(self):
        """사이드바 렌더링"""
        with st.sidebar:
            st.title("🗾 오르다")
            st.caption("제주도 여행 추천 챗봇")
            
            # 새 대화 시작 버튼
            if st.button("🆕 새 대화 시작", use_container_width=True):
                self.start_new_chat()
            
            st.divider()
            
            # 채팅 히스토리
            self.render_chat_history_sidebar()
            
            st.divider()
            
            # 데이터베이스 상태
            self.render_database_status()
    
    def render_chat_history_sidebar(self):
        """채팅 히스토리 사이드바"""
        st.subheader("💬 채팅 기록")
        
        # 검색 기능
        search_query = st.text_input("채팅 검색", placeholder="키워드로 검색...")
        
        if search_query:
            sessions = chat_history_manager.search_sessions(search_query)
        else:
            sessions = chat_history_manager.get_session_list()
        
        if sessions:
            for session in sessions[:10]:  # 최대 10개만 표시
                session_title = session["title"]
                if len(session_title) > 30:
                    session_title = session_title[:30] + "..."
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    if st.button(
                        f"📝 {session_title}",
                        key=f"load_{session['session_id']}",
                        use_container_width=True
                    ):
                        self.load_chat_session(session["session_id"])
                
                with col2:
                    if st.button(
                        "🗑️",
                        key=f"delete_{session['session_id']}",
                        help="삭제",
                        use_container_width=True
                    ):
                        self.delete_chat_session(session["session_id"])
        else:
            st.info("저장된 채팅이 없습니다.")
    
    def render_database_status(self):
        """데이터베이스 상태 표시"""
        st.subheader("📊 데이터베이스 상태")
        
        try:
            stats = chromadb_manager.get_collection_stats()
            st.metric("총 문서 수", stats["total_documents"])
            st.success("✅ 연결됨")
        except Exception as e:
            st.error(f"❌ 연결 실패: {str(e)}")
    
    def render_main_content(self):
        """메인 컨텐츠 렌더링"""
        st.title("제주도 여행 추천 챗봇")
        
        # 현재 세션 정보
        if st.session_state.session_id:
            session_data = chat_history_manager.load_session(st.session_state.session_id)
            if session_data:
                st.caption(f"현재 대화: {session_data.get('title', '제목 없음')}")
        
        # 탭 구성 (radio 버튼으로 탭 효과)
        selected_tab = st.radio(
            "메뉴",
            ["💬 채팅", "🗺️ 지도"],
            index=0 if st.session_state.active_tab == "💬 채팅" else 1,
            horizontal=True,
            label_visibility="collapsed"
        )
        
        # 선택된 탭 저장
        st.session_state.active_tab = selected_tab
        
        if selected_tab == "💬 채팅":
            self.render_chat_messages()
        elif selected_tab == "🗺️ 지도":
            self.render_map_interface()
    
    def render_chat_messages(self):
        """채팅 메시지만 렌더링 (입력창 제외)"""
        # 현재 세션의 메시지 로드
        if st.session_state.session_id:
            messages = chat_history_manager.get_chat_history(st.session_state.session_id)
            st.session_state.messages = messages
        
        # 메시지 표시
        for message in st.session_state.messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "user":
                with st.chat_message("user"):
                    st.write(content)
            else:
                with st.chat_message("assistant"):
                    st.write(content)
                    
                    # 추천 정보가 있는 경우 표시
                    metadata = message.get("metadata", {})
                    if metadata.get("recommendations"):
                        self.render_recommendations(metadata["recommendations"])
    
    def render_chat_input(self):
        """채팅 입력 인터페이스 (탭 밖에 위치)"""
        if prompt := st.chat_input("제주도 여행에 대해 무엇이든 물어보세요!"):
            # 세션이 없으면 새로 생성
            if not st.session_state.session_id:
                st.session_state.session_id = chat_history_manager.create_new_session()
            
            # 채팅 탭으로 자동 전환
            st.session_state.active_tab = "💬 채팅"
            
            # 사용자 메시지 추가
            self.add_user_message(prompt)
            
            # AI 응답 처리
            with st.spinner("답변을 생성 중입니다..."):
                response = self.process_user_query(prompt)
                
                # 추천 정보 업데이트
                if response.get("search_results"):
                    recommendations = self.extract_recommendations_from_response(response)
                    if recommendations:
                        st.session_state.current_recommendations = recommendations
            
            # 페이지 새로고침
            st.rerun()
    
    def render_recommendations(self, recommendations: List[Dict[str, Any]]):
        """추천 정보 렌더링"""
        if not recommendations:
            return
        
        st.subheader("📍 추천 장소")
        
        for i, rec in enumerate(recommendations[:5]):  # 최대 5개만 표시
            with st.expander(f"{i+1}. {rec.get('name', '이름 없음')} ({rec.get('category', '기타')})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**주소:** {rec.get('address', '주소 없음')}")
                    if rec.get('phone'):
                        st.write(f"**전화:** {rec.get('phone')}")
                    if rec.get('description'):
                        st.write(f"**설명:** {rec.get('description')[:200]}...")
                    if rec.get('tags'):
                        st.write(f"**태그:** {rec.get('tags')}")
                
                with col2:
                    st.metric("유사도 점수", f"{1 - rec.get('distance', 0):.2f}")
    
    def render_map_interface(self):
        """지도 인터페이스 렌더링"""
        st.subheader("🗺️ 제주도 여행 지도")
        
        # 지도 타입 선택
        map_type = st.selectbox(
            "지도 유형 선택",
            ["기본 지도 (Folium)", "카카오 지도"],
            index=0
        )
        
        if st.session_state.current_recommendations:
            try:
                if map_type == "카카오 지도":
                    # 카카오 지도 시도
                    travel_map_html = map_visualizer.create_travel_route_map(
                        st.session_state.current_recommendations
                    )
                    components.html(travel_map_html, height=520)
                else:
                    # Folium 지도 사용 (더 안정적)
                    import folium
                    from streamlit_folium import st_folium
                    
                    # 제주도 중심 좌표
                    m = folium.Map(
                        location=[33.4996, 126.5312],
                        zoom_start=10,
                        tiles='CartoDB positron'
                    )
                    
                    colors = {"음식": "red", "숙소": "blue", "관광지": "green", "행사": "purple"}
                    
                    for rec in st.session_state.current_recommendations:
                        if rec.get('latitude') and rec.get('longitude'):
                            category = rec.get('category', '기타')
                            color = colors.get(category, 'gray')
                            
                            folium.Marker(
                                location=[rec['latitude'], rec['longitude']],
                                popup=f"<b>{rec.get('name', '이름 없음')}</b><br>{rec.get('address', '')}",
                                tooltip=rec.get('name', '이름 없음'),
                                icon=folium.Icon(color=color, icon='info-sign')
                            ).add_to(m)
                    
                    st_folium(m, width=700, height=500)
                
                # 추가 정보
                st.subheader("📊 여행 정보")
                
                # 거리/시간 매트릭스 계산 (카카오 지도일 때만)
                if map_type == "카카오 지도" and len(st.session_state.current_recommendations) >= 2:
                    with st.spinner("여행지간 거리/시간을 계산 중입니다..."):
                        matrix_result = map_visualizer.calculate_travel_time_matrix(
                            st.session_state.current_recommendations
                        )
                        
                        if "error" not in matrix_result:
                            self.render_travel_matrix(matrix_result)
                        else:
                            st.warning(matrix_result["error"])
                            
            except Exception as e:
                st.error(f"지도 표시 오류: {str(e)}")
                st.info("💡 **해결책**: 카카오 개발자 센터에서 다음 도메인들을 플랫폼에 등록하세요:")
                st.code("""
• http://localhost:8501
• https://localhost:8501  
• http://127.0.0.1:8501
• https://127.0.0.1:8501
                """)
                
                # 백업: 추천 장소 리스트만 표시
                st.subheader("📍 추천 장소 목록")
                for i, rec in enumerate(st.session_state.current_recommendations, 1):
                    with st.expander(f"{i}. {rec.get('name', '이름 없음')} ({rec.get('category', '기타')})"):
                        st.write(f"**주소:** {rec.get('address', '주소 없음')}")
                        if rec.get('phone'):
                            st.write(f"**전화:** {rec.get('phone')}")
                        st.write(f"**설명:** {rec.get('description', '설명 없음')}")
        else:
            try:
                if map_type == "카카오 지도":
                    # 기본 제주도 카카오 지도
                    default_map_html = map_visualizer.create_map()
                    components.html(default_map_html, height=520)
                else:
                    # 기본 Folium 지도
                    import folium
                    from streamlit_folium import st_folium
                    
                    m = folium.Map(
                        location=[33.4996, 126.5312],
                        zoom_start=10,
                        tiles='CartoDB positron'
                    )
                    st_folium(m, width=700, height=500)
                    
                st.info("채팅에서 장소를 추천받으면 지도에 표시됩니다.")
                
            except Exception as e:
                st.error(f"기본 지도 로드 오류: {str(e)}")
                st.info("지도 서비스에 일시적인 문제가 있습니다.")
    
    def render_travel_matrix(self, matrix_result: Dict[str, Any]):
        """여행 매트릭스 렌더링"""
        locations = matrix_result.get("locations", [])
        matrix = matrix_result.get("matrix", [])
        
        if not locations or not matrix:
            return
        
        st.subheader("🚗 장소간 이동 정보")
        
        # 매트릭스 테이블 생성
        import pandas as pd
        
        # 거리 매트릭스
        distance_data = []
        for i, origin in enumerate(locations):
            row = [origin]
            for j, destination in enumerate(locations):
                if i == j:
                    row.append("-")
                else:
                    distance = matrix[i][j].get("distance")
                    if distance:
                        row.append(f"{distance/1000:.1f}km")
                    else:
                        row.append("N/A")
            distance_data.append(row)
        
        distance_df = pd.DataFrame(
            distance_data,
            columns=["출발지"] + [f"→ {loc}" for loc in locations]
        )
        
        st.write("**거리 정보**")
        st.dataframe(distance_df, use_container_width=True)
        
        # 시간 매트릭스
        time_data = []
        for i, origin in enumerate(locations):
            row = [origin]
            for j, destination in enumerate(locations):
                if i == j:
                    row.append("-")
                else:
                    duration = matrix[i][j].get("duration")
                    if duration:
                        row.append(f"{duration//60:.0f}분")
                    else:
                        row.append("N/A")
            time_data.append(row)
        
        time_df = pd.DataFrame(
            time_data,
            columns=["출발지"] + [f"→ {loc}" for loc in locations]
        )
        
        st.write("**소요 시간**")
        st.dataframe(time_df, use_container_width=True)
    
    def start_new_chat(self):
        """새 채팅 시작"""
        st.session_state.session_id = chat_history_manager.create_new_session()
        st.session_state.messages = []
        st.session_state.current_recommendations = []
        st.session_state.active_tab = "💬 채팅"  # 채팅 탭으로 자동 전환
        st.rerun()
    
    def load_chat_session(self, session_id: str):
        """채팅 세션 로드"""
        st.session_state.session_id = session_id
        messages = chat_history_manager.get_chat_history(session_id)
        st.session_state.messages = messages
        st.session_state.active_tab = "💬 채팅"  # 채팅 탭으로 자동 전환
        
        # 마지막 추천 정보 복원
        for message in reversed(messages):
            if message.get("role") == "assistant":
                metadata = message.get("metadata", {})
                if metadata.get("recommendations"):
                    st.session_state.current_recommendations = metadata["recommendations"]
                    break
        
        st.rerun()
    
    def delete_chat_session(self, session_id: str):
        """채팅 세션 삭제"""
        if chat_history_manager.delete_session(session_id):
            if st.session_state.session_id == session_id:
                st.session_state.session_id = None
                st.session_state.messages = []
                st.session_state.current_recommendations = []
            st.rerun()
    
    def add_user_message(self, content: str):
        """사용자 메시지 추가"""
        # 세션에 저장
        chat_history_manager.add_message(
            st.session_state.session_id,
            "user",
            content
        )
        
        # 현재 세션 상태 업데이트
        st.session_state.messages.append({
            "role": "user",
            "content": content
        })
    
    def process_user_query(self, query: str) -> Dict[str, Any]:
        """사용자 쿼리 처리"""
        # 현재 채팅 히스토리 가져오기
        chat_history = chat_history_manager.get_chat_history(
            st.session_state.session_id,
            limit=10  # 최근 10개 메시지만
        )
        
        # 워크플로우 매니저로 처리
        result = workflow_manager.process_query(query, chat_history)
        
        # 응답을 세션에 저장
        recommendations = self.extract_recommendations_from_response(result)
        chat_history_manager.add_message(
            st.session_state.session_id,
            "assistant",
            result["response"],
            metadata={
                "analysis": result.get("analysis", {}),
                "search_results": result.get("search_results", {}),
                "recommendations": recommendations
            }
        )
        
        # 현재 세션 상태 업데이트
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["response"],
            "metadata": {
                "recommendations": recommendations
            }
        })
        
        return result
    
    def extract_recommendations_from_response(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """응답에서 추천 정보 추출"""
        recommendations = []
        search_results = result.get("search_results", {})
        
        for category, results in search_results.items():
            for item in results:
                metadata = item.get("metadata", {})
                recommendations.append({
                    "category": category,
                    "name": metadata.get("이름") or metadata.get("title", ""),
                    "address": metadata.get("주소") or metadata.get("roadaddress", ""),
                    "phone": metadata.get("전화번호", ""),
                    "description": metadata.get("소개") or metadata.get("introduction", ""),
                    "tags": metadata.get("태그") or metadata.get("alltag", ""),
                    "distance": item.get("distance", 0)
                })
        
        return recommendations

# 전역 앱 인스턴스
streamlit_app = StreamlitApp()
