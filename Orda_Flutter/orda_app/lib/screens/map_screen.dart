import 'package:flutter/material.dart';
import 'package:kakao_map_plugin/kakao_map_plugin.dart';
import '../services/chat_room_service.dart';
import 'chat_room_screen.dart';

class MapScreen extends StatefulWidget {
  const MapScreen({super.key});

  @override
  State<MapScreen> createState() => _MapScreenState();
}

class _MapScreenState extends State<MapScreen> {
  late KakaoMapController mapController;
  Set<Marker> markers = {};
  
  // 채팅 히스토리 관련
  List<ChatRoom> _chatRooms = [];
  bool _isLoadingHistory = false;
  bool _showSidebar = false; // 사이드바 표시 여부
  final ChatRoomService _chatRoomService = ChatRoomService();

  @override
  void initState() {
    super.initState();
  }

  // 채팅 히스토리 로드
  Future<void> _loadChatHistory() async {
    setState(() {
      _isLoadingHistory = true;
    });

    try {
      final chatRooms = await _chatRoomService.getAllChatRooms();
      setState(() {
        _chatRooms = chatRooms;
      });
    } catch (e) {
      debugPrint('채팅 히스토리 로드 실패: $e');
    } finally {
      setState(() {
        _isLoadingHistory = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text('지도'),
        backgroundColor: Colors.white,
        foregroundColor: Colors.black,
        elevation: 1,
        automaticallyImplyLeading: false,
        actions: [
          // 채팅 히스토리 토글 버튼
          IconButton(
            icon: Icon(
              Icons.chat_bubble_outline,
              color: _showSidebar ? const Color(0xFF4A90E2) : Colors.grey,
            ),
            onPressed: () {
              setState(() {
                _showSidebar = !_showSidebar;
              });
              if (_showSidebar && _chatRooms.isEmpty) {
                _loadChatHistory();
              }
            },
          ),
        ],
      ),
      body: Stack(
        children: [
          // 메인 지도 화면
          _buildMapArea(),
          
          // 채팅 히스토리 오버레이
          if (_showSidebar)
            _buildChatHistoryOverlay(),
        ],
      ),
    );
  }

  // 지도 영역 빌드
  Widget _buildMapArea() {
    return Stack(
      children: [
        // 카카오 지도
        KakaoMap(
          onMapCreated: (controller) async {
            mapController = controller;
            
            // 기본 마커 추가 (제주공항)
            markers.add(
              Marker(
                markerId: 'jeju_airport',
                latLng: LatLng(33.5097, 126.4929),
                width: 30,
                height: 44,
              ),
            );
            
            setState(() {});
          },
          markers: markers.toList(),
          center: LatLng(33.5097, 126.4929), // 제주공항 중심 좌표
          onMarkerTap: (markerId, latLng, zoomLevel) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text('마커 클릭: $markerId'),
                duration: const Duration(seconds: 1),
              ),
            );
          },
          onMapTap: (latLng) {
            // 지도 클릭 시 마커 추가
            final newMarkerId = 'marker_${markers.length}';
            markers.add(
              Marker(
                markerId: newMarkerId,
                latLng: latLng,
                width: 30,
                height: 44,
              ),
            );
            setState(() {});
          },
        ),
        
        // 현재 위치 버튼
        Positioned(
          right: 20,
          bottom: 100,
          child: FloatingActionButton(
            onPressed: () async {
              // 제주공항으로 이동
              await mapController.setCenter(LatLng(33.5097, 126.4929));
              
              if (mounted) {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    content: Text('제주공항으로 이동'),
                    duration: Duration(seconds: 1),
                  ),
                );
              }
            },
            backgroundColor: Colors.white,
            foregroundColor: const Color(0xFF6B73FF),
            child: const Icon(Icons.my_location),
          ),
        ),
        
        // 상단 검색바
        Positioned(
          top: 20,
          left: 20,
          right: _showSidebar ? 20 : 20,
          child: Container(
            height: 50,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(25),
                              boxShadow: [
                  BoxShadow(
                    color: Colors.black.withValues(alpha: 0.1),
                    blurRadius: 10,
                    offset: const Offset(0, 2),
                  ),
                ],
            ),
            child: TextField(
              decoration: InputDecoration(
                hintText: '장소를 검색해보세요',
                prefixIcon: const Icon(Icons.search),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(25),
                  borderSide: BorderSide.none,
                ),
                filled: true,
                fillColor: Colors.white,
              ),
              onSubmitted: (value) {
                // 검색 기능 구현 (추후 확장 가능)
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text('검색: $value'),
                    duration: const Duration(seconds: 1),
                  ),
                );
              },
            ),
          ),
        ),
      ],
    );
  }

  // 채팅 시간 포맷팅
  String _formatChatTime(DateTime dateTime) {
    final now = DateTime.now();
    final difference = now.difference(dateTime);

    if (difference.inDays > 0) {
      return '${difference.inDays}일 전';
    } else if (difference.inHours > 0) {
      return '${difference.inHours}시간 전';
    } else if (difference.inMinutes > 0) {
      return '${difference.inMinutes}분 전';
    } else {
      return '방금 전';
    }
  }

  // 채팅 히스토리 오버레이
  Widget _buildChatHistoryOverlay() {
    return GestureDetector(
      onTap: () {
        setState(() {
          _showSidebar = false;
        });
      },
      child: Container(
        color: Colors.black.withValues(alpha: 0.5), // 반투명 배경
        child: GestureDetector(
          onTap: () {}, // 내부 컨테이너 클릭 시 이벤트 전파 방지
          child: SafeArea(
            child: Center(
              child: Container(
                margin: const EdgeInsets.all(20),
                constraints: BoxConstraints(
                  maxHeight: MediaQuery.of(context).size.height * 0.8,
                  maxWidth: MediaQuery.of(context).size.width * 0.9,
                ),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withValues(alpha: 0.3),
                      blurRadius: 20,
                      offset: const Offset(0, 8),
                    ),
                  ],
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
              // 헤더
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: const Color(0xFF4A90E2),
                  borderRadius: const BorderRadius.only(
                    topLeft: Radius.circular(12),
                    topRight: Radius.circular(12),
                  ),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.chat, color: Colors.white, size: 24),
                    const SizedBox(width: 12),
                    const Expanded(
                      child: Text(
                        '채팅 히스토리',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                    IconButton(
                      icon: const Icon(Icons.close, color: Colors.white, size: 24),
                      onPressed: () {
                        setState(() {
                          _showSidebar = false;
                        });
                      },
                    ),
                  ],
                ),
              ),

          // 채팅 목록
          Flexible(
            child: Container(
              constraints: BoxConstraints(
                maxHeight: MediaQuery.of(context).size.height * 0.5,
                minHeight: 200,
              ),
              child: _isLoadingHistory
                ? const Center(
                    child: CircularProgressIndicator(
                      color: Color(0xFF4A90E2),
                    ),
                  )
                : _chatRooms.isEmpty
                    ? const Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              Icons.chat_bubble_outline,
                              size: 48,
                              color: Colors.grey,
                            ),
                            SizedBox(height: 16),
                            Text(
                              '채팅 히스토리가 없습니다',
                              style: TextStyle(
                                color: Colors.grey,
                                fontSize: 14,
                              ),
                            ),
                          ],
                        ),
                      )
                    : ListView.builder(
                        padding: const EdgeInsets.all(8),
                        itemCount: _chatRooms.length,
                        itemBuilder: (context, index) {
                          final chatRoom = _chatRooms[index];
                          final timeFormat = _formatChatTime(chatRoom.lastChatTime);
                          
                          return Container(
                            margin: const EdgeInsets.only(bottom: 8),
                            padding: const EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color: Colors.grey.shade50,
                              borderRadius: BorderRadius.circular(12),
                              border: Border.all(
                                color: Colors.grey.shade200,
                                width: 1,
                              ),
                            ),
                            child: InkWell(
                              onTap: () {
                                // 오버레이 닫기
                                setState(() {
                                  _showSidebar = false;
                                });
                                // 채팅방으로 이동
                                Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                    builder: (context) => ChatRoomScreen(
                                      existingSessionId: chatRoom.sessionId,
                                      chatTitle: chatRoom.title,
                                    ),
                                  ),
                                );
                              },
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  // 채팅방 제목
                                  Text(
                                    chatRoom.title,
                                    style: const TextStyle(
                                      fontSize: 14,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.black87,
                                    ),
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                  const SizedBox(height: 4),
                                  // 마지막 메시지
                                  Text(
                                    chatRoom.lastMessage,
                                    style: TextStyle(
                                      fontSize: 12,
                                      color: Colors.grey.shade600,
                                    ),
                                    maxLines: 2,
                                    overflow: TextOverflow.ellipsis,
                                  ),
                                  const SizedBox(height: 4),
                                  // 시간
                                  Text(
                                    timeFormat,
                                    style: TextStyle(
                                      fontSize: 10,
                                      color: Colors.grey.shade500,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          );
                        },
                      ),
              ),
            ),


        ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
} 