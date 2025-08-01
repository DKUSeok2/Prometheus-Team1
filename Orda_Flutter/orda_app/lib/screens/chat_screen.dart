import 'package:flutter/material.dart';
import 'chat_room_screen.dart';
import 'festival_group_selection_screen.dart';
import '../services/chat_room_service.dart';
import '../services/chat_service.dart';

class ChatScreen extends StatefulWidget {
  final Function(int)? onTabChange;
  
  const ChatScreen({super.key, this.onTabChange});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> with WidgetsBindingObserver {
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();
  final ChatRoomService _chatRoomService = ChatRoomService();
  final ChatService _chatService = ChatService();
  
  List<ChatRoom> _chatRooms = [];
  bool _isLoadingHistory = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _loadChatHistory();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    super.didChangeAppLifecycleState(state);
    if (state == AppLifecycleState.resumed) {
      // 앱이 다시 활성화될 때 히스토리 새로고침
      _loadChatHistory();
    }
  }

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
      print('채팅 히스토리 로드 실패: $e');
    } finally {
      setState(() {
        _isLoadingHistory = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      key: _scaffoldKey,
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.black),
          onPressed: () {
            // 홈 탭으로 이동
            if (widget.onTabChange != null) {
              widget.onTabChange!(0); // 홈 탭 인덱스
            }
          },
        ),
        title: const Text(
          '채팅하기',
          style: TextStyle(
            color: Colors.black,
            fontSize: 18,
            fontWeight: FontWeight.w600,
          ),
        ),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.menu, color: Colors.black),
            onPressed: () {
              // 메뉴 열기 전에 히스토리 새로고침
              _loadChatHistory();
              _scaffoldKey.currentState?.openEndDrawer();
            },
          ),
        ],
      ),
      endDrawer: _buildChatHistoryDrawer(),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 20),
            
            // 안내 메시지 (검은색 bold로 변경)
            const Text(
              '여행 준비를 시작해보세요!',
              style: TextStyle(
                fontSize: 16,
                color: Colors.black,
                fontWeight: FontWeight.bold,
              ),
            ),
            
            const SizedBox(height: 30),
            
            // 오르미와 채팅 시작하기 버튼
            GestureDetector(
              onTap: () {
                _startChatWithOrmi();
              },
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(16),
                margin: const EdgeInsets.only(bottom: 12),
                decoration: BoxDecoration(
                  color: const Color(0xFFF0F8FF),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  children: [
                    // 텍스트
                    const Expanded(
                      child: Text(
                        '오르미와 채팅 시작하기',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w500,
                          color: Colors.black,
                        ),
                      ),
                    ),
                    
                    // 오르미 캐릭터
                    _buildOrmiCharacter(),
                  ],
                ),
              ),
            ),
            
            // 프로메테우스 멤버라면? 버튼
            GestureDetector(
              onTap: () {
                _startPrometheusChat();
              },
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: const Color(0xFFF0F8FF),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  children: [
                    // 텍스트
                    const Expanded(
                      child: Text(
                        '혹시 프로메테우스 멤버라면?',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w500,
                          color: Colors.black,
                        ),
                      ),
                    ),
                    
                    // 오르미 캐릭터 (오렌지 액센트)
                    _buildOrmiCharacter(isOrange: true),
                  ],
                ),
              ),
            ),
            
            const Spacer(),
          ],
        ),
      ),
    );
  }

  Widget _buildChatHistoryDrawer() {
    return Container(
      width: MediaQuery.of(context).size.width * 0.85,
      child: Drawer(
        child: Column(
          children: [
            // 드로어 헤더
            Container(
              height: 100,
              width: double.infinity,
              padding: const EdgeInsets.only(top: 40, left: 20, right: 20),
              decoration: const BoxDecoration(
                color: Colors.white,
                border: Border(
                  bottom: BorderSide(color: Colors.grey, width: 0.5),
                ),
              ),
              child: Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.arrow_back, color: Colors.black),
                    onPressed: () => Navigator.pop(context),
                  ),
                  const Expanded(
                    child: Text(
                      '메뉴',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w600,
                        color: Colors.black,
                      ),
                    ),
                  ),
                  const SizedBox(width: 48), // IconButton과 같은 크기로 균형 맞추기
                ],
              ),
            ),

            // 채팅 히스토리 섹션
            Container(
              padding: const EdgeInsets.all(20),
              alignment: Alignment.centerLeft,
              child: const Text(
                '채팅 히스토리',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: Colors.black,
                ),
              ),
            ),

            // 채팅 목록
            Expanded(
              child: _isLoadingHistory
                  ? const Center(child: CircularProgressIndicator())
                  : _chatRooms.isEmpty
                      ? const Center(
                          child: Text(
                            '아직 채팅 기록이 없습니다.\n오르미와 첫 대화를 시작해보세요!',
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              color: Colors.grey,
                              fontSize: 14,
                            ),
                          ),
                        )
                      : ListView.builder(
                          padding: const EdgeInsets.symmetric(horizontal: 16),
                          itemCount: _chatRooms.length,
                          itemBuilder: (context, index) {
                            final chatRoom = _chatRooms[index];
                            final isLatest = index == 0; // 첫 번째(최신) 채팅방
                            
                            return _buildChatHistoryItem(
                              chatRoom: chatRoom,
                              title: chatRoom.title,
                              lastChat: '마지막 채팅: ${_chatRoomService.formatTime(chatRoom.lastChatTime)}',
                              isSelected: isLatest,
                              onTap: () {
                                // 해당 채팅방으로 이동
                                _openChatRoom(chatRoom);
                              },
                              onLongPress: () {
                                // 편집/삭제 옵션 표시
                                _showChatRoomOptions(chatRoom);
                              },
                            );
                          },
                        ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildChatHistoryItem({
    required ChatRoom chatRoom,
    required String title,
    required String lastChat,
    required bool isSelected,
    required VoidCallback onTap,
    required VoidCallback onLongPress,
  }) {
    return GestureDetector(
      onTap: onTap,
      onLongPress: onLongPress,
      child: Container(
        margin: const EdgeInsets.only(bottom: 12),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: isSelected ? const Color(0xFFE3F2FD) : const Color(0xFFF5F5F5),
          borderRadius: BorderRadius.circular(12),
          border: isSelected 
            ? Border.all(color: const Color(0xFF2196F3), width: 1)
            : null,
        ),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                      color: isSelected ? const Color(0xFF1976D2) : Colors.black,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    lastChat,
                    style: TextStyle(
                      fontSize: 12,
                      color: isSelected ? const Color(0xFF1976D2) : Colors.grey,
                    ),
                  ),
                ],
              ),
            ),
            // 메뉴 버튼
            IconButton(
              icon: Icon(
                Icons.more_vert,
                color: isSelected ? const Color(0xFF1976D2) : Colors.grey,
                size: 20,
              ),
              onPressed: () => _showChatRoomOptions(chatRoom),
              constraints: const BoxConstraints(),
              padding: const EdgeInsets.all(4),
            ),
          ],
        ),
      ),
    );
  }
  
  Widget _buildOrmiCharacter({bool isOrange = false}) {
    return Container(
      width: 60,
      height: 60,
      child: Stack(
        children: [
          // 오르미 캐릭터 이미지 (원본 모양 그대로)
          Image.asset(
            'assets/images/ormi_travel.png',
            width: 60,
            height: 60,
            fit: BoxFit.contain,
            errorBuilder: (context, error, stackTrace) {
              // 이미지 로드 실패 시 기본 컨테이너
              return Container(
                width: 50,
                height: 50,
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    begin: Alignment.topCenter,
                    end: Alignment.bottomCenter,
                    colors: [
                      const Color(0xFF87CEEB), // 하늘색
                      const Color(0xFF98FB98), // 연한 녹색
                    ],
                  ),
                  borderRadius: BorderRadius.circular(25),
                ),
              );
            },
          ),
          
          // 오렌지 액센트 (축제 버튼일 때)
          if (isOrange)
            Positioned(
              top: 5,
              right: 5,
              child: Container(
                width: 15,
                height: 15,
                decoration: BoxDecoration(
                  color: const Color(0xFFFF6B35),
                  borderRadius: BorderRadius.circular(7.5),
                ),
                child: const Icon(
                  Icons.celebration,
                  color: Colors.white,
                  size: 10,
                ),
              ),
            ),
        ],
      ),
    );
  }
  
  void _startChatWithOrmi() async {
    // 새로운 채팅룸 화면으로 이동
    final result = await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => const ChatRoomScreen(),
      ),
    );
    
    // 채팅방에서 돌아왔을 때 히스토리 새로고침
    if (result == null) {
      _loadChatHistory();
    }
  }
  
  void _startPrometheusChat() {
    // 프로메테우스 그룹 선택 화면으로 이동
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => const FestivalGroupSelectionScreen(),
      ),
    );
  }

  void _openChatRoom(ChatRoom chatRoom) async {
    // 드로어 닫기
    Navigator.pop(context);
    
    final result = await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => ChatRoomScreen(
          existingSessionId: chatRoom.sessionId,
          chatTitle: chatRoom.title,
        ),
      ),
    );

    // 채팅방에서 돌아왔을 때 히스토리 새로고침
    if (result == null) {
      _loadChatHistory();
    }
  }

  // 채팅방 옵션 모달 표시
  void _showChatRoomOptions(ChatRoom chatRoom) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        decoration: const BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.only(
            topLeft: Radius.circular(20),
            topRight: Radius.circular(20),
          ),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // 핸들
            Container(
              width: 40,
              height: 4,
              margin: const EdgeInsets.only(top: 12),
              decoration: BoxDecoration(
                color: Colors.grey[300],
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            
            // 제목
            Padding(
              padding: const EdgeInsets.all(20),
              child: Text(
                chatRoom.title,
                style: const TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
                textAlign: TextAlign.center,
              ),
            ),
            
            // 옵션 버튼들
            ListTile(
              leading: const Icon(Icons.edit, color: Colors.blue),
              title: const Text('채팅방 이름 변경'),
              onTap: () {
                Navigator.pop(context);
                _showEditTitleDialog(chatRoom);
              },
            ),
            
            ListTile(
              leading: const Icon(Icons.delete, color: Colors.red),
              title: const Text('채팅방 삭제'),
              onTap: () {
                Navigator.pop(context);
                _showDeleteConfirmDialog(chatRoom);
              },
            ),
            
            // 취소 버튼
            ListTile(
              leading: const Icon(Icons.cancel, color: Colors.grey),
              title: const Text('취소'),
              onTap: () => Navigator.pop(context),
            ),
            
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }

  // 채팅방 제목 변경 다이얼로그
  void _showEditTitleDialog(ChatRoom chatRoom) {
    final TextEditingController titleController = TextEditingController(text: chatRoom.title);
    
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('채팅방 이름 변경'),
        content: TextField(
          controller: titleController,
          decoration: const InputDecoration(
            labelText: '새로운 이름',
            border: OutlineInputBorder(),
          ),
          maxLength: 50,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('취소'),
          ),
          ElevatedButton(
            onPressed: () async {
              final newTitle = titleController.text.trim();
              if (newTitle.isNotEmpty && newTitle != chatRoom.title) {
                Navigator.pop(context);
                await _updateChatRoomTitle(chatRoom, newTitle);
              } else {
                Navigator.pop(context);
              }
            },
            child: const Text('변경'),
          ),
        ],
      ),
    );
  }

  // 채팅방 삭제 확인 다이얼로그
  void _showDeleteConfirmDialog(ChatRoom chatRoom) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('채팅방 삭제'),
        content: Text('\'${chatRoom.title}\' 채팅방을 삭제하시겠습니까?\n\n이 작업은 되돌릴 수 없습니다.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('취소'),
          ),
          ElevatedButton(
            onPressed: () async {
              Navigator.pop(context);
              await _deleteChatRoom(chatRoom);
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.red,
              foregroundColor: Colors.white,
            ),
            child: const Text('삭제'),
          ),
        ],
      ),
    );
  }

  // 채팅방 제목 업데이트 실행
  Future<void> _updateChatRoomTitle(ChatRoom chatRoom, String newTitle) async {
    try {
      // 로딩 표시
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('제목을 변경하는 중...')),
      );

      // 로컬 전용 세션인지 확인
      if (chatRoom.sessionId.startsWith('local_')) {
        // 로컬 전용 세션은 로컬 데이터만 업데이트
        await _chatRoomService.updateChatRoomTitle(chatRoom.sessionId, newTitle);
      } else {
        // 백엔드 API 호출
        final result = await _chatService.updateChatRoomTitle(chatRoom.sessionId, newTitle);
        
        if (result['success']) {
          // 로컬 데이터도 업데이트
          await _chatRoomService.updateChatRoomTitle(chatRoom.sessionId, newTitle);
        } else {
          throw Exception(result['error']);
        }
      }

      // 히스토리 새로고침
      await _loadChatHistory();
      
      // 성공 메시지
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('채팅방 이름이 변경되었습니다.'),
          backgroundColor: Colors.green,
        ),
      );
    } catch (e) {
      // 에러 메시지
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('제목 변경에 실패했습니다: $e'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  // 채팅방 삭제 실행
  Future<void> _deleteChatRoom(ChatRoom chatRoom) async {
    try {
      // 로딩 표시
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('채팅방을 삭제하는 중...')),
      );

      // 로컬 전용 세션인지 확인
      if (chatRoom.sessionId.startsWith('local_')) {
        // 로컬 전용 세션은 로컬 데이터만 삭제
        await _chatRoomService.deleteChatRoom(chatRoom.sessionId);
      } else {
        // 백엔드 API 호출
        final result = await _chatService.deleteChatRoom(chatRoom.sessionId);
        
        if (result['success']) {
          // 로컬 데이터도 삭제
          await _chatRoomService.deleteChatRoom(chatRoom.sessionId);
        } else {
          throw Exception(result['error']);
        }
      }

      // 히스토리 새로고침
      await _loadChatHistory();
      
      // 성공 메시지
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('채팅방이 삭제되었습니다.'),
          backgroundColor: Colors.green,
        ),
      );
    } catch (e) {
      // 에러 메시지
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('채팅방 삭제에 실패했습니다: $e'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }
} 