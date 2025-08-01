import 'package:flutter/material.dart';
import '../services/chat_service.dart';
import '../services/chat_room_service.dart';
import '../services/travel_plan_service.dart';
import '../models/travel_plan.dart';
import 'travel_map_screen.dart';

class ChatRoomScreen extends StatefulWidget {
  final String? existingSessionId; // 기존 세션 ID (선택적)
  final String? chatTitle; // 채팅방 제목 (선택적)
  final String? userName; // 사용자 이름 (개인화용)
  final String? personality; // 성향 (개인화용)
  final String? travelStyle; // 여행 스타일 (개인화용)
  final String? presetTravelPlan; // 미리 정해진 여행 일정 (데모용)
  
  const ChatRoomScreen({
    super.key, 
    this.existingSessionId,
    this.chatTitle,
    this.userName,
    this.personality,
    this.travelStyle,
    this.presetTravelPlan,
  });

  @override
  State<ChatRoomScreen> createState() => _ChatRoomScreenState();
}

class _ChatRoomScreenState extends State<ChatRoomScreen> with WidgetsBindingObserver {
  final TextEditingController _messageController = TextEditingController();
  final ChatService _chatService = ChatService();
  final ChatRoomService _chatRoomService = ChatRoomService();
  final ScrollController _scrollController = ScrollController();
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();
  
  List<Map<String, dynamic>> _messages = [];
  String? _currentSessionId;
  String _chatTitle = '채팅하기';
  bool _isLoading = false;
  bool _isSendingMessage = false;
  bool _isFirstMessage = true; // 첫 메시지 여부 추적
  
  // 사이드바 관련
  List<ChatRoom> _chatRooms = [];
  bool _isLoadingHistory = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _currentSessionId = widget.existingSessionId;
    _chatTitle = widget.chatTitle ?? '채팅하기'; // 기본 제목을 "채팅하기"로 변경
    _initializeChat();
    _loadChatHistory();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _messageController.dispose();
    _scrollController.dispose();
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

  Future<void> _initializeChat() async {
    setState(() {
      _isLoading = true;
    });

    try {
      if (_currentSessionId != null) {
        // 기존 세션 로드
        await _loadExistingChatHistory();
        _isFirstMessage = false; // 기존 채팅방이므로 첫 메시지가 아님
      } else if (widget.presetTravelPlan != null && widget.presetTravelPlan!.isNotEmpty) {
        // 미리 정해진 여행일정이 있으면 챗봇 API 호출 없이 바로 표시
        _addWelcomeMessage();
        _isFirstMessage = false; // 일정이 표시되었으므로 첫 메시지가 아님
      } else {
        // 새로운 세션 시작 (일반 챗봇)
        await _chatService.startNewSession();
        _addWelcomeMessage();
      }
    } catch (e) {
      _showError('채팅 초기화 중 오류가 발생했습니다: $e');
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<void> _loadExistingChatHistory() async {
    if (_currentSessionId == null) return;

    // 로컬 전용 세션인 경우 백엔드 API 호출 건너뛰기
    if (_currentSessionId!.startsWith('local_')) {
      print('로컬 전용 세션이므로 샘플 메시지를 로드합니다: $_currentSessionId');
      _loadLocalSampleMessages();
      return;
    }

    try {
      final result = await _chatService.getChatHistory(_currentSessionId!);
      
      if (result['success']) {
        setState(() {
          _messages = result['messages'] ?? [];
        });
        
        // 메시지를 로드한 후 맨 아래로 스크롤
        WidgetsBinding.instance.addPostFrameCallback((_) {
          _scrollToBottom();
        });
      } else {
        _showError('채팅 히스토리 로드 실패: ${result['error']}');
        // 실패 시 새로운 채팅으로 시작
        _addWelcomeMessage();
      }
    } catch (e) {
      _showError('채팅 히스토리 로드 중 오류: $e');
      // 실패 시 새로운 채팅으로 시작
      _addWelcomeMessage();
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

  void _addWelcomeMessage() {
    // 디버그: presetTravelPlan 확인
    print('🔍 ChatRoomScreen 디버그: presetTravelPlan 상태');
    print('📝 presetTravelPlan null 여부: ${widget.presetTravelPlan == null}');
    print('📝 presetTravelPlan 길이: ${widget.presetTravelPlan?.length ?? 0}');
    print('📋 presetTravelPlan 내용 (처음 100자): ${widget.presetTravelPlan?.substring(0, widget.presetTravelPlan!.length > 100 ? 100 : widget.presetTravelPlan!.length) ?? 'null'}');
    
    setState(() {
      // 강제로 조건 확인
      bool hasPresetPlan = widget.presetTravelPlan != null;
      bool isNotEmpty = widget.presetTravelPlan?.isNotEmpty ?? false;
      
      print('🔍 조건 확인:');
      print('📝 hasPresetPlan: $hasPresetPlan');
      print('📝 isNotEmpty: $isNotEmpty');
      print('📝 최종 조건: ${hasPresetPlan && isNotEmpty}');
      
      if (hasPresetPlan && isNotEmpty) {
        print('✅ 미리 정해진 여행일정 사용');
        // 미리 정해진 여행일정이 있으면 바로 표시
        _messages = [
          {
            'text': widget.presetTravelPlan!,
            'isUser': false,
            'time': ChatService.getCurrentTime(),
            'timestamp': DateTime.now().toIso8601String(),
            'isWelcome': true,
          }
        ];
      } else {
        print('❌ 기본 환영 메시지 사용');
        print('❌ hasPresetPlan: $hasPresetPlan, isNotEmpty: $isNotEmpty');
        // 기본 환영 메시지
        _messages = [
          {
            'text': '환영해요!\n당신의 여행을 도와줄 오르미에요.',
            'isUser': false,
            'time': ChatService.getCurrentTime(),
            'timestamp': DateTime.now().toIso8601String(),
            'isWelcome': true,
          }
        ];
      }
    });
  }

  Future<void> _sendMessage() async {
    final messageText = _messageController.text.trim();
    if (messageText.isEmpty || _isSendingMessage) return;

    setState(() {
      _isSendingMessage = true;
      _messages.add({
        'text': messageText,
        'isUser': true,
        'time': ChatService.getCurrentTime(),
        'timestamp': DateTime.now().toIso8601String(),
      });
    });

    _messageController.clear();
    _scrollToBottom();

    try {
      if (widget.presetTravelPlan != null && widget.presetTravelPlan!.isNotEmpty) {
        // 미리 정해진 여행일정이 있는 경우 간단한 응답
        String response;
        if (widget.personality?.contains('테토') == true) {
          response = "일정은 이미 정해져 있음. 다른 질문 있으면 말해.";
        } else {
          response = "앞서 보여드린 여행 일정이 도움이 되셨나요? 혹시 다른 궁금한 점이 있으시면 언제든 말씀해 주세요! 😊";
        }
        
        setState(() {
          _messages.add({
            'text': response,
            'isUser': false,
            'time': ChatService.getCurrentTime(),
            'timestamp': DateTime.now().toIso8601String(),
          });
        });
      } else {
        // 일반 챗봇 API 호출
        final result = await _chatService.sendMessage(
          messageText,
          sessionId: _currentSessionId,
          userName: widget.userName,
          personality: widget.personality,
          travelStyle: widget.travelStyle,
        );

        if (result['success']) {
          final response = result['response'];
          final sessionId = result['session_id'];
          
          setState(() {
            _currentSessionId = sessionId;
            
            _messages.add({
              'text': response,
              'isUser': false,
              'time': ChatService.getCurrentTime(),
              'timestamp': DateTime.now().toIso8601String(),
            });
          });

          // 첫 메시지인 경우 채팅방 생성
          if (_isFirstMessage && sessionId != null) {
            await _chatRoomService.createChatRoom(sessionId, messageText);
            _isFirstMessage = false;
          } else if (sessionId != null) {
            // 기존 채팅방 마지막 메시지 업데이트
            await _chatRoomService.updateLastMessage(sessionId, response);
          }

          _scrollToBottom();
        } else {
          _showError('메시지 전송 실패: ${result['error']}');
        }
      }
      
      _scrollToBottom(); // presetTravelPlan 경우에도 스크롤
    } catch (e) {
      _showError('메시지 전송 중 오류: $e');
    } finally {
      setState(() {
        _isSendingMessage = false;
      });
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _showError(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red,
        duration: const Duration(seconds: 3),
      ),
    );
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
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text(
          '채팅하기', // 고정으로 "채팅하기" 표시
          style: TextStyle(
            color: Colors.black,
            fontSize: 18,
            fontWeight: FontWeight.w600,
          ),
        ),
        centerTitle: true,
        actions: [
          // 테스트용 샘플 여행 일정 버튼
          IconButton(
            icon: const Icon(Icons.flight_takeoff, color: Colors.blue),
            onPressed: _addSampleTravelPlan,
            tooltip: '샘플 여행 일정',
          ),
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
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                // 채팅 메시지 영역
                Expanded(
                  child: _messages.isEmpty
                      ? _buildWelcomeScreen()
                      : ListView.builder(
                          controller: _scrollController,
                          padding: const EdgeInsets.all(16),
                          itemCount: _messages.length,
                          itemBuilder: (context, index) {
                            final message = _messages[index];
                            if (message['isWelcome'] == true) {
                              return _buildWelcomeMessage();
                            }
                            return _buildMessageBubble(message);
                          },
                        ),
                ),

                // 메시지 입력 영역
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    boxShadow: [
                      BoxShadow(
                        offset: const Offset(0, -2),
                        blurRadius: 5,
                        color: Colors.black.withOpacity(0.1),
                      ),
                    ],
                  ),
                  child: Row(
                    children: [
                      Expanded(
                        child: TextField(
                          controller: _messageController,
                          decoration: InputDecoration(
                            hintText: '메시지를 입력하세요...',
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(25),
                              borderSide: const BorderSide(color: Colors.blue, width: 2),
                            ),
                            focusedBorder: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(25),
                              borderSide: const BorderSide(color: Colors.blue, width: 2),
                            ),
                            contentPadding: const EdgeInsets.symmetric(
                              horizontal: 20,
                              vertical: 10,
                            ),
                          ),
                          maxLines: null,
                          onSubmitted: (_) => _sendMessage(),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Container(
                        decoration: const BoxDecoration(
                          color: Colors.blue,
                          shape: BoxShape.circle,
                        ),
                        child: IconButton(
                          onPressed: _isSendingMessage ? null : _sendMessage,
                          icon: _isSendingMessage
                              ? const SizedBox(
                                  width: 20,
                                  height: 20,
                                  child: CircularProgressIndicator(
                                    color: Colors.white,
                                    strokeWidth: 2,
                                  ),
                                )
                              : const Icon(Icons.send, color: Colors.white),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
    );
  }

  Widget _buildWelcomeScreen() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          // 큰 오르미 캐릭터 (이미지 사이즈 축소)
          Container(
            width: 150, // 200 -> 150으로 축소
            height: 150, // 200 -> 150으로 축소
            child: Image.asset(
              'assets/images/ormi.png',
              fit: BoxFit.contain,
            ),
          ),
          const SizedBox(height: 30), // 40 -> 30으로 축소
          
          // 환영 메시지
          const Text(
            '환영해요!',
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: Colors.black,
            ),
          ),
          const SizedBox(height: 16),
          const Text(
            '당신의 여행을 도와줄 오르미에요.',
            style: TextStyle(
              fontSize: 16,
              color: Colors.grey,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildWelcomeMessage() {
    return Container(
      margin: const EdgeInsets.only(bottom: 20),
      child: Column(
        children: [
          // 큰 오르미 캐릭터 (이미지 사이즈 축소)
          Container(
            width: 150, // 200 -> 150으로 축소
            height: 150, // 200 -> 150으로 축소
            child: Image.asset(
              'assets/images/ormi.png',
              fit: BoxFit.contain,
            ),
          ),
          const SizedBox(height: 15), // 20 -> 15로 축소
          
          // 환영 메시지 버블
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.grey[200],
              borderRadius: BorderRadius.circular(18),
            ),
            child: const Text(
              '환영해요!\n당신의 여행을 도와줄 오르미에요.',
              style: TextStyle(
                color: Colors.black,
                fontSize: 16,
              ),
              textAlign: TextAlign.center,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLargeOrmiCharacter() {
    return Container(
      width: 200,
      height: 200,
      child: Image.asset(
        'assets/images/ormi.png',
        fit: BoxFit.contain,
      ),
    );
  }

  Widget _buildMessageBubble(Map<String, dynamic> message) {
    final isUser = message['isUser'] as bool;
    final messageText = message['text'] as String;
    final hasTravelPlan = _containsTravelPlan(messageText);
    
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      child: Row(
        mainAxisAlignment: isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (!isUser) ...[
            Container(
              width: 40,
              height: 40,
              margin: const EdgeInsets.only(right: 8),
              child: Image.asset(
                'assets/images/ormi.png',
                fit: BoxFit.contain,
              ),
            ),
          ],
          Flexible(
            child: Column(
              crossAxisAlignment: isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
              children: [
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: isUser ? Colors.blue : Colors.grey[200],
                    borderRadius: BorderRadius.circular(18),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        messageText,
                        style: TextStyle(
                          color: isUser ? Colors.white : Colors.black,
                          fontSize: 14,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        message['time'] as String,
                        style: TextStyle(
                          color: isUser ? Colors.white70 : Colors.grey[600],
                          fontSize: 10,
                        ),
                      ),
                    ],
                  ),
                ),
                // 지도로 보기 버튼 (챗봇 메시지이고 여행 일정이 포함된 경우)
                if (!isUser && hasTravelPlan) ...[
                  const SizedBox(height: 8),
                  GestureDetector(
                    onTap: () => _showTravelMap(messageText),
                    child: Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: const Color(0xFF6B73FF),
                        borderRadius: BorderRadius.circular(15),
                        border: Border.all(color: const Color(0xFF6B73FF), width: 1),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: const [
                          Icon(
                            Icons.map,
                            color: Colors.white,
                            size: 16,
                          ),
                          SizedBox(width: 4),
                          Text(
                            '지도로 보기',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 12,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ],
              ],
            ),
          ),
          if (isUser) ...[
            Container(
              width: 32,
              height: 32,
              margin: const EdgeInsets.only(left: 8),
              decoration: BoxDecoration(
                color: Colors.grey[300],
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.person, color: Colors.grey, size: 18),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildSmallOrmiCharacter() {
    return Container(
      width: 40,
      height: 40,
      child: Image.asset(
        'assets/images/ormi.png',
        fit: BoxFit.contain,
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
                            final isCurrentRoom = chatRoom.sessionId == _currentSessionId;
                            
                            return _buildChatHistoryItem(
                              chatRoom: chatRoom,
                              title: chatRoom.title,
                              lastChat: '마지막 채팅: ${_chatRoomService.formatTime(chatRoom.lastChatTime)}',
                              isSelected: isCurrentRoom,
                              onTap: () {
                                // 현재 채팅방이 아닐 때만 이동
                                if (!isCurrentRoom) {
                                  _openChatRoom(chatRoom);
                                } else {
                                  Navigator.pop(context); // 현재 채팅방이면 사이드바만 닫기
                                }
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

  void _openChatRoom(ChatRoom chatRoom) {
    Navigator.pop(context); // 사이드바 닫기
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (context) => ChatRoomScreen(
          existingSessionId: chatRoom.sessionId,
          chatTitle: chatRoom.title,
        ),
      ),
    );
  }

  // 메시지가 여행 일정을 포함하는지 확인
  bool _containsTravelPlan(String content) {
    return content.contains(RegExp(r'\d+일차')) && 
           (content.contains('일정') || 
            content.contains('여행') || 
            content.contains('추천') ||
            content.contains('📅'));
  }

  // 여행 일정 지도 표시
  void _showTravelMap(String content) async {
    // 로딩 표시
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => const AlertDialog(
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('장소 검색 중...'),
          ],
        ),
      ),
    );

    try {
      // 샘플 데이터 생성
      List<TravelPlan> travelPlans = _createDirectSampleData();
      
      // 실제 카카오 API로 정확한 좌표 검색
      print('🔍 카카오 API로 장소 좌표 검색 시작...');
      await TravelPlanService.searchPlaceCoordinates(travelPlans);
      print('✅ 장소 좌표 검색 완료!');
      
      // 로딩 다이얼로그 닫기
      Navigator.pop(context);
      
      // 지도 화면으로 이동
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => TravelMapScreen(travelPlans: travelPlans),
        ),
      );
    } catch (e) {
      // 로딩 다이얼로그 닫기
      Navigator.pop(context);
      
      // 오류 메시지 표시
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('지도를 불러오는 중 오류가 발생했습니다: $e'),
          backgroundColor: Colors.red,
        ),
      );
      print('❌ 지도 표시 오류: $e');
    }
  }

  // 원본 샘플 데이터 제대로 파싱 (4일차까지)
  List<TravelPlan> _createDirectSampleData() {
    // 원본 샘플 텍스트 사용
    String samplePlan = TravelPlanService.getSampleTravelPlan();
    
    // 개선된 파싱 로직으로 시도
    List<TravelPlan> parsed = _parseWithImprovedLogic(samplePlan);
    
    // 파싱 실패 시 직접 생성된 4일차 데이터 사용
    if (parsed.length < 4) {
      print('🔧 파싱 실패, 직접 생성된 4일차 데이터 사용');
      return _getDirectFourDayData();
    }
    
    return parsed;
  }

  // 개선된 파싱 로직
  List<TravelPlan> _parseWithImprovedLogic(String text) {
    List<TravelPlan> plans = [];
    
    // 일차별 섹션 분할 (더 간단한 패턴)
    List<String> sections = text.split(RegExp(r'📅\s*\*\*'));
    
    for (String section in sections) {
      if (!section.contains('일차')) continue;
      
      // 일차 번호 추출
      RegExp dayRegex = RegExp(r'(\d+일차)');
      RegExpMatch? dayMatch = dayRegex.firstMatch(section);
      if (dayMatch == null) continue;
      
      String day = dayMatch.group(1)!;
      List<Place> places = _extractPlacesFromSection(section, day);
      
      if (places.isNotEmpty) {
        plans.add(TravelPlan(day: day, places: places));
      }
    }
    
    return plans;
  }

  // 섹션에서 장소 추출
  List<Place> _extractPlacesFromSection(String section, String day) {
    List<Place> places = [];
    int sequence = 1;
    
    // 시간과 장소가 포함된 라인 찾기
    List<String> lines = section.split('\n');
    
    for (String line in lines) {
      if (!line.contains('(') || !line.contains(':')) continue;
      
      // 시간 추출
      RegExp timeRegex = RegExp(r'\((\d{2}:\d{2})\)');
      RegExpMatch? timeMatch = timeRegex.firstMatch(line);
      if (timeMatch == null) continue;
      
      String time = timeMatch.group(1)!;
      
      // 장소와 설명 추출
      String cleanLine = line.replaceAll(RegExp(r'[📅•]'), '').trim();
      cleanLine = cleanLine.replaceAll(RegExp(r'\([^)]*\)'), '').trim();
      cleanLine = cleanLine.replaceAll(':', '').trim();
      
      if (cleanLine.isEmpty) continue;
      
      String placeName = _extractPlaceNameFromLine(cleanLine);
      String category = _categorizeFromLine(cleanLine);
      
      if (placeName.isNotEmpty) {
        places.add(Place(
          name: placeName,
          category: category,
          time: time,
          description: cleanLine,
          sequence: sequence++,
          latitude: _getLatitude(placeName, day),
          longitude: _getLongitude(placeName, day),
        ));
      }
    }
    
    return places;
  }

  // 실제 제주도 유명 장소들로 구성된 4일차 데이터
  List<TravelPlan> _getDirectFourDayData() {
    return [
      TravelPlan(day: '1일차', places: [
        Place(name: '제주국제공항', category: '교통', time: '09:00', description: '제주도 도착', sequence: 1),
        Place(name: '명진전복', category: '음식', time: '12:00', description: '제주 대표 전복 맛집에서 점심', sequence: 2),
        Place(name: '용두암', category: '관광지', time: '14:00', description: '제주의 상징 용두암 관광', sequence: 3),
        Place(name: '동문시장', category: '쇼핑', time: '16:00', description: '제주 전통시장 구경', sequence: 4),
        Place(name: '흑돼지 거리', category: '음식', time: '18:00', description: '제주 흑돼지 구이 저녁식사', sequence: 5),
        Place(name: '제주칼호텔', category: '숙박', time: '20:00', description: '제주시내 숙박', sequence: 6),
      ]),
      TravelPlan(day: '2일차', places: [
        Place(name: '성산일출봉', category: '관광지', time: '08:00', description: '유네스코 세계자연유산 성산일출봉 등반', sequence: 1),
        Place(name: '일출랜드', category: '관광지', time: '10:30', description: '성산일출봉 주변 관광', sequence: 2),
        Place(name: '해녀의집', category: '음식', time: '12:30', description: '신선한 해산물 점심', sequence: 3),
        Place(name: '섭지코지', category: '관광지', time: '15:00', description: '아름다운 해안 절경 감상', sequence: 4),
        Place(name: '서귀포매일올레시장', category: '쇼핑', time: '17:30', description: '서귀포 전통시장 구경', sequence: 5),
        Place(name: '신라호텔제주', category: '숙박', time: '20:00', description: '서귀포 리조트 숙박', sequence: 6),
      ]),
      TravelPlan(day: '3일차', places: [
        Place(name: '한라산 어리목탐방로', category: '관광지', time: '09:00', description: '한라산 등반 (어리목 코스)', sequence: 1),
        Place(name: '한라산 1100고지', category: '관광지', time: '13:00', description: '한라산 1100고지 휴게소', sequence: 2),
        Place(name: '천제연폭포', category: '관광지', time: '15:30', description: '중문 천제연폭포 관광', sequence: 3),
        Place(name: '중문관광단지', category: '관광지', time: '17:00', description: '중문 해수욕장 산책', sequence: 4),
        Place(name: '돌하르방공원', category: '관광지', time: '18:30', description: '제주 돌하르방 구경', sequence: 5),
        Place(name: '롯데호텔제주', category: '숙박', time: '20:00', description: '중문 리조트 숙박', sequence: 6),
      ]),
      TravelPlan(day: '4일차', places: [
        Place(name: '주상절리대', category: '관광지', time: '09:30', description: '중문 주상절리대 관광', sequence: 1),
        Place(name: '테디베어뮤지엄', category: '관광지', time: '11:00', description: '중문 테디베어뮤지엄 관람', sequence: 2),
        Place(name: '제주신라면세점', category: '쇼핑', time: '13:00', description: '면세점에서 쇼핑', sequence: 3),
        Place(name: '제주국제공항', category: '교통', time: '15:30', description: '제주공항에서 출발', sequence: 4),
      ]),
    ];
  }

  // 라인에서 장소명 추출
  String _extractPlaceNameFromLine(String line) {
    // 실제 제주도 유명 장소들
    List<String> knownPlaces = [
      '제주국제공항', '명진전복', '용두암', '동문시장', '흑돼지 거리', '제주칼호텔',
      '성산일출봉', '일출랜드', '해녀의집', '섭지코지', '서귀포매일올레시장', '신라호텔제주',
      '한라산 어리목탐방로', '한라산 1100고지', '천제연폭포', '중문관광단지', '돌하르방공원', '롯데호텔제주',
      '주상절리대', '테디베어뮤지엄', '제주신라면세점'
    ];
    
    for (String place in knownPlaces) {
      if (line.contains(place)) return place;
    }
    
    // 일반적인 패턴으로 추출
    RegExp generalPattern = RegExp(r'([가-힣\s]+(?:호텔|펜션|식당|카페|공원|봉|암|포|시장|마트|공항|본점|휴게소|레스토랑|뮤지엄|면세점|거리|랜드))');
    RegExpMatch? match = generalPattern.firstMatch(line);
    return match?.group(1)?.trim() ?? '';
  }

  // 카테고리 분류
  String _categorizeFromLine(String line) {
    if (line.contains('호텔') || line.contains('펜션') || line.contains('숙박') || line.contains('체크아웃')) return '숙박';
    if (line.contains('식당') || line.contains('맛집') || line.contains('국수') || line.contains('식사') || line.contains('조식')) return '음식';
    if (line.contains('시장') || line.contains('쇼핑') || line.contains('마트') || line.contains('기념품')) return '쇼핑';
    if (line.contains('공항') || line.contains('도착') || line.contains('출발')) return '교통';
    return '관광지';
  }

  // 기본 좌표 (카카오 API 검색 실패 시 제주도 중심 좌표 사용)
  double _getLatitude(String placeName, String day) {
    // 카카오 API가 실제 좌표를 검색하므로 기본값만 제공
    return 33.3617; // 제주도 중심
  }

  double _getLongitude(String placeName, String day) {
    // 카카오 API가 실제 좌표를 검색하므로 기본값만 제공  
    return 126.5292; // 제주도 중심
  }

  // 예시 여행 일정 메시지 추가 (로컬 히스토리에 저장)
  void _addSampleTravelPlan() async {
    try {
      setState(() {
        _isLoading = true;
      });

      // 새로운 세션 ID 생성 (로컬 전용 표시)
      String newSessionId = 'local_${DateTime.now().millisecondsSinceEpoch}';
      
      // 사용자 메시지와 샘플 응답 준비
      String userMessage = "제주도 3박 4일 여행 일정 추천해줘";
      String samplePlan = TravelPlanService.getSampleTravelPlan();

      // 1. 사용자 메시지 UI에 추가
      final userMsg = {
        'text': userMessage,
        'isUser': true,
        'time': _formatTime(DateTime.now()),
      };
      
      setState(() {
        _messages.add(userMsg);
      });
      _scrollToBottom();

      // 2. 로딩 시뮬레이션 (1.5초)
      await Future.delayed(const Duration(milliseconds: 1500));

      // 3. 챗봇 응답 UI에 추가 (샘플 데이터)
      final botMsg = {
        'text': samplePlan,
        'isUser': false,
        'time': _formatTime(DateTime.now()),
      };

      setState(() {
        _messages.add(botMsg);
        _isLoading = false;
      });
      _scrollToBottom();

      // 4. 로컬 채팅방 히스토리에 저장
      await _chatRoomService.createChatRoom(newSessionId, userMessage);
      
      // 5. 마지막 메시지를 챗봇 응답으로 업데이트
      await _chatRoomService.updateLastMessage(newSessionId, samplePlan);
      
      // 6. 현재 세션 ID 업데이트
      _currentSessionId = newSessionId;
      _isFirstMessage = false;
      
      print('✅ 샘플 여행 일정이 로컬 히스토리에 저장되었습니다. Session ID: $newSessionId');
      
      // 6. 사이드바 히스토리 새로고침을 위한 알림
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('샘플 여행 일정이 채팅 히스토리에 추가되었습니다! 사이드바를 확인해보세요 🗺️'),
          backgroundColor: Colors.green,
          duration: Duration(seconds: 2),
        ),
      );

    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      
      print('❌ 샘플 여행 일정 추가 오류: $e');
      
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('샘플 여행 일정 추가 중 오류가 발생했습니다: $e'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  // 로컬 샘플 메시지 로드
  void _loadLocalSampleMessages() {
    String userMessage = "제주도 3박 4일 여행 일정 추천해줘";
    String samplePlan = TravelPlanService.getSampleTravelPlan();
    
    setState(() {
      _messages = [
        {
          'text': userMessage,
          'isUser': true,
          'time': '10:30',
        },
        {
          'text': samplePlan,
          'isUser': false,
          'time': '10:32',
        },
      ];
    });
    
    // 메시지 로드 후 맨 아래로 스크롤
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _scrollToBottom();
    });
    
    print('✅ 로컬 샘플 메시지 로드 완료');
  }

  // 시간 포맷팅 헬퍼 메서드
  String _formatTime(DateTime time) {
    return '${time.hour.toString().padLeft(2, '0')}:${time.minute.toString().padLeft(2, '0')}';
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

      // 현재 채팅방의 제목이 변경되었는지 확인 (앱바는 자동으로 새로고침됨)

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

      // 현재 채팅방을 삭제하려는 경우
      if (chatRoom.sessionId == _currentSessionId) {
        // 현재 채팅방 삭제 시 이전 화면으로 이동
        Navigator.pop(context); // 사이드바 닫기
        Navigator.pop(context); // 채팅방 나가기
      }

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

      // 히스토리 새로고침 (현재 채팅방이 아닌 경우에만)
      if (chatRoom.sessionId != _currentSessionId) {
        await _loadChatHistory();
      }
      
      // 성공 메시지
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('채팅방이 삭제되었습니다.'),
            backgroundColor: Colors.green,
          ),
        );
      }
    } catch (e) {
      // 에러 메시지
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('채팅방 삭제에 실패했습니다: $e'),
            backgroundColor: Colors.red,
          ),
        );
      }
    }
  }
} 