import 'package:flutter/material.dart';
import '../services/chat_service.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _messageController = TextEditingController();
  final ChatService _chatService = ChatService();
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();
  
  List<Map<String, dynamic>> _chatHistory = [];
  Map<String, dynamic>? _currentChat;
  List<Map<String, dynamic>> _currentMessages = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadChatData();
  }

  Future<void> _loadChatData() async {
    try {
      final history = await _chatService.getChatHistory();
      final currentChatId = await _chatService.getCurrentChatId();
      
      setState(() {
        _chatHistory = history;
        _isLoading = false;
      });
      
      if (currentChatId != null) {
        await _selectChat(currentChatId);
      } else if (history.isNotEmpty) {
        await _selectChat(history.first['id']);
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<void> _selectChat(String chatId) async {
    final chat = await _chatService.getChatById(chatId);
    if (chat != null) {
      await _chatService.setCurrentChatId(chatId);
      setState(() {
        _currentChat = chat;
        _currentMessages = List<Map<String, dynamic>>.from(chat['messages']);
      });
    }
  }

  Future<void> _createNewChat() async {
    final newChat = await _chatService.createNewChat();
    await _loadChatData();
    await _selectChat(newChat['id'] as String);
  }

  Future<void> _sendMessage() async {
    if (_messageController.text.trim().isEmpty || _currentChat == null) return;
    
    final userMessage = {
      'text': _messageController.text.trim(),
      'isUser': true,
      'time': ChatService.getCurrentTime(),
    };
    
    // 현재 채팅에 사용자 메시지 추가
    await _chatService.addMessageToChat(_currentChat!['id'] as String, userMessage);
    
    setState(() {
      _currentMessages.add(userMessage);
    });
    
    _messageController.clear();
    
    // 자동 응답 생성
    Future.delayed(const Duration(seconds: 1), () async {
      final autoResponse = ChatService.generateAutoResponse(userMessage['text'] as String);
      
      await _chatService.addMessageToChat(_currentChat!['id'] as String, autoResponse);
      
      setState(() {
        _currentMessages.add(autoResponse);
      });
      
      await _loadChatData();
    });
  }

  Future<void> _deleteChat(String chatId) async {
    await _chatService.deleteChat(chatId);
    await _loadChatData();
    
    if (_currentChat?['id'] == chatId) {
      setState(() {
        _currentChat = null;
        _currentMessages = [];
      });
      
      if (_chatHistory.isNotEmpty) {
        await _selectChat(_chatHistory.first['id'] as String);
      }
    }
  }

  Future<void> _editChatTitle(String chatId, String currentTitle) async {
    final TextEditingController titleController = TextEditingController(text: currentTitle);
    
    final result = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('채팅 이름 변경'),
        content: TextField(
          controller: titleController,
          decoration: const InputDecoration(
            hintText: '새로운 채팅 이름을 입력하세요',
            border: OutlineInputBorder(),
          ),
          autofocus: true,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('취소'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(context, titleController.text.trim()),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF6B73FF),
              foregroundColor: Colors.white,
            ),
            child: const Text('변경'),
          ),
        ],
      ),
    );
    
    if (result != null && result.isNotEmpty && result != currentTitle) {
      await _chatService.updateChatTitle(chatId, result);
      await _loadChatData();
      
      // 현재 선택된 채팅의 제목이 변경된 경우 UI 업데이트
      if (_currentChat?['id'] == chatId) {
        setState(() {
          _currentChat!['title'] = result;
        });
      }
    }
    
    titleController.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      key: _scaffoldKey,
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: Text(_currentChat?['title'] ?? '채팅하기'),
        backgroundColor: Colors.white,
        foregroundColor: Colors.black,
        elevation: 1,
        leading: IconButton(
          icon: const Icon(Icons.menu),
          onPressed: () => _scaffoldKey.currentState?.openDrawer(),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.add),
            onPressed: _createNewChat,
          ),
        ],
      ),
      drawer: _buildSidebar(),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : _currentChat == null
              ? _buildWelcomeScreen()
              : _buildChatInterface(),
    );
  }

  Widget _buildSidebar() {
    return Drawer(
      child: Column(
        children: [
          // 사이드바 헤더
          Container(
            height: 120,
            width: double.infinity,
            decoration: const BoxDecoration(
              color: Color(0xFF6B73FF),
            ),
            child: SafeArea(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
                child: Row(
                  children: [
                    const Icon(Icons.chat, color: Colors.white, size: 28),
                    const SizedBox(width: 12),
                    const Text(
                      '채팅 기록',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          
          // 새로운 채팅 생성 버튼 (더 컴팩트)
          Container(
            margin: const EdgeInsets.all(12),
            child: ElevatedButton(
              onPressed: () {
                _createNewChat();
                Navigator.pop(context);
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF6B73FF),
                foregroundColor: Colors.white,
                minimumSize: const Size(double.infinity, 44),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.add, size: 20),
                  SizedBox(width: 8),
                  Text('새로운 채팅', style: TextStyle(fontSize: 14)),
                ],
              ),
            ),
          ),
          
          // 채팅 리스트
          Expanded(
            child: _chatHistory.isEmpty
                ? const Center(
                    child: Text(
                      '채팅 기록이 없습니다',
                      style: TextStyle(
                        color: Colors.grey,
                        fontSize: 16,
                      ),
                    ),
                  )
                : ListView.separated(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    itemCount: _chatHistory.length,
                    separatorBuilder: (context, index) => const SizedBox(height: 4),
                    itemBuilder: (context, index) {
                      final chat = _chatHistory[index];
                      final isSelected = _currentChat?['id'] == chat['id'];
                      
                      return Container(
                        margin: const EdgeInsets.symmetric(horizontal: 8),
                        decoration: BoxDecoration(
                          color: isSelected ? const Color(0xFF6B73FF).withOpacity(0.1) : null,
                          borderRadius: BorderRadius.circular(12),
                          border: isSelected 
                              ? Border.all(color: const Color(0xFF6B73FF).withOpacity(0.3))
                              : null,
                        ),
                        child: ListTile(
                          contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                          leading: CircleAvatar(
                            backgroundColor: const Color(0xFF6B73FF),
                            radius: 22,
                            child: Text(
                              (chat['title'] as String).isNotEmpty 
                                  ? (chat['title'] as String)[0].toUpperCase() 
                                  : '오',
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                          title: Text(
                            chat['title'] as String,
                            style: TextStyle(
                              fontSize: 15,
                              fontWeight: isSelected ? FontWeight.w600 : FontWeight.w500,
                              color: isSelected ? const Color(0xFF6B73FF) : Colors.black87,
                            ),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                          subtitle: Padding(
                            padding: const EdgeInsets.only(top: 4),
                            child: Text(
                              chat['lastMessage'] as String,
                              style: TextStyle(
                                fontSize: 13,
                                color: Colors.grey[600],
                              ),
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                          trailing: PopupMenuButton<String>(
                            onSelected: (value) {
                              if (value == 'edit') {
                                _editChatTitle(chat['id'] as String, chat['title'] as String);
                              } else if (value == 'delete') {
                                _deleteChat(chat['id'] as String);
                              }
                            },
                            itemBuilder: (context) => [
                              const PopupMenuItem(
                                value: 'edit',
                                child: Row(
                                  children: [
                                    Icon(Icons.edit, size: 18, color: Color(0xFF6B73FF)),
                                    SizedBox(width: 8),
                                    Text('이름 변경'),
                                  ],
                                ),
                              ),
                              const PopupMenuItem(
                                value: 'delete',
                                child: Row(
                                  children: [
                                    Icon(Icons.delete, size: 18, color: Colors.red),
                                    SizedBox(width: 8),
                                    Text('삭제'),
                                  ],
                                ),
                              ),
                            ],
                          ),
                          onTap: () {
                            _selectChat(chat['id'] as String);
                            Navigator.pop(context);
                          },
                        ),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }

  Widget _buildWelcomeScreen() {
    return const Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.chat_bubble_outline,
            size: 64,
            color: Color(0xFF6B73FF),
          ),
          SizedBox(height: 16),
          Text(
            '새로운 채팅을 시작해보세요!',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.w500,
              color: Colors.grey,
            ),
          ),
          SizedBox(height: 8),
          Text(
            '왼쪽 상단의 메뉴를 클릭하여 채팅 기록을 확인하거나\n+ 버튼을 눌러 새로운 채팅을 시작하세요.',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 14,
              color: Colors.grey,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildChatInterface() {
    return Column(
      children: [
        // 메시지 리스트
        Expanded(
          child: ListView.builder(
            padding: const EdgeInsets.all(16),
            itemCount: _currentMessages.length,
            itemBuilder: (context, index) {
              final message = _currentMessages[index];
              return _buildMessageBubble(message);
            },
          ),
        ),
        
        // 메시지 입력 영역
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.grey[50],
            border: Border(
              top: BorderSide(color: Colors.grey[200]!),
            ),
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
                      borderSide: BorderSide.none,
                    ),
                    filled: true,
                    fillColor: Colors.white,
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 20,
                      vertical: 10,
                    ),
                  ),
                  onSubmitted: (_) => _sendMessage(),
                ),
              ),
              const SizedBox(width: 8),
              IconButton(
                onPressed: _sendMessage,
                icon: const Icon(Icons.send),
                style: IconButton.styleFrom(
                  backgroundColor: const Color(0xFF6B73FF),
                  foregroundColor: Colors.white,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildMessageBubble(Map<String, dynamic> message) {
    final isUser = message['isUser'] as bool;
    
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      child: Row(
        mainAxisAlignment: isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          if (!isUser) ...[
            CircleAvatar(
              radius: 16,
              backgroundColor: const Color(0xFF6B73FF),
              child: const Text(
                '오',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            const SizedBox(width: 8),
          ],
          
          Flexible(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              decoration: BoxDecoration(
                color: isUser ? const Color(0xFF6B73FF) : Colors.grey[100],
                borderRadius: BorderRadius.circular(16),
              ),
              child: Text(
                message['text'],
                style: TextStyle(
                  color: isUser ? Colors.white : Colors.black,
                  fontSize: 14,
                ),
              ),
            ),
          ),
          
          if (isUser) ...[
            const SizedBox(width: 8),
            Text(
              message['time'],
              style: TextStyle(
                fontSize: 12,
                color: Colors.grey[600],
              ),
            ),
          ],
        ],
      ),
    );
  }

  @override
  void dispose() {
    _messageController.dispose();
    super.dispose();
  }
} 