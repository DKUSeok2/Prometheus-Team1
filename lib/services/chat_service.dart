import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class ChatService {
  static const String _chatHistoryKey = 'chat_history';
  static const String _currentChatKey = 'current_chat';

  // 채팅 대화 모델
  static Map<String, dynamic> createChatConversation(String title) {
    return {
      'id': DateTime.now().millisecondsSinceEpoch.toString(),
      'title': title,
      'lastMessage': '새로운 대화',
      'timestamp': DateTime.now().toIso8601String(),
      'messages': [
        {
          'text': '안녕하세요! 오르다입니다. 당신의 여행을 도와드릴게요.',
          'isUser': false,
          'time': getCurrentTime(),
        },
      ],
    };
  }

  // 현재 시간 가져오기
  static String getCurrentTime() {
    final now = DateTime.now();
    return '${now.hour}:${now.minute.toString().padLeft(2, '0')}';
  }

  // 채팅 기록 가져오기
  Future<List<Map<String, dynamic>>> getChatHistory() async {
    final prefs = await SharedPreferences.getInstance();
    final historyJson = prefs.getString(_chatHistoryKey);
    
    if (historyJson != null) {
      final List<dynamic> historyList = json.decode(historyJson);
      return historyList.cast<Map<String, dynamic>>();
    }
    
    // 기본 채팅 생성
    final defaultChat = createChatConversation('오르다와의 대화');
    await saveChatHistory([defaultChat]);
    return [defaultChat];
  }

  // 채팅 기록 저장
  Future<void> saveChatHistory(List<Map<String, dynamic>> history) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_chatHistoryKey, json.encode(history));
  }

  // 현재 선택된 채팅 ID 가져오기
  Future<String?> getCurrentChatId() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_currentChatKey);
  }

  // 현재 선택된 채팅 ID 설정
  Future<void> setCurrentChatId(String chatId) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_currentChatKey, chatId);
  }

  // 새로운 채팅 생성
  Future<Map<String, dynamic>> createNewChat() async {
    final history = await getChatHistory();
    final newChat = createChatConversation('새로운 대화 ${history.length + 1}');
    
    history.insert(0, newChat);
    await saveChatHistory(history);
    await setCurrentChatId(newChat['id'] as String);
    
    return newChat;
  }

  // 특정 채팅 가져오기
  Future<Map<String, dynamic>?> getChatById(String chatId) async {
    final history = await getChatHistory();
    try {
      return history.firstWhere((chat) => chat['id'] == chatId);
    } catch (e) {
      return null;
    }
  }

  // 채팅에 메시지 추가
  Future<void> addMessageToChat(String chatId, Map<String, dynamic> message) async {
    final history = await getChatHistory();
    final chatIndex = history.indexWhere((chat) => chat['id'] == chatId);
    
    if (chatIndex != -1) {
      history[chatIndex]['messages'].add(message);
      history[chatIndex]['lastMessage'] = message['text'];
      history[chatIndex]['timestamp'] = DateTime.now().toIso8601String();
      
      // 최근 사용한 채팅을 맨 위로 이동
      final updatedChat = history.removeAt(chatIndex);
      history.insert(0, updatedChat);
      
      await saveChatHistory(history);
    }
  }

  // 채팅 삭제
  Future<void> deleteChat(String chatId) async {
    final history = await getChatHistory();
    history.removeWhere((chat) => chat['id'] == chatId);
    await saveChatHistory(history);
    
    // 삭제된 채팅이 현재 선택된 채팅이면 초기화
    final currentChatId = await getCurrentChatId();
    if (currentChatId == chatId) {
      final prefs = await SharedPreferences.getInstance();
      await prefs.remove(_currentChatKey);
    }
  }

  // 채팅 제목 업데이트
  Future<void> updateChatTitle(String chatId, String newTitle) async {
    final history = await getChatHistory();
    final chatIndex = history.indexWhere((chat) => chat['id'] == chatId);
    
    if (chatIndex != -1) {
      history[chatIndex]['title'] = newTitle;
      await saveChatHistory(history);
    }
  }

  // 자동 응답 생성 (간단한 예시)
  static Map<String, dynamic> generateAutoResponse(String userMessage) {
    String response = '네, 도와드리겠습니다! 어떤 도움이 필요하신가요?';
    
    if (userMessage.contains('여행')) {
      response = '여행 관련 질문이시네요! 어떤 여행지를 찾고 계신가요?';
    } else if (userMessage.contains('맛집')) {
      response = '맛집 추천이 필요하시군요! 어떤 지역의 맛집을 찾고 계신가요?';
    } else if (userMessage.contains('안녕')) {
      response = '안녕하세요! 오르다입니다. 즐거운 여행을 계획해보세요!';
    }
    
    return {
      'text': response,
      'isUser': false,
      'time': getCurrentTime(),
    };
  }
} 