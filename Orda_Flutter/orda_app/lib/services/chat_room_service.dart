import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';

class ChatRoom {
  final String sessionId;
  final String title;
  final String lastMessage;
  final DateTime lastChatTime;
  final DateTime createdAt;

  ChatRoom({
    required this.sessionId,
    required this.title,
    required this.lastMessage,
    required this.lastChatTime,
    required this.createdAt,
  });

  Map<String, dynamic> toJson() {
    return {
      'sessionId': sessionId,
      'title': title,
      'lastMessage': lastMessage,
      'lastChatTime': lastChatTime.toIso8601String(),
      'createdAt': createdAt.toIso8601String(),
    };
  }

  factory ChatRoom.fromJson(Map<String, dynamic> json) {
    return ChatRoom(
      sessionId: json['sessionId'],
      title: json['title'],
      lastMessage: json['lastMessage'],
      lastChatTime: DateTime.parse(json['lastChatTime']),
      createdAt: DateTime.parse(json['createdAt']),
    );
  }
}

class ChatRoomService {
  static const String _chatRoomsKey = 'chat_rooms';

  // 모든 채팅방 가져오기 (최신순)
  Future<List<ChatRoom>> getAllChatRooms() async {
    final prefs = await SharedPreferences.getInstance();
    final chatRoomsJson = prefs.getString(_chatRoomsKey);
    
    if (chatRoomsJson == null) return [];
    
    final List<dynamic> chatRoomsList = json.decode(chatRoomsJson);
    final chatRooms = chatRoomsList.map((json) => ChatRoom.fromJson(json)).toList();
    
    // 최신순으로 정렬
    chatRooms.sort((a, b) => b.lastChatTime.compareTo(a.lastChatTime));
    
    return chatRooms;
  }

  // 새 채팅방 생성
  Future<ChatRoom> createChatRoom(String sessionId, String firstMessage) async {
    final now = DateTime.now();
    
    // 채팅방 제목 생성 (첫 메시지의 첫 15글자 + "...")
    String title = firstMessage.length > 15 
        ? '${firstMessage.substring(0, 15)}...'
        : firstMessage;
    
    final chatRoom = ChatRoom(
      sessionId: sessionId,
      title: title,
      lastMessage: firstMessage,
      lastChatTime: now,
      createdAt: now,
    );

    await _saveChatRoom(chatRoom);
    return chatRoom;
  }

  // 채팅방 마지막 메시지 업데이트
  Future<void> updateLastMessage(String sessionId, String lastMessage) async {
    final chatRooms = await getAllChatRooms();
    final index = chatRooms.indexWhere((room) => room.sessionId == sessionId);
    
    if (index != -1) {
      final updatedRoom = ChatRoom(
        sessionId: chatRooms[index].sessionId,
        title: chatRooms[index].title,
        lastMessage: lastMessage,
        lastChatTime: DateTime.now(),
        createdAt: chatRooms[index].createdAt,
      );
      
      chatRooms[index] = updatedRoom;
      await _saveAllChatRooms(chatRooms);
    }
  }

  // 채팅방 저장
  Future<void> _saveChatRoom(ChatRoom chatRoom) async {
    final chatRooms = await getAllChatRooms();
    
    // 중복 세션 ID 제거
    chatRooms.removeWhere((room) => room.sessionId == chatRoom.sessionId);
    
    // 새 채팅방을 맨 앞에 추가
    chatRooms.insert(0, chatRoom);
    
    await _saveAllChatRooms(chatRooms);
  }

  // 모든 채팅방 저장
  Future<void> _saveAllChatRooms(List<ChatRoom> chatRooms) async {
    final prefs = await SharedPreferences.getInstance();
    final chatRoomsJson = json.encode(chatRooms.map((room) => room.toJson()).toList());
    await prefs.setString(_chatRoomsKey, chatRoomsJson);
  }

  // 현재 시간을 한국 시간으로 포맷
  String formatTime(DateTime dateTime) {
    final now = DateTime.now();
    final difference = now.difference(dateTime);
    
    if (difference.inMinutes < 1) {
      return '방금 전';
    } else if (difference.inMinutes < 60) {
      return '${difference.inMinutes}분 전';
    } else if (difference.inHours < 24) {
      return '${difference.inHours}시간 전';
    } else if (difference.inDays < 7) {
      return '${difference.inDays}일 전';
    } else {
      return '${dateTime.year}.${dateTime.month.toString().padLeft(2, '0')}.${dateTime.day.toString().padLeft(2, '0')}';
    }
  }

  // 모든 채팅방 삭제 (초기화)
  Future<void> clearAllChatRooms() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_chatRoomsKey);
  }

  // 채팅방 제목 변경
  Future<void> updateChatRoomTitle(String sessionId, String newTitle) async {
    List<ChatRoom> chatRooms = await getAllChatRooms();
    
    for (int i = 0; i < chatRooms.length; i++) {
      if (chatRooms[i].sessionId == sessionId) {
        chatRooms[i] = ChatRoom(
          sessionId: chatRooms[i].sessionId,
          title: newTitle, // 새로운 제목으로 변경
          lastMessage: chatRooms[i].lastMessage,
          lastChatTime: chatRooms[i].lastChatTime,
          createdAt: chatRooms[i].createdAt,
        );
        break;
      }
    }
    
    await _saveAllChatRooms(chatRooms);
  }

  // 채팅방 삭제
  Future<void> deleteChatRoom(String sessionId) async {
    List<ChatRoom> chatRooms = await getAllChatRooms();
    chatRooms.removeWhere((room) => room.sessionId == sessionId);
    await _saveAllChatRooms(chatRooms);
  }

  // 특정 채팅방 조회
  Future<ChatRoom?> getChatRoom(String sessionId) async {
    List<ChatRoom> chatRooms = await getAllChatRooms();
    try {
      return chatRooms.firstWhere((room) => room.sessionId == sessionId);
    } catch (e) {
      return null;
    }
  }
} 