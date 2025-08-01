import 'package:flutter/material.dart';
import 'dart:async';
import 'chat_room_screen.dart';

class FestivalLoadingScreen extends StatefulWidget {
  final int selectedGroup;
  final String groupName;
  final String? userName;
  final String? personality;
  final String? travelStyle;

  const FestivalLoadingScreen({
    super.key,
    required this.selectedGroup,
    required this.groupName,
    this.userName,
    this.personality,
    this.travelStyle,
  });

  @override
  State<FestivalLoadingScreen> createState() => _FestivalLoadingScreenState();
}

class _FestivalLoadingScreenState extends State<FestivalLoadingScreen> 
    with TickerProviderStateMixin {
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();
    
    // 맥박 애니메이션 설정
    _pulseController = AnimationController(
      duration: const Duration(seconds: 1),
      vsync: this,
    );
    _pulseAnimation = Tween<double>(
      begin: 0.95,
      end: 1.05,
    ).animate(CurvedAnimation(
      parent: _pulseController,
      curve: Curves.easeInOut,
    ));
    
    // 애니메이션 반복
    _pulseController.repeat(reverse: true);
    
    // 20초 후 채팅 화면으로 이동
    Timer(const Duration(seconds: 20), () {
      if (mounted) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (context) => ChatRoomScreen(
              // 개인화된 채팅으로 시작
              chatTitle: widget.userName != null 
                ? '${widget.userName}님의 개인화된 제주 여행 상담'
                : '프로메테우스 ${widget.groupName} 채팅',
              userName: widget.userName,
              personality: widget.personality,
              travelStyle: widget.travelStyle,
            ),
          ),
        );
      }
    });
  }

  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back, color: Colors.black),
          onPressed: () => Navigator.pop(context),
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
        // actions에서 김 아이콘 제거
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // 오르미 캐릭터 (애니메이션 적용)
            AnimatedBuilder(
              animation: _pulseAnimation,
              builder: (context, child) {
                return Transform.scale(
                  scale: _pulseAnimation.value,
                  child: Container(
                    width: 180,
                    height: 180,
                    child: Image.asset(
                      'assets/images/ormi_travel.png',
                      fit: BoxFit.contain,
                    ),
                  ),
                );
              },
            ),
            
            const SizedBox(height: 40),
            
            // 로딩 메시지
            Text(
              widget.userName != null
                ? '${widget.userName}님에게 맞는\n개인화된 채팅 환경을 세팅 중이에요.'
                : '${widget.groupName}에 맞는\n 채팅 환경을 세팅 중이에요.',
              textAlign: TextAlign.center,
              style: const TextStyle(
                fontSize: 18,
                color: Colors.black87,
                fontWeight: FontWeight.w500,
                height: 1.4,
              ),
            ),
            
            const SizedBox(height: 30),
            
            // 로딩 인디케이터
            const CircularProgressIndicator(
              valueColor: AlwaysStoppedAnimation<Color>(Color(0xFF6B73FF)),
              strokeWidth: 3,
            ),
            
            const SizedBox(height: 20),
            
            // 로딩 텍스트
            const Text(
              '잠시만 기다려주세요...',
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey,
                fontWeight: FontWeight.w400,
              ),
            ),
          ],
        ),
      ),
    );
  }
} 