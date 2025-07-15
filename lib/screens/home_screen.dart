import 'package:flutter/material.dart';
import 'chat_screen.dart';
import 'map_screen.dart';
import 'my_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _currentIndex = 0;

  final List<Widget> _screens = [
    const HomeTab(),
    const ChatScreen(),
    const MapScreen(),
    const MyScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _screens[_currentIndex],
      bottomNavigationBar: BottomNavigationBar(
        type: BottomNavigationBarType.fixed,
        currentIndex: _currentIndex,
        onTap: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
        selectedItemColor: const Color(0xFF6B73FF),
        unselectedItemColor: Colors.grey,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.home_outlined),
            activeIcon: Icon(Icons.home),
            label: '홈',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.chat_bubble_outline),
            activeIcon: Icon(Icons.chat_bubble),
            label: '채팅',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.map_outlined),
            activeIcon: Icon(Icons.map),
            label: '지도',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.person_outline),
            activeIcon: Icon(Icons.person),
            label: 'MY',
          ),
        ],
      ),
    );
  }
}

class HomeTab extends StatelessWidget {
  const HomeTab({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Column(
          children: [
            const SizedBox(height: 20),
            
            // 상단 여행 진행 상황
            Container(
              margin: const EdgeInsets.symmetric(horizontal: 20),
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: const Color(0xFFF0F4FF),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: const Color(0xFFE0E8FF)),
              ),
              child: const Row(
                children: [
                  Icon(
                    Icons.schedule,
                    color: Color(0xFF6B73FF),
                    size: 20,
                  ),
                  SizedBox(width: 8),
                  Text(
                    '2박 3일 홀로 제주도 여행',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                      color: Color(0xFF333333),
                    ),
                  ),
                ],
              ),
            ),
            
            const SizedBox(height: 40),
            
            // 중앙 캐릭터 영역
            Expanded(
              flex: 3,
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    // 캐릭터 (임시로 Container 사용)
                    Container(
                      width: 200,
                      height: 200,
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                          colors: [
                            Color(0xFF87CEEB),
                            Color(0xFF98FB98),
                          ],
                          begin: Alignment.topCenter,
                          end: Alignment.bottomCenter,
                        ),
                        borderRadius: BorderRadius.circular(100),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withOpacity(0.1),
                            blurRadius: 20,
                            offset: const Offset(0, 10),
                          ),
                        ],
                      ),
                                              child: Stack(
                          children: [
                            // 눈
                            const Positioned(
                              top: 60,
                              left: 60,
                              child: CircleAvatar(
                                radius: 8,
                                backgroundColor: Colors.black,
                              ),
                            ),
                            const Positioned(
                              top: 60,
                              right: 60,
                              child: CircleAvatar(
                                radius: 8,
                                backgroundColor: Colors.black,
                              ),
                            ),
                            // 모자 (상단 그라데이션)
                            Positioned(
                              top: 20,
                              left: 50,
                              child: Container(
                                width: 100,
                                height: 40,
                                decoration: const BoxDecoration(
                                  color: Color(0xFFFFD700),
                                  borderRadius: BorderRadius.only(
                                    topLeft: Radius.circular(50),
                                    topRight: Radius.circular(50),
                                  ),
                                ),
                              ),
                            ),
                          ],
                        ),
                    ),
                    
                    const SizedBox(height: 40),
                    
                    // 상태 텍스트
                    const Text(
                      '현재 상태',
                      style: TextStyle(
                        fontSize: 16,
                        color: Colors.grey,
                        fontWeight: FontWeight.w400,
                      ),
                    ),
                    
                    const SizedBox(height: 8),
                    
                    const Text(
                      '여행 중',
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: Color(0xFF333333),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            
            // 하단 여백
            const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }
} 