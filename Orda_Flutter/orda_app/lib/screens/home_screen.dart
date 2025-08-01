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

  void _changeTab(int index) {
    setState(() {
      _currentIndex = index;
    });
  }

  List<Widget> get _screens => [
    const HomeTab(),
    ChatScreen(onTabChange: _changeTab),
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

// 여행 상태 enum
enum TravelStatus {
  resting('쉬는 중', 'assets/images/rest.png', '집에서 편안하게 쉬는 중'),
  preparing('여행 준비', 'assets/images/tavel_prepare.png', '여행 계획을 세우고 준비하는 중'),
  traveling('여행 중', 'assets/images/traveling.png', '2박 3일 홀로 제주도 여행'),
  finished('여행 끝', 'assets/images/travel_finish.png', '즐거운 여행을 마치고 돌아옴');

  const TravelStatus(this.label, this.imagePath, this.description);
  final String label;
  final String imagePath;
  final String description;
}

class HomeTab extends StatefulWidget {
  const HomeTab({super.key});

  @override
  State<HomeTab> createState() => _HomeTabState();
}

class _HomeTabState extends State<HomeTab> {
  TravelStatus _currentStatus = TravelStatus.resting; // 기본값: 쉬는중

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Column(
          children: [
            const SizedBox(height: 60),
            
            // 현재 여행 상태 정보
            Container(
              margin: const EdgeInsets.symmetric(horizontal: 20),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [
                    Color(0xFF6B73FF),
                    Color(0xFF9C88FF),
                    Color(0xFF70D0FF),
                    Color(0xFF98FB98),
                  ],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(24),
              ),
              padding: const EdgeInsets.all(2), // 그라데이션 테두리 두께
              child: Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(22),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(
                          _getStatusIcon(_currentStatus),
                          color: const Color(0xFF6B73FF),
                          size: 18,
                        ),
                        const SizedBox(width: 8),
                        const Text(
                          '현재 진행 중인 여행',
                          style: TextStyle(
                            fontSize: 14,
                            color: Colors.grey,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Text(
                      _currentStatus.description,
                      style: const TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: Color(0xFF333333),
                      ),
                    ),
                  ],
                ),
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
                    // 상태별 캐릭터 이미지 (위치 일정하게 정렬)
                    Container(
                      width: 220,
                      height: 220,
                      alignment: Alignment.center, // 중앙 정렬로 위치 일정하게
                      child: Image.asset(
                        _currentStatus.imagePath,
                        width: 200,
                        height: 200,
                        fit: BoxFit.contain, // 이미지 전체를 보여주되 비율 유지
                        errorBuilder: (context, error, stackTrace) {
                          // 이미지 로드 실패 시 기본 캐릭터 표시
                          return Container(
                            width: 200,
                            height: 200,
                            decoration: const BoxDecoration(
                              gradient: LinearGradient(
                                colors: [
                                  Color(0xFF87CEEB),
                                  Color(0xFF98FB98),
                                ],
                                begin: Alignment.topCenter,
                                end: Alignment.bottomCenter,
                              ),
                              shape: BoxShape.circle,
                            ),
                            child: const Center(
                              child: Text(
                                '🤖',
                                style: TextStyle(fontSize: 60),
                              ),
                            ),
                          );
                        },
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
                    
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(
                          _currentStatus.label,
                          style: const TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                            color: Color(0xFF333333),
                          ),
                        ),
                        const SizedBox(width: 12),
                        // 상태 옆 토글 버튼
                        GestureDetector(
                          onTap: _showStatusSelector,
                          child: Container(
                            padding: const EdgeInsets.all(6),
                            decoration: BoxDecoration(
                              color: const Color(0xFF6B73FF).withOpacity(0.1),
                              borderRadius: BorderRadius.circular(20),
                              border: Border.all(
                                color: const Color(0xFF6B73FF).withOpacity(0.3),
                              ),
                            ),
                            child: const Icon(
                              Icons.tune,
                              color: Color(0xFF6B73FF),
                              size: 20,
                            ),
                          ),
                        ),
                      ],
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
  
  // 상태 선택 다이얼로그 표시
  void _showStatusSelector() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (BuildContext context) {
        return Container(
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
              const SizedBox(height: 20),
              Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: Colors.grey[300],
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              const SizedBox(height: 20),
              const Text(
                '상태 변경',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: Color(0xFF333333),
                ),
              ),
              const SizedBox(height: 20),
              ...TravelStatus.values.map((status) => ListTile(
                leading: Icon(
                  _getStatusIcon(status),
                  color: _currentStatus == status 
                      ? const Color(0xFF6B73FF)
                      : Colors.grey,
                ),
                title: Text(
                  status.label,
                  style: TextStyle(
                    fontWeight: _currentStatus == status 
                        ? FontWeight.bold 
                        : FontWeight.normal,
                    color: _currentStatus == status 
                        ? const Color(0xFF6B73FF)
                        : const Color(0xFF333333),
                  ),
                ),
                subtitle: Text(
                  status.description,
                  style: const TextStyle(
                    fontSize: 12,
                    color: Colors.grey,
                  ),
                ),
                trailing: _currentStatus == status 
                    ? const Icon(
                        Icons.check,
                        color: Color(0xFF6B73FF),
                      )
                    : null,
                onTap: () {
                  setState(() {
                    _currentStatus = status;
                  });
                  Navigator.pop(context);
                },
              )).toList(),
              const SizedBox(height: 20),
            ],
          ),
        );
      },
    );
  }
  
  // 상태별 아이콘 반환
  IconData _getStatusIcon(TravelStatus status) {
    switch (status) {
      case TravelStatus.resting:
        return Icons.home;
      case TravelStatus.preparing:
        return Icons.checklist;
      case TravelStatus.traveling:
        return Icons.flight;
      case TravelStatus.finished:
        return Icons.check_circle;
    }
  }
} 