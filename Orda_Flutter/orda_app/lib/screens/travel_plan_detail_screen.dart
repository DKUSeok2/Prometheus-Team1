import 'package:flutter/material.dart';
import 'package:flutter_markdown/flutter_markdown.dart';

class TravelPlanDetailScreen extends StatelessWidget {
  final String userName;
  final String personality;
  final String travelStyle;
  final String travelPlan;

  const TravelPlanDetailScreen({
    super.key,
    required this.userName,
    required this.personality,
    required this.travelStyle,
    required this.travelPlan,
  });

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
        title: Text(
          '$userName님의 제주여행',
          style: const TextStyle(
            color: Colors.black,
            fontSize: 18,
            fontWeight: FontWeight.w600,
          ),
        ),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.share, color: Colors.black),
            onPressed: () {
              // 공유 기능 (추후 구현)
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('공유 기능 준비 중입니다.')),
              );
            },
          ),
        ],
      ),
      body: Column(
        children: [
          // 사용자 프로필 정보
          Container(
            width: double.infinity,
            margin: const EdgeInsets.all(16),
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  personality.contains('에겐') 
                    ? Colors.pink.withOpacity(0.1)
                    : Colors.blue.withOpacity(0.1),
                  personality.contains('에겐')
                    ? Colors.orange.withOpacity(0.1) 
                    : Colors.indigo.withOpacity(0.1),
                ],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                color: personality.contains('에겐') 
                  ? Colors.pink.withOpacity(0.3)
                  : Colors.blue.withOpacity(0.3),
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: personality.contains('에겐') 
                          ? Colors.pink.withOpacity(0.2)
                          : Colors.blue.withOpacity(0.2),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Text(
                        personality,
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          color: personality.contains('에겐') 
                            ? Colors.pink[700]
                            : Colors.blue[700],
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      '$userName님',
                      style: const TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        color: Colors.black87,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Text(
                  '여행스타일',
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.grey[600],
                    fontWeight: FontWeight.w500,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  travelStyle,
                  style: const TextStyle(
                    fontSize: 14,
                    color: Colors.black87,
                    height: 1.3,
                  ),
                ),
              ],
            ),
          ),
          
          // 여행일정 타이틀
          Container(
            width: double.infinity,
            margin: const EdgeInsets.symmetric(horizontal: 16),
            child: const Text(
              '🗓️ 맞춤형 제주 여행일정',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Colors.black87,
              ),
            ),
          ),
          
          const SizedBox(height: 8),

          // 여행일정 마크다운 내용
          Expanded(
            child: Container(
              margin: const EdgeInsets.symmetric(horizontal: 16),
              decoration: BoxDecoration(
                color: Colors.grey[50],
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.grey[200]!),
              ),
              child: Markdown(
                data: travelPlan,
                styleSheet: MarkdownStyleSheet(
                  // 제목 스타일
                  h3: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                    color: Colors.black87,
                    height: 1.5,
                  ),
                  // 본문 스타일
                  p: const TextStyle(
                    fontSize: 14,
                    color: Colors.black87,
                    height: 1.4,
                  ),
                  // 리스트 스타일
                  listBullet: const TextStyle(
                    fontSize: 14,
                    color: Colors.black87,
                  ),
                  // 강조 텍스트 스타일
                  strong: const TextStyle(
                    fontWeight: FontWeight.bold,
                    color: Colors.black,
                  ),
                  // 블록 인용문 스타일 (시간대별 구분용)
                  blockquote: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    color: Colors.blue[700],
                    fontStyle: FontStyle.normal,
                  ),
                  // 코드 스타일 (시간 표시용)
                  code: TextStyle(
                    backgroundColor: Colors.blue[50],
                    color: Colors.blue[700],
                    fontSize: 12,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                padding: const EdgeInsets.all(20),
              ),
            ),
          ),
          
          const SizedBox(height: 16),
        ],
      ),
    );
  }
}