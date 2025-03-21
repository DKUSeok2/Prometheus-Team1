from langchain_tests import LangChainTester
from langgraph_tests import LangGraphTester
from utils import performance_tracker, calculate_f1_score
import time

def run_comparison_tests():
    # 기존 결과 초기화
    performance_tracker.clear_results()
    
    # 테스트 인스턴스 생성
    print("Initializing testers...")
    langchain_tester = LangChainTester()
    langgraph_tester = LangGraphTester()
    
    # 테스트 질문 및 참조 응답 세트
    test_data = [
        {"question": "What is the capital of France?", "reference": "The capital of France is Paris."},
        {"question": "What is 2+2?", "reference": "2 plus 2 equals 4."},
        {"question": "Who wrote Romeo and Juliet?", "reference": "Romeo and Juliet was written by William Shakespeare."},
        {"question": "What would be the environmental impact of replacing all cars with electric vehicles?", 
         "reference": "Replacing all cars with electric vehicles would reduce greenhouse gas emissions, decrease air pollution, and lower dependency on fossil fuels, but it would also require significant energy resources for battery production and electricity generation."},
        {"question": "Compare and contrast the economic systems of capitalism and socialism.", 
         "reference": "Capitalism is an economic system where private individuals own and control property and businesses, while socialism is characterized by collective or governmental ownership of resources. Capitalism emphasizes free markets and competition, whereas socialism focuses on reducing inequality and providing public services."},
        {"question": "Explain the process of photosynthesis and its importance to life on Earth.", 
         "reference": "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy, carbon dioxide, and water into glucose and oxygen. It is crucial for life on Earth as it provides oxygen for respiration and is the foundation of the food chain."},
        {"question": "How does climate change affect global food security?", 
         "reference": "Climate change affects global food security by altering weather patterns, reducing crop yields, and increasing the frequency of extreme weather events, which can lead to food shortages and increased prices."},
        {"question": "What are the ethical implications of artificial intelligence?", 
         "reference": "The ethical implications of artificial intelligence include concerns about privacy, job displacement, decision-making transparency, and the potential for bias and discrimination in AI systems."}
    ]
    
    # Simple QA 및 Multi-step Reasoning 테스트
    print("\n=== Running Tests ===")
    for data in test_data:
        question = data["question"]
        reference = data["reference"]
        
        print(f"\nTesting question: {question}")
        
        # LangChain 테스트
        print("\nLangChain response:")
        lc_response = langchain_tester.simple_qa(question)
        print(f"Response: {lc_response}")
        lc_precision, lc_recall, lc_f1 = calculate_f1_score(reference, lc_response)
        print(f"LangChain F1 Score: {lc_f1:.2f} (Precision: {lc_precision:.2f}, Recall: {lc_recall:.2f})")
        
        # LangGraph 테스트
        print("\nLangGraph response:")
        lg_response = langgraph_tester.simple_qa(question)
        print(f"Response: {lg_response}")
        lg_precision, lg_recall, lg_f1 = calculate_f1_score(reference, lg_response)
        print(f"LangGraph F1 Score: {lg_f1:.2f} (Precision: {lg_precision:.2f}, Recall: {lg_recall:.2f})")

if __name__ == "__main__":
    run_comparison_tests()