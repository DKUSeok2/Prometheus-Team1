from langgraph.graph import Graph
from utils import performance_tracker as tracker  # 각 파일의 상단에 추가
from model_config import ModelConfig
from typing import Dict, TypedDict, Annotated


class State(TypedDict):
    question: str
    response: str
    analysis: str
    final_answer: str

class LangGraphTester:
    def __init__(self):
        self.model_config = ModelConfig()
    
    @tracker.measure_performance("LangGraph")
    def simple_qa(self, question: str) -> str:
        def process_question(state: Dict) -> Dict:
            try:
                prompt = (
                    "Answer the following question directly and concisely:\n"
                    f"Question: {state['question']}\n"
                    "Answer:"
                )
                response = self.model_config.generate_response(prompt, max_new_tokens=50)
                # 첫 번째 문장만 추출
                response = response.split('.')[0].strip() + '.'
                return {
                    "question": state["question"],
                    "response": response,
                    "output": response
                }
            except Exception as e:
                print(f"Error in process_question: {e}")
                return {
                    "question": state["question"],
                    "response": str(e),
                    "output": str(e)
                }

        # 워크플로우 구성
        workflow = Graph()
        workflow.add_node("process", process_question)
        workflow.set_entry_point("process")
        workflow.set_finish_point("process")  # 종료 지점 명시
        
        # 그래프 컴파일
        chain = workflow.compile()
        
        try:
            # 실행
            result = chain.invoke({"question": question})
            return result.get("output", "No response generated")
        except Exception as e:
            print(f"Error in simple_qa chain: {e}")
            return f"Error occurred: {str(e)}"
    
    @tracker.measure_performance("LangGraph")
    def multi_step_reasoning(self, question: str) -> str:
        def analyze_question(state: Dict) -> Dict:
            try:
                prompt = (
                    "Provide a brief analysis of this question:\n"
                    f"Question: {state['question']}\n"
                    "Analysis:"
                )
                analysis = self.model_config.generate_response(prompt, max_new_tokens=75)
                return {
                    "question": state["question"],
                    "analysis": analysis,
                    "output": analysis
                }
            except Exception as e:
                print(f"Error in analyze_question: {e}")
                return {
                    "question": state["question"],
                    "analysis": str(e),
                    "output": str(e)
                }

        def provide_answer(state: Dict) -> Dict:
            try:
                prompt = (
                    "Based on this analysis, provide a concise answer:\n"
                    f"Analysis: {state['analysis']}\n"
                    f"Question: {state['question']}\n"
                    "Answer:"
                )
                answer = self.model_config.generate_response(prompt, max_new_tokens=75)
                # 첫 번째 문단만 추출
                answer = answer.split('\n')[0].strip()
                return {
                    "question": state["question"],
                    "analysis": state["analysis"],
                    "final_answer": answer,
                    "output": answer
                }
            except Exception as e:
                print(f"Error in provide_answer: {e}")
                return {
                    "question": state["question"],
                    "analysis": state.get("analysis", ""),
                    "final_answer": str(e),
                    "output": str(e)
                }

        # 워크플로우 구성
        workflow = Graph()
        workflow.add_node("analyze", analyze_question)
        workflow.add_node("answer", provide_answer)
        workflow.add_edge("analyze", "answer")
        workflow.set_entry_point("analyze")
        workflow.set_finish_point("answer")  # 종료 지점 명시
        
        # 그래프 컴파일
        chain = workflow.compile()
        
        try:
            # 실행
            result = chain.invoke({"question": question})
            return result.get("output", "No response generated")
        except Exception as e:
            print(f"Error in multi_step_reasoning chain: {e}")
            return f"Error occurred: {str(e)}"
