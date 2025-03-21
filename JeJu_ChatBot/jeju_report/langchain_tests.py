from langchain_community.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from transformers import pipeline
from utils import performance_tracker as tracker
from model_config import ModelConfig

class LangChainTester:
    def __init__(self):
        self.model_config = ModelConfig()
        # HuggingFace 파이프라인 설정
        self.pipe = pipeline(
            "text-generation",
            model=self.model_config.model,
            tokenizer=self.model_config.tokenizer,
            max_new_tokens=75,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            device=self.model_config.device
        )
        self.llm = HuggingFacePipeline(pipeline=self.pipe)
    
    @tracker.measure_performance("LangChain")
    def simple_qa(self, question: str) -> str:
        try:
            prompt = PromptTemplate(
                input_variables=["question"],
                template="Answer the following question directly and concisely:\nQuestion: {question}\nAnswer:"
            )
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = chain.run(question=question)
            return response.split('.')[0].strip() + '.'
        except Exception as e:
            print(f"Error in LangChain simple_qa: {e}")
            return f"Error occurred: {str(e)}"
    
    @tracker.measure_performance("LangChain")
    def multi_step_reasoning(self, question: str) -> str:
        try:
            # 첫 번째 단계: 문제 분석
            analysis_prompt = PromptTemplate(
                input_variables=["question"],
                template="Provide a brief analysis of this question:\nQuestion: {question}\nAnalysis:"
            )
            analysis_chain = LLMChain(llm=self.llm, prompt=analysis_prompt)
            analysis = analysis_chain.run(question=question)
            
            # 두 번째 단계: 최종 답변
            answer_prompt = PromptTemplate(
                input_variables=["analysis", "question"],
                template="Based on this analysis, provide a concise answer:\nAnalysis: {analysis}\nQuestion: {question}\nAnswer:"
            )
            answer_chain = LLMChain(llm=self.llm, prompt=answer_prompt)
            final_answer = answer_chain.run(analysis=analysis, question=question)
            
            return final_answer.split('\n')[0].strip()
        except Exception as e:
            print(f"Error in LangChain multi_step_reasoning: {e}")
            return f"Error occurred: {str(e)}"